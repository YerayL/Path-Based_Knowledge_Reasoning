import argparse
import logging
import sys

import tqdm
import torch
from torch import nn, optim
from src.datasets import load_iters

from src.model import Classifier, BertEncoder
from src.train import train, evaluate,eval
from src.CNN import TextCNNEncoder, OriEncoder
from src.regularization import Regularization
import os

from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)

# 日志模块
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)




def main(args):
    dataset_name = args.dataset_name
    logger.info("==========={}============".format(dataset_name))

    train_iter, dev_iter, test_iter, TEXT, LABEL = \
        load_iters(root=args.dataset_name, tokenizer_name="bert",
                   bert_pretrained_model=args.pretrained_model, batch_size=args.batch_size,
                   batch_first=True, padding_to=args.padding_to, padding_to2=args.padding_to2, sort_within_batch=True, device="cuda")
    emb = nn.Embedding(num_embeddings=30522, embedding_dim=250, padding_idx=0)
    if args.CNN:
        encoder0 = TextCNNEncoder(emb=emb, out_dim=args.out_dim)
    else:
        encoder0 = OriEncoder(emb=emb, out_dim=args.out_dim)
    encoder1 = BertEncoder(out_dim=args.out_dim, dropout_p=args.dropout_p, pretrained_model=args.pretrained_model)
    encoder = [encoder0, encoder1]
    model = Classifier(encoder=encoder, out_class=args.out_class, dropout_p=args.dropout_p)

    # from torchsummary import summary
    # summary(model, input_size=(20, 31, 30))

    # 多gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to("cuda")

    if args.weight_decay > 0:
        reg_loss = Regularization(model, args.weight_decay, p=2 ).to(args.device)
    else:
        print("no regularization")

    criterion = nn.NLLLoss(reduction='sum').to("cuda")
    # 限制Bert内部学习率
    # encoder = list(map(id, model.encoder.parameters()))
    # out = list(map(id, model.out.parameters()))
    if torch.cuda.device_count() > 1:
        params = [
            {"params": model.module.encoder.parameters(), "lr": 0.00005},
            {"params": model.module.cnn_encoder.parameters(), "lr": 0.001},
            {"params": model.module.out.parameters(), "lr": 0.001},
        ]
        print(params)
    else:
        params = [
            {"params": model.encoder.parameters(), "lr": 0.00001},
            {"params": model.cnn_encoder.parameters(), "lr": 0.001},
            {"params": model.out.parameters(), "lr": 0.001},
        ]
        print(params)

    optimizer = optim.Adam(params, lr=0.00001)
    # optimizer = optim.Adam(model.parameters(), lr=0.00001)

    def model_train(e, padding_to2):
        e.model.zero_grad()
        (x, x_length),query, label = e.batch.text,e.batch.query, e.batch.label
        y_p, y_predict = e.model(x, padding_to2,query)
        loss = e.criterion(y_predict, label)
        if args.weight_decay > 0:
            loss = loss + reg_loss(model)
        loss.backward()
        e.optimizer.step()
        e.progress.set_postfix(loss=loss.item())

    def model_eval(e,padding_to2=50):
        (x, x_length), label = e.batch.text, e.batch.label
        y_predict = e.model(x, padding_to2)
        predict_list = []
        for i in zip(y_predict[:, 1].tolist(), label.tolist()):
            predict_list.append(i)
        return predict_list

    # if args.train:
    #     if args.continue_train:
    #         results = train(train_iter, dev_iter, model, criterion, optimizer, max_epoch=args.max_epoch, max_iters=args.max_iters,
    #               save_every=args.save_every, continue_train=args.continue_train,device=args.device, handler=model_train,
    #               patience=args.patience, padding_to2=args.padding_to2)
    #
    #     else:
    #         results = train(train_iter, dev_iter, model, criterion, optimizer,max_epoch=args.max_epoch, max_iters=args.max_iters,
    #               save_every=args.save_every,device=args.device, handler=model_train,
    #               patience=args.patience,padding_to2=args.padding_to2)


    if args.test:
        evaluate(dev_iter, model, model_location=args.model_location,padding_to2=args.padding_to2,
                 criterion=criterion, device=args.device)

        # AP = eval(test_iter,model,model_location=args.model_location,criterion=criterion, device=args.device, handler=model_eval)
        # print("AP: ", AP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="path bert")

    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--continue_train", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=True)
    parser.add_argument("--CNN", type=bool, default=False)
    parser.add_argument("--out_dim", type=int, default=250)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--out_class", type=int, default=1)
    parser.add_argument("--pretrained_model", type=str, default="bert-base-chinese")
    parser.add_argument("--patience", type=int, default=None)
    # parser.add_argument("--dataset_name", type=str, default="chains_of_reasoning_eacl_dataset/_broadcast_content_artist")
    # parser.add_argument("--dataset_name", type=str, default="chains_of_reasoning_eacl_dataset/_architecture_structure_address")
    parser.add_argument("--dataset_name", type=str, default="data_0.01")
    parser.add_argument("--max_iters", type=int, default=12200001)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--save_every", type=int, default=12200000)
    parser.add_argument("--padding_to", type=int, default=120)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--padding_to2", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_location", type=str, default="checkpoint/best_0.51.pt")

    # exp_list = ["dataset/_time_event_locations"]
    #DataParallel.2000.pt
    #Classifier

    args = parser.parse_args()

    # for i in exp_list:
    #     args.dataset_name = i
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    print('=========================================================')
    print(args)
    print('=========================================================')
    main(args)
