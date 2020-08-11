# from argparse import ArgumentParser
from configargparse import ArgumentParser
import KR.opts as opts
import sys
from KR.utils.logging import init_logger, logger
from KR.utils.regularization import Regularization
from KR.bin.dataset import load_iters
from KR.bin.rnn import RNNModel
from KR.bin.rnn import Classifier

import torch
from torch import nn, optim
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from operator import itemgetter
# import pandas as pd
import os
import time
current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())







def Att_Model(opt):

    torch.manual_seed(opt.seed)
    logger.info("==========={}============".format(opt.root))

    # build data iter
    train_iter_dict, dev_iter_dict, test_iter_dict, ENTITY, RELATION, LABEL, PATHNUM = load_iters(root=opt.root,batch_size=opt.batch_size,
               batch_first=opt.batch_first, device=opt.device)

    # build model
    encoder = RNNModel(e_input_size=100, r_input_size=250, hidden_size=250, num_layers=1, bidirectional=False, batch_first=True, dropout=0,
                 mode="RNN")
    model = Classifier(encoder=encoder)
    loss_func = nn.NLLLoss(reduction='sum').to("cuda")

    # multi gpu
    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        model = nn.DataParallel(model)
    model = model.to("cuda")

    # L2 Regularization
    if opt.weight_decay > 0:
        reg_loss = Regularization(model, opt.weight_decay, p=2).to(opt.device)
    else:
        print("no regularization")

    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter(current_time, comment="Att-Model")

    # train
    for epoch in trange(opt.train_epochs, desc="Epoch"):
        model.train()
        ep_loss = 0
        for step, (path_num, train_iter) in enumerate(tqdm(train_iter_dict.items(), desc="Iteration_list")):
            path_num = int(path_num)
            for step2, batch in enumerate(train_iter):
                entity, relation, labels,  query = batch.entity, batch.relation, batch.label, batch.query
                y_p, y_predict = model(entity, relation, query, path_num)
                loss = loss_func(y_predict, labels)
                if opt.weight_decay > 0:
                    loss += reg_loss(model)
                print(f'loss:{loss}')
                ep_loss += loss.item()
                model.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), clip)  no need clip , I think.
                optimizer.step()
                writer.add_scalar('Train_dif_path_Loss', loss, step)
            if step % 100 == 0:
                tqdm.write("Epoch %d, Step %d, Loss %.2f" % (epoch, step, loss.item()))
        writer.add_scalar('Train_epoch,Loss', ep_loss, epoch)

        # evaluating for every epoch
        model.eval()
        predict_dict = dict()
        with torch.no_grad():
            predict_list = []
            for step, (path_num, dev_iter) in enumerate(tqdm(dev_iter_dict.items(), desc="Iteration_list")):
                path_num = int(path_num)
                for step2, batch in enumerate(dev_iter):
                    entity, relation, labels,  query = batch.entity, batch.relation, batch.label, batch.query
                    y_p, y_predict = model(entity, relation, query, path_num)
                    for i in zip(y_p[:, 1].tolist(), labels.tolist()):
                        predict_list.append(i)
            results = sorted(predict_list, key=itemgetter(0), reverse=True)
            count_1 = 0
            precision = float(0)
            for i, j in enumerate(tqdm(results)):
                if j[1] == 1:
                    count_1 += 1
                    precision += float(count_1 / (i + 1))
            AP = float(precision / count_1)
            print('_aviation_airport_serves', '关系AP：', AP)
            predict_dict['_aviation_airport_serves'] = AP
                # tqdm.write("Epoch %d, Accuracy %.3f" % (epoch, corr_num / (corr_num + err_num)))

    # predicting for the test data
    # model.eval()
    # with torch.no_grad():
    #     predicts = []
    #     for batch in test_iter:
    #         inputs, lens = batch.text
    #         outputs = model(inputs, lens)
    #         predicts.extend(outputs.argmax(1).cpu().numpy())
    #     test_data = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t')
    #     test_data["Sentiment"] = predicts
    #     test_data[['PhraseId', 'Sentiment']].set_index('PhraseId').to_csv('result.csv')


def _get_parser():
    parser = ArgumentParser(description='Att-model.py')

    opts.config_opts(parser)
    return parser

def main():
    parser = _get_parser()

    opt = parser.parse_args()
    init_logger(opt.log_file)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    Att_Model(opt)

if __name__ == '__main__':
    main()