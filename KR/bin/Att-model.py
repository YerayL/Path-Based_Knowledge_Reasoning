from argparse import ArgumentParser
import KR.opts as opts
import sys
from KR.utils.logging import init_logger, logger

import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch.optim import Adam
from tensorboardX import SummaryWriter
import pandas as pd
import os
import time
current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())

from torchtext import data
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors

from models import RNN, CNN




def Att_Model(opt):
    torch.manual_seed(opt.seed)
    dataset_name = opt.dataset_name
    logger.info("==========={}============".format(dataset_name))


    # build data iter
    train_iter, dev_iter, test_iter, TEXT, LABEL = \
        load_iters(root=args.dataset_name, tokenizer_name="bert",
                   bert_pretrained_model=args.pretrained_model, batch_size=args.batch_size,
                   batch_first=True, padding_to=args.padding_to, padding_to2=args.padding_to2, sort_within_batch=True,
                   device="cuda")
    vocab_size = len(TEXT.vocab.itos)

    
    # build model
    emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=250, padding_idx=0)
    encoder
    model

    if torch.cuda.device_count() > 1 and opt.mutil_gpu:
        model = nn.DataParallel(model)
    model = model.to("cuda")

    L2范式
    criterion
    optimizer

    writer = SummaryWriter(current_time, comment="rnn")
    # train
    for epoch in trange(opt.train_epochs, desc="Epoch"):
        model.train()
        ep_loss = 0
        for step, batch in enumerate(tqdm(train_iter, desc="Iteration")):
            (inputs, lens), labels = batch.text, batch.label
            outputs = model(inputs, lens)
            loss = loss_func(outputs, labels)
            ep_loss += loss.item()

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            writer.add_scalar('Train_Loss', loss, epoch)
            if step % 100 == 0:
                tqdm.write("Epoch %d, Step %d, Loss %.2f" % (epoch, step, loss.item()))

                # evaluating
                model.eval()
                with torch.no_grad():
                    corr_num = 0
                    err_num = 0
                    for batch in dev_iter:
                        (inputs, lens), labels = batch.text, batch.label
                        outputs = model(inputs, lens)
                        corr_num += (outputs.argmax(1) == labels).sum().item()
                        err_num += (outputs.argmax(1) != labels).sum().item()
                    tqdm.write("Epoch %d, Accuracy %.3f" % (epoch, corr_num / (corr_num + err_num)))

    # predicting
    model.eval()
    with torch.no_grad():
        predicts = []
        for batch in test_iter:
            inputs, lens = batch.text
            outputs = model(inputs, lens)
            predicts.extend(outputs.argmax(1).cpu().numpy())
        test_data = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t')
        test_data["Sentiment"] = predicts
        test_data[['PhraseId', 'Sentiment']].set_index('PhraseId').to_csv('result.csv')


def _get_parser():
    parser = ArgumentParser(description='Att-model.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser

def main():
    parser = _get_parser()

    opt = parser.parse_args()
    init_logger(opt.log_file)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    Att_Model(opt)

if __name__ == '__main__':
    main()