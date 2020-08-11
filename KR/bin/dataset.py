import csv
from functools import partial
from torchtext.data import *
import dill
import os
import timeit
from random import sample
from tqdm import tqdm


def load_iters(root, tokenizer_name="bert",
               bert_pretrained_model="bert-large-uncased", batch_size=256,
               batch_first=True, device="cuda"):

    def tokenizer_ENTITY(text):
        entity_path_list = text.split(';')
        path_list = []
        for i in entity_path_list:
            entity_list = i.split(' ')
            path = []
            for j in entity_list:
                entity = j.split(',')
                path.append([int(x) for x in entity])
            path_list.append(path)
        return path_list

    def tokenizer_RELATION(text):
        relation_path_list = text.split(';')
        path_list = []
        for i in relation_path_list:
            relation_list = i.split(' ')
            path = []
            for j in relation_list:
                path.append(int(j))
            path_list.append(path)
        return path_list


    ENTITY = Field(
                 pad_token=None,
                 unk_token=None,
                 use_vocab=False,
                 batch_first=batch_first,
                 tokenize=tokenizer_ENTITY)
    RELATION = Field(
                 pad_token=None,
                 unk_token=None,
                 use_vocab=False,
                 batch_first=batch_first,
                 tokenize=tokenizer_RELATION)
    LABEL = Field(sequential=False, unk_token=None, pad_token=None, use_vocab=False, is_target=True)
    QUERRY = Field(sequential=False, unk_token=None, pad_token=None, use_vocab=False)

    # _train = TabularDataset(
    #     path='./ChainsofReasoning/examples/data_small_output/_aviation_airport_serves/train/train.csv',
    #     format='csv', fields=[("label", LABEL), ("entity", ENTITY), ("relation", RELATION), ("query", QUERRY)],
    #     csv_reader_params={"quoting": csv.QUOTE_NONE, "delimiter": "\t"})


    # _train, _dev, _test = TabularDataset.splits(
    #     path="./data/data_output/_aviation_airport_serves", train="train.csv", validation='dev.csv', test="test.csv",
    #     format='csv', fields=[("label", LABEL), ("entity", ENTITY), ("relation", RELATION), ("pathnum", PATHNUM)],
    #     csv_reader_params={"quoting": csv.QUOTE_NONE, "delimiter": "\t"})

    _train = dict()
    _dev = dict()
    _test = dict()
    train_iter_dict = dict()
    dev_iter_dict = dict()
    test_iter_dict = dict()
    # for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'ChainsofReasoning/examples/data_small_output/'), topdown=True):
    for root, dirs, files in os.walk(root, topdown=True):
        for file in files:
            # if file.endswith('csv'):
            path_num = file[:-4].split('.')[-1]
            path = os.path.join(root, file)
            t = TabularDataset(
                path=path,
                format='csv',
                fields=[("label", LABEL), ("entity", ENTITY), ("relation", RELATION), ("query", QUERRY)],
                csv_reader_params={"quoting": csv.QUOTE_NONE, "delimiter": "\t"})
            #change this for linux, '\\' to '/'
            if root.split('/')[-1] == 'train':
                _train[path_num] = t
                Iter = Iterator(t, batch_size=batch_size, repeat=False, shuffle=True, device=device)
                train_iter_dict[path_num] = Iter
            elif root.split('/')[-1] == 'dev':
                _dev[path_num] = t
                Iter = Iterator(t, batch_size=1, repeat=False, shuffle=True, device=device)
                dev_iter_dict[path_num] = Iter
            elif root.split('/')[-1] == 'test':
                _test[path_num] = t
                Iter = Iterator(t, batch_size=1, repeat=False, shuffle=True, device=device)
                test_iter_dict[path_num] = Iter

    print('========================Dataset done!=======================')



    # train_iter = Iterator(_train, batch_size=batch_size, repeat=False, shuffle=True, device=device)
    # dev_iter = Iterator(_dev, batch_size=1, repeat=False, shuffle=True, device=device)
    # test_iter = Iterator(_test, batch_size=1, repeat=False, shuffle=True, device=device)

    return train_iter_dict, dev_iter_dict, test_iter_dict, ENTITY, RELATION, LABEL, QUERRY

if __name__ == '__main__':
    load_iters()
