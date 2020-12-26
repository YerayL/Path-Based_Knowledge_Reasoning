import csv
from functools import partial
from torchtext.data import *
from transformers import BertTokenizer
import dill
import os
import timeit
from random import sample
from tqdm import tqdm


def load_split_datasets(path, fields):
    with open(os.path.join(path, 'train.dill'), 'rb') as f:
        train_examples = dill.load(f)
    # with open(os.path.join(path, 'valid.dill'), 'rb') as f:
    #     dev_examples = dill.load(f)
    with open(os.path.join(path, 'test.dill'), 'rb') as f:
        test_examples = dill.load(f)
    train = Dataset(examples=train_examples, fields=fields)
    # dev = Dataset(examples=dev_examples, fields=fields)
    test = Dataset(examples=test_examples, fields=fields)
    return train, test, test
    # return train


def load_test_datasets(fields):
    test_dict = dict()
    path3 = './dataset/pppp'
    for root, dirs, files in os.walk(path3, topdown=True):
        for _, file in enumerate(tqdm(files)):
            with open(os.path.join(root, file), 'rb') as f:
                name = str(file).split('_test')[0]
                test_examples = dill.load(f)
                test = Dataset(examples=test_examples, fields=fields)
                test_dict[name] = test
    return test_dict


def dump_split_datasets(data, path):
    with open(path, 'wb') as f:
        dill.dump(data, f)


def load_iters(root="chains_of_reasoning_eacl_dataset", tokenizer_name="bert",
               bert_pretrained_model="bert-large-uncased", batch_size=8,
               batch_first=False, padding_to=0, padding_to2=0, sort_within_batch=True, device="cpu"):
    bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)

    def padding(batch, vocab, to=padding_to):
        if not padding_to or padding_to <= 0 or tokenizer_name == "bert":
            return batch
        pad_idx = vocab.stoi['<pad>']
        for idx, ex in enumerate(batch):
            if len(ex) < padding_to:
                batch[idx] = ex + [pad_idx] * (padding_to - len(ex))
        return batch

    def tokenizer(text=""):
        text = text.split('###')
        # 优先选短路径。
        query = text[0:1]
        path = text[1:]
        text = query + path
        if len(text) >= padding_to2:
            path = sorted(path, key=lambda i:len(i))
            input = query + path[:padding_to2-1]
        else:
            pad_count = padding_to2 - len(text)
            while pad_count > 0:
                text.append('')
                pad_count -= 1
            input = text

        # if len(path) < padding_to2 - 1:
        #     pad_count = padding_to2 - 1 - len(path)
        #     while pad_count > 0:
        #         path.append('')
        #         pad_count -= 1
        # else:
        #     path = sample(path, padding_to2 - 1)
        # input = query + path

        # if len(text) < padding_to2:
        #     pad_count = padding_to2 - len(text)
        #     while pad_count > 0:
        #         text.append('')
        #         pad_count -= 1
        # input = sample(text, padding_to2)
        tokens = []
        masks = []
        for i in input:
            token = bert_tokenizer.encode_plus(i, add_special_tokens=True, return_attention_mask=True,
                                               max_length=padding_to, pad_to_max_length=True)
            tokens.append(token["input_ids"])
            masks.append(token["attention_mask"])
        return [tokens, masks]

    print(f"unk: 100 pad: 0")
    use_vocab = False
    TEXT = Field(include_lengths=True,
                 init_token=None,
                 eos_token=None,
                 pad_token='[PAD]',
                 unk_token='[UNK]',
                 use_vocab=use_vocab,
                 lower=None,
                 batch_first=batch_first,
                 postprocessing=partial(padding, to=padding_to),
                 tokenize=tokenizer)
    LABEL = Field(sequential=False, unk_token=None, use_vocab=False)
    QUERY = Field(sequential=False, unk_token=None, pad_token=None, use_vocab=False)

    # _train, _dev, _test = TabularDataset.splits(path="./" + root, root="./", train="testdata2.csv",
    #                                             validation='testdata2.csv', test="testdata2.csv",
    #                                             format='csv', skip_header=False,
    #                                             fields=[("text", TEXT), ("label", LABEL)],
    #                                             csv_reader_params={"quoting": csv.QUOTE_NONE, "delimiter": "\t"})

    # start = timeit.default_timer()
    # print('划分数据集1……')

    # ============================================== #
    # path0 = './dataset/ooo'
    # path3 = './dataset/ppp11'
    # path4 = './dataset/train.csv'
    # _train = TabularDataset(path=path4, format='csv', skip_header=False,
    #                        fields=[("text", TEXT), ("label", LABEL)],
    #                        csv_reader_params={"quoting": csv.QUOTE_NONE, "delimiter": "\t"})
    # dump_split_datasets(_train.examples, './dataset/train.dill')
    # for root, dirs, files in os.walk(path0, topdown=True):
    #     for _, file in enumerate(tqdm(files)):
    #         path1 = os.path.join(root, file)
    #         print('\n划分'+file+'数据集')
    #         _test = TabularDataset(path=path1, format='csv', skip_header=False,
    #                                fields=[("text", TEXT), ("label", LABEL)],
    #                                csv_reader_params={"quoting": csv.QUOTE_NONE, "delimiter": "\t"})
    #         print('\n划分' + file + '数据集完成！')
    #         print('\n保存' + file + '数据集')
    #         rel = file.split('test_')[0] + 'test'
    #         dump_split_datasets(_test.examples, os.path.join(path3, file+'.dill'))
    #         print('\n保存' + rel + '完成')
    # ============================================== #
    #
    _train, _dev, _test = TabularDataset.splits(path="./" + root, root="./", train="train.csv",
                                                validation='valid.csv', test="test.csv",
                                                format='csv', skip_header=False,
                                                fields=[("label", LABEL),("query", QUERY), ("text", TEXT)],
                                                csv_reader_params={"quoting": csv.QUOTE_NONE, "delimiter": "\t"})

    # _test = TabularDataset(path="./" + root + "/test.csv", format='csv', skip_header=False,
    #                        fields=[("text", TEXT), ("label", LABEL)],
    #                        csv_reader_params={"quoting": csv.QUOTE_NONE, "delimiter": "\t"})
    # print('完成……')
    # end = timeit.default_timer()
    # print('splited dataset cost time:', end - start)
    # start = timeit.default_timer()
    # print('保存预处理文件……')
    # dump_split_datasets(_train.examples, "./" + root + "/train.dill")
    # dump_split_datasets(_dev.examples, "./"+root+"/dev.dill")
    # dump_split_datasets(_test.examples, "./" + root + "/test.dill")
    # print('保存预处理文件完毕！')
    # end = timeit.default_timer()
    # print('dump data cost time:', end - start)

    # start = timeit.default_timer()
    # print('读取预处理文件……')
    # _train, _dev, _test = load_split_datasets("./"+root+"/", fields=[("text", TEXT), ("query", QUERY), ("label", LABEL)])
    # _train = load_split_datasets("./" + root + "/", fields=[("text", TEXT),  ("label", LABEL)])
    # test_dict = load_test_datasets(fields=[("text", TEXT), ("label", LABEL)])
    # print('读取预处理文件完毕！')
    # end = timeit.default_timer()
    # print('load data cost time:', end - start)

    sort_key = lambda x: len(x.text)
    train_iter = BucketIterator(_train,
                                batch_size=batch_size,
                                train=True,
                                repeat=False,
                                shuffle=True,
                                sort_within_batch=sort_within_batch,
                                sort_key=(sort_key if sort_within_batch else None),
                                device=device)
    # dev_iter = BucketIterator(_dev, batch_size=1, train=False, repeat=False, shuffle=True,
    #                           sort_within_batch=sort_within_batch, sort_key=(sort_key if sort_within_batch else None),
    #                           device=device)
    # test_iter_dict = dict()
    # for key, value in test_dict.items():
    #     test_iter_dict[key] = BucketIterator(value, batch_size=64, train=False, repeat=False, shuffle=True,
    #                            sort_within_batch=False, sort_key=lambda x: len(x.text), device=device)
    test_iter = BucketIterator(_test, batch_size=16, train=False, repeat=False, shuffle=True,
                               sort_within_batch=False, sort_key=lambda x: len(x.text), device=device)
    # test_iter_dict['_time_event_locations']=test_iter
    return train_iter, test_iter, test_iter, TEXT, LABEL


if __name__ == '__main__':
    load_iters(root="dataset/_time_event_locations", tokenizer_name="bert",
               bert_pretrained_model="bert-base-uncased", batch_size=16,
               batch_first=True, padding_to=30, padding_to2=36, sort_within_batch=True, device="cuda")
