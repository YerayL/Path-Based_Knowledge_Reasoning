import csv
from functools import partial
from torchtext.data import *
from transformers import BertTokenizer
import dill
import os
import timeit
from random import sample
from tqdm import tqdm

def load_iters(root="chains_of_reasoning_eacl_dataset", tokenizer_name="bert",
               bert_pretrained_model="bert-large-uncased", batch_size=8,
               batch_first=False, padding_to=0, padding_to2=0, sort_within_batch=True, device="cpu"):

    def padding(batch, vocab, to=padding_to):
        pass

    def tokenizer(text=""):
        pass

    print(f"unk: 100 pad: 0")
    ENTITY = Field(include_lengths=False,
                 init_token=None,
                 eos_token=None,
                 pad_token=None,
                 unk_token=None,
                 use_vocab=False,
                 lower=None,
                 batch_first=batch_first,
                 postprocessing=partial(padding, to=padding_to),
                 tokenize=tokenizer)
    RELATION = Field(sequential=False, unk_token=None, use_vocab=False)
    LABEL = Field(sequential=False, unk_token=None, use_vocab=False)








if __name__ == '__main__':
    pass
