from typing import Tuple
from torch.utils.data import Dataset, DataLoader, random_split
import os
from pathlib import Path
from collections import namedtuple
import numpy as np
from gensim.models import word2vec
from gensim.models import KeyedVectors, Word2Vec
from tempfile import TemporaryFile, NamedTemporaryFile
import pandas as pd
import torch
import torchtext
from torchtext.data import Field, LabelField, RawField, TabularDataset, Dataset
from torchtext.vocab import GloVe, Vectors, Vocab
from torchtext.data import Iterator, BucketIterator



def array_to_text(array, ix2word):
    text = [ix2word[ix] for ix in array]
    text = filter(lambda x: x not in ("</s>", "<START>", "<EOP>",), text)
    text = "".join(text)
    return text


def get_dataset_and_vocab(
        train_path="/data/users/wangyuanzheng/projects/ucas_DL/4-text_classification/data/train.txt",
        valid_path="/data/users/wangyuanzheng/projects/ucas_DL/4-text_classification/data/validation.txt",
        test_path="/data/users/wangyuanzheng/projects/ucas_DL/4-text_classification/data/test.txt",
        w2v_path="/data/users/wangyuanzheng/projects/ucas_DL/4-text_classification/data/wiki_word2vec_50.bin",
        word_embedding_dim=50,
        word_min_count=5,
        train_percent=1.,
        valid_percent=1.,
        test_percent=1.,
        truncate_len=120,
        pad_token="</s>",
        unk_token="<unk>",
) -> Tuple[TabularDataset, TabularDataset, TabularDataset, Vocab]:

    # TextCNN的输入，需要是定长的
    def truncate(sentence):
        sentence += [pad_token] * truncate_len
        sentence = sentence[:truncate_len]
        return sentence

    # 创建fields
    # 与Field相比，LabelField不会额外添加<unk> <pad>等token
    Field_LABEL = LabelField(sequential=False, dtype=torch.long, use_vocab=False, preprocessing=int)
    # Field_TEXT = Field(sequential=True, use_vocab=True, lower=False, pad_token=pad_token, unk_token=unk_token, tokenize=lambda s: s.split(" "))
    Field_TEXT = Field(sequential=True, use_vocab=True, lower=False, pad_token=pad_token, unk_token=unk_token,
                       tokenize=lambda s: s.split(" "), preprocessing=truncate, batch_first=True)  # preprocessing在tokenize之后

    fields = [
        ('label', Field_LABEL),
        ('text', Field_TEXT),
    ]

    # 创建数据集
    dataset_train = TabularDataset(path=train_path, format="TSV", fields=fields)
    dataset_valid = TabularDataset(path=valid_path, format="TSV", fields=fields)
    dataset_test = TabularDataset(path=test_path, format="TSV", fields=fields)

    # 划分数据集
    if train_percent < 1.:
        dataset_train.examples = dataset_train.examples[:int(train_percent * len(dataset_train))]
    if valid_percent < 1.:
        dataset_valid.examples = dataset_valid.examples[:int(valid_percent * len(dataset_valid))]
    if test_percent < 1.:
        dataset_test.examples = dataset_test.examples[:int(test_percent * len(dataset_test))]

    # 构建词表
    Field_TEXT.build_vocab(dataset_train, min_freq=word_min_count)

    # 加载词向量
    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    word2vec_vectors = []
    for token, idx in Field_TEXT.vocab.stoi.items():
        if token in w2v.vocab.keys():
            word2vec_vectors.append(torch.FloatTensor(w2v[token]))
        else:
            word2vec_vectors.append(torch.zeros(word_embedding_dim))
    Field_TEXT.vocab.set_vectors(Field_TEXT.vocab.stoi, word2vec_vectors, word_embedding_dim)
    assert all(w2v["我"] == Field_TEXT.vocab.vectors[Field_TEXT.vocab.stoi["我"]].numpy())

    # 返回值
    return dataset_train, dataset_valid, dataset_test, Field_TEXT.vocab


if __name__=="__main__":
    dataset_train, dataset_valid, dataset_test,  vocab= get_dataset_and_vocab(word_min_count=5)
    it = iter(Iterator(dataset_test, batch_size=5, shuffle=False))
    batch = next(it)
    print(array_to_text(batch.text[0], vocab.itos))
