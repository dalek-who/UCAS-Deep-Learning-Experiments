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



def array_to_poem(array, ix2word):
    poem = [ix2word[ix] for ix in array]
    poem = filter(lambda x: x not in ("</s>", "<START>", "<EOP>",), poem)
    poem = "".join(poem)
    return poem

# todo：参数表
# input_path = "/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/tang.npz"
# predict_input_path = "/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/predict_input.txt"
# preprocessed_data_path = "/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/preprocessed.tsv"
# word_embedding_dim = 128
# window_size = 5
# word_min_count = 3
# train_percent = 0.1
# valid_percent = 0.1
# test_percent = 0.1
# pad_token = "</s>"
# unk_token = "<unk>"
# input_init_token = "<INPUT>"
# output_init_token = "<OUTPUT>"
# output_eos_token = "<EOP>"

def get_dataset_and_vocab(
        input_path="/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/tang.npz",
        preprocessed_data_path="/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/preprocessed.tsv",
        predict_input_path="/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/predict_input.txt",
        word_embedding_dim=128,
        window_size=5,
        word_min_count=3,
        train_percent=0.1,
        valid_percent=0.1,
        test_percent=0.1,
        pad_token="</s>",
        unk_token="<unk>",
        input_init_token="<INPUT>",
        output_init_token="<OUTPUT>",
        output_eos_token="<EOP>",
) -> Tuple[TabularDataset, TabularDataset, TabularDataset, TabularDataset, Vocab, Vocab]:

    # 导入原始数据集
    raw_data = np.load(input_path, allow_pickle=True)
    data = raw_data["data"]
    ix2word = raw_data['ix2word'].item()
    word2ix = raw_data["word2ix"].item()

    START_ix = word2ix["<START>"]
    EOP_ix = word2ix["<EOP>"]
    white_space_ix = word2ix['</s>']
    special_symbols = ['、', '●', '〖', '〗', '○', '/', '：', 'Ｂ', '□', '「', '『', '』', '；', '［', '（', '」',
                       '］']  # 有这些特殊符号的诗有噪音
    special_symbols_ix = np.array([word2ix[word] for word in special_symbols])

    # (50, 34, 66, 26): 五言八句，七言四句，七言八句，五言四句
    sentence_num = {50: 8, 34: 4, 66: 8, 26: 4}  # key：全诗长度（包括<START>,<EOP>)，value：全诗一共几句
    sentence_len = {50: 5, 34: 7, 66: 7, 26: 5}  # key：全诗长度（包括<START>,<EOP>)，value：每句多少字（不包括标点、<START>,<EOP>）
    # allow_len = (50, 34, 66, 26)  # 全诗许可的长度（包括<START>,<EOP>)
    allow_len = (50, 34, )  # 全诗许可的长度（包括<START>,<EOP>)
    preprocessed_data_list = []
    poem_to_word_list = []  # 将每首诗都转化成词列表
    for array_example in data:
        input_sentence_num = 1  # 从诗中选几个句子作为输入
        example = dict()
        real_length = np.sum(array_example != white_space_ix)  # 去掉空白符后的实际长度。长度里包括<START>和<EOP>
        has_special_symbol = len(np.intersect1d(special_symbols_ix, array_example)) > 0  # 是否有特殊符号
        if not real_length in allow_len or has_special_symbol:
            continue
        # 统计诗的句子数、句子里的词数
        example["sentence_num"] = sentence_num[real_length]
        example["sentence_len"] = sentence_len[real_length]
        poem = array_to_poem(array_example, ix2word)
        # 创建输入输出
        # example["input"] = "<START> " + " ".join(poem[:example["sentence_len"]+1])
        # example["input_len"] = example["sentence_len"] + 2
        example["input_len"] = input_sentence_num * (example["sentence_len"] + 1)
        example["input"] = poem[: example["input_len"]]  # 原文形式的诗的输入
        # example["output"] = " ".join(poem[example["sentence_len"]+1:]) + " <EOP>"
        # example["output_len"] = (example["sentence_len"]+1) * (example["sentence_num"]-1) + 1
        example["output_len"] = (example["sentence_len"] + 1) * (example["sentence_num"] - input_sentence_num)
        example["output"] = poem[example["input_len"]:]  # 原文形式的诗的输出
        assert example["input_len"] + example["output_len"] + 2 == real_length, (
        example["input_len"], example["output_len"], real_length)
        preprocessed_data_list.append(example)
        # space_split_poem = f"<START> {' '.join(poem)} <EOP>"
        poem_to_word_list.append(list(poem))  # 将每首诗分割成单个字的列表，用于后面的词向量训练
    df = pd.DataFrame(preprocessed_data_list)

    # 训练word2vec，torchtext需要用训练好的word2vec构建词表
    w2v_model = word2vec.Word2Vec(poem_to_word_list, size=word_embedding_dim, window=window_size,
                                  min_count=word_min_count, workers=4)

    # 创建fields
    NUMBER = Field(sequential=False, dtype=torch.long, use_vocab=False, preprocessing=int)
    SENTENCE_INPUT = Field(
        sequential=True, use_vocab=True, lower=False, pad_token=pad_token, unk_token=unk_token,
        init_token=input_init_token,
        tokenize=lambda s: list(s), batch_first=False, include_lengths=True)  # include_lengths可以统计加了上面的特殊符号后的长度
    SENTENCE_OUTPUT = Field(
        sequential=True, use_vocab=True, lower=False, pad_token=pad_token, unk_token=unk_token,
        init_token=output_init_token,
        eos_token=output_eos_token, tokenize=lambda s: list(s), batch_first=False, include_lengths=True)
    # SENTENCE_LEN = Field(sequential=False, dtype=torch.long, use_vocab=False, preprocessing=lambda ls: len(ls))

    fields = [
        ('sentence_num', NUMBER),
        ('sentence_len', NUMBER),
        (None, None),  # ('input_len', NUMBER),
        ('input', SENTENCE_INPUT),
        (None, None),  # ('output_len', NUMBER),
        ('output', SENTENCE_OUTPUT),
    ]

    # 创建数据集
    # with NamedTemporaryFile() as temp:  # 用临时文件创建
    #     df.to_csv(temp.name, sep="\t", index=False)
    #     dataset = TabularDataset(path=temp.name, format="TSV", fields=fields, skip_header=True)
    with open(preprocessed_data_path, "w") as f:
        df.to_csv(f, sep="\t", index=False)
    dataset = TabularDataset(path=preprocessed_data_path, format="TSV", fields=fields, skip_header=True)

    # 划分数据集。必须在这里就划分好，之后才能建立训练集的词表
    used_percent = train_percent + valid_percent + test_percent
    if used_percent < 1.:
        dataset_used, dataset_unused = dataset.split(split_ratio=[used_percent, 1 - used_percent])  # 划分使用的和不使用的
    else:
        dataset_used = dataset
    dataset_train, dataset_valid, dataset_test = dataset_used.split(
        split_ratio=[percent / used_percent for percent in [train_percent, valid_percent, test_percent]])
    dataset_predict = TabularDataset(path=predict_input_path, format="TSV", fields=[('input', SENTENCE_INPUT)])

    # 利用word2vec创建词表vocab
    for field in (SENTENCE_INPUT, SENTENCE_OUTPUT):
        # field.build_vocab(dataset_train, min_freq=word_min_count)
        field.build_vocab(dataset, min_freq=word_min_count)  # 构建个比较大的词表，减少unk
        word2vec_vectors = []
        for token, idx in field.vocab.stoi.items():
            if token in w2v_model.wv.vocab.keys():
                word2vec_vectors.append(torch.FloatTensor(w2v_model[token]))
            else:
                word2vec_vectors.append(torch.zeros(word_embedding_dim))
        field.vocab.set_vectors(field.vocab.stoi, word2vec_vectors, word_embedding_dim)
        assert all(w2v_model["我"] == field.vocab.vectors[field.vocab.stoi["我"]].numpy())

    # 返回值
    return dataset_train, dataset_valid, dataset_test, dataset_predict, SENTENCE_INPUT.vocab, SENTENCE_OUTPUT.vocab


def get_dataset_and_vocab_language_model(
        input_path="/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/tang.npz",
        preprocessed_data_path="/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/preprocessed.tsv",
        predict_input_path="/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/predict_input.txt",
        word_embedding_dim=128,
        window_size=5,
        word_min_count=3,
        train_percent=0.1,
        valid_percent=0.1,
        test_percent=0.1,
        pad_token="</s>",
        unk_token="<unk>",
        input_init_token="<INPUT>",
        output_init_token="<OUTPUT>",
        output_eos_token="<EOP>",
) -> Tuple[TabularDataset, TabularDataset, TabularDataset, TabularDataset, Vocab, Vocab]:

    # 导入原始数据集
    raw_data = np.load(input_path, allow_pickle=True)
    data = raw_data["data"]
    ix2word = raw_data['ix2word'].item()
    word2ix = raw_data["word2ix"].item()

    START_ix = word2ix["<START>"]
    EOP_ix = word2ix["<EOP>"]
    white_space_ix = word2ix['</s>']
    special_symbols = ['、', '●', '〖', '〗', '○', '/', '：', 'Ｂ', '□', '「', '『', '』', '；', '［', '（', '」',
                       '］']  # 有这些特殊符号的诗有噪音
    special_symbols_ix = np.array([word2ix[word] for word in special_symbols])

    # (50, 34, 66, 26): 五言八句，七言四句，七言八句，五言四句
    sentence_num = {50: 8, 34: 4, 66: 8, 26: 4}  # key：全诗长度（包括<START>,<EOP>)，value：全诗一共几句
    sentence_len = {50: 5, 34: 7, 66: 7, 26: 5}  # key：全诗长度（包括<START>,<EOP>)，value：每句多少字（不包括标点、<START>,<EOP>）
    allow_len = (50, 34, 66, 26)  # 全诗许可的长度（包括<START>,<EOP>)
    # allow_len = (50, 34, )  # 全诗许可的长度（包括<START>,<EOP>)
    preprocessed_data_list = []
    poem_to_word_list = []  # 将每首诗都转化成词列表
    for array_example in data:
        input_sentence_num = 1  # 从诗中选几个句子作为输入
        example = dict()
        real_length = np.sum(array_example != white_space_ix)  # 去掉空白符后的实际长度。长度里包括<START>和<EOP>
        has_special_symbol = len(np.intersect1d(special_symbols_ix, array_example)) > 0  # 是否有特殊符号
        if not real_length in allow_len or has_special_symbol:
            continue
        # 统计诗的句子数、句子里的词数
        example["sentence_num"] = sentence_num[real_length]
        example["sentence_len"] = sentence_len[real_length]
        poem = array_to_poem(array_example, ix2word)
        # 创建输入输出
        # example["input"] = "<START> " + " ".join(poem[:example["sentence_len"]+1])
        # example["input_len"] = example["sentence_len"] + 2
        example["input_len"] = input_sentence_num * (example["sentence_len"] + 1)
        example["input"] = poem[: example["input_len"]]  # 原文形式的诗的输入
        # example["output"] = " ".join(poem[example["sentence_len"]+1:]) + " <EOP>"
        # example["output_len"] = (example["sentence_len"]+1) * (example["sentence_num"]-1) + 1
        example["output_len"] = (example["sentence_len"] + 1) * (example["sentence_num"] - input_sentence_num)
        example["output"] = poem[example["input_len"]:]  # 原文形式的诗的输出
        assert example["input_len"] + example["output_len"] + 2 == real_length, (
        example["input_len"], example["output_len"], real_length)
        preprocessed_data_list.append(example)
        # space_split_poem = f"<START> {' '.join(poem)} <EOP>"
        poem_to_word_list.append(list(poem))  # 将每首诗分割成单个字的列表，用于后面的词向量训练
    df = pd.DataFrame(preprocessed_data_list)

    # 训练word2vec，torchtext需要用训练好的word2vec构建词表
    w2v_model = word2vec.Word2Vec(poem_to_word_list, size=word_embedding_dim, window=window_size,
                                  min_count=word_min_count, workers=4)

    # 创建fields
    NUMBER = Field(sequential=False, dtype=torch.long, use_vocab=False, preprocessing=int)
    SENTENCE = Field(
        sequential=True, use_vocab=True, lower=False, pad_token=pad_token, unk_token=unk_token,
        tokenize=lambda s: list(s), batch_first=False, include_lengths=True)  # include_lengths可以统计加了上面的特殊符号后的长度

    fields = [
        ('sentence_num', NUMBER),
        ('sentence_len', NUMBER),
        (None, None),  # ('input_len', NUMBER),
        ('input', SENTENCE),
        (None, None),  # ('output_len', NUMBER),
        ('output', SENTENCE),
    ]

    # 创建数据集
    # with NamedTemporaryFile() as temp:  # 用临时文件创建
    #     df.to_csv(temp.name, sep="\t", index=False)
    #     dataset = TabularDataset(path=temp.name, format="TSV", fields=fields, skip_header=True)
    with open(preprocessed_data_path, "w") as f:
        df.to_csv(f, sep="\t", index=False)
    dataset = TabularDataset(path=preprocessed_data_path, format="TSV", fields=fields, skip_header=True)

    # 划分数据集。必须在这里就划分好，之后才能建立训练集的词表
    used_percent = train_percent + valid_percent + test_percent
    if used_percent < 1.:
        dataset_used, dataset_unused = dataset.split(split_ratio=[used_percent, 1 - used_percent])  # 划分使用的和不使用的
    else:
        dataset_used = dataset
    dataset_train, dataset_valid, dataset_test = dataset_used.split(
        split_ratio=[percent / used_percent for percent in [train_percent, valid_percent, test_percent]])
    dataset_predict = TabularDataset(path=predict_input_path, format="TSV", fields=[('input', SENTENCE)])

    # 利用word2vec创建词表vocab
    for field in (SENTENCE, ):
        # field.build_vocab(dataset_train, min_freq=word_min_count)
        field.build_vocab(dataset, min_freq=word_min_count)  # 构建个比较大的词表，减少unk
        word2vec_vectors = []
        for token, idx in field.vocab.stoi.items():
            if token in w2v_model.wv.vocab.keys():
                word2vec_vectors.append(torch.FloatTensor(w2v_model[token]))
            else:
                word2vec_vectors.append(torch.zeros(word_embedding_dim))
        field.vocab.set_vectors(field.vocab.stoi, word2vec_vectors, word_embedding_dim)
        assert all(w2v_model["我"] == field.vocab.vectors[field.vocab.stoi["我"]].numpy())

    # 返回值
    return dataset_train, dataset_valid, dataset_test, dataset_predict, SENTENCE.vocab, SENTENCE.vocab

if __name__=="__main__":
    dataset_train, dataset_valid, dataset_test, dataset_predict, vocab_input, vocab_output = get_dataset_and_vocab_language_model()
    it = iter(Iterator(dataset_test, batch_size=5, shuffle=False))
    batch = next(it)
    array_input, sentence_len = batch.input
    for i in range(5):
        print(array_to_poem(array_input[:, i], vocab_input.itos))
