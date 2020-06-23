from gensim.models import Word2Vec
import torch
import torchtext
from torchtext.data import Field, LabelField, RawField, TabularDataset
from torchtext.vocab import GloVe, Vectors
from torchtext.data import Iterator, BucketIterator
import numpy as np

#%%
# vectors = Vectors(cache="/data/users/wangyuanzheng/projects/ucas_DL/3-poem/dev/vocab/", name=f"w2v_{dim}.txt")
# vectors = Vectors(cache="/data/users/wangyuanzheng/projects/ucas_DL/3-poem/dev/vocab/", name=f"glove_{dim}.txt")

# SENTENCE_NUM = Field(sequential=False, dtype=torch.long, preprocessing=int)
# SENTENCE_LEN = Field(sequential=False, dtype=torch.long, preprocessing=int)
# INPUT = Field(sequential=True, use_vocab=True, lower=False, pad_token="</s>", unk_token="<unk>", tokenize=lambda s: s.split(" "))
# INPUT_LEN = Field(sequential=False, dtype=torch.long, preprocessing=int)
# OUTPUT = Field(sequential=True, use_vocab=True, lower=False, pad_token="</s>", unk_token="<unk>", tokenize=lambda s: s.split(" "))
# OUTPUT_LEN = Field(sequential=False, dtype=torch.long, preprocessing=int)

# 创建fields
NUMBER = Field(sequential=False, dtype=torch.long, use_vocab=False, preprocessing=int)
SENTENCE = Field(sequential=True, use_vocab=True, lower=False, pad_token="</s>", unk_token="<unk>", tokenize=lambda s: s.split(" "))

fields = [
    ('sentence_num', NUMBER),
    ('sentence_len', NUMBER),
    ('input', SENTENCE),
    ('input_len', NUMBER),
    ('output', SENTENCE),
    ('output_len', NUMBER),
]

# fields = [
#     ('sentence_num', SENTENCE_NUM),
#     ('sentence_len', SENTENCE_LEN),
#     ('input', INPUT),
#     ('input_len', INPUT_LEN),
#     ('output', OUTPUT),
#     ('output_len', OUTPUT_LEN),
# ]

# 创建数据集
path = "./poem.tsv"
dataset = TabularDataset(path=path, format="TSV", fields=fields, skip_header=True)

# 划分数据集
train_percent = 0.1
valid_percent = 0.1
test_percent = 0.1
used_percent = train_percent + valid_percent + test_percent
if used_percent<1.:
    dataset_used, dataset_unused = dataset.split(split_ratio=[used_percent, 1-used_percent])  # 划分使用的和不使用的
else:
    dataset_used = dataset
dataset_train, dataset_valid, dataset_test = dataset_used.split(split_ratio=[percent/used_percent for percent in [train_percent, valid_percent, test_percent]])

# 加载word2vec
SENTENCE.build_vocab(dataset_train)

dim=128
w2v = Word2Vec.load(f"/data/users/wangyuanzheng/projects/ucas_DL/3-poem/dev/vocab/w2v_{dim}.txt")
word2vec_vectors = []
for token, idx in SENTENCE.vocab.stoi.items():
    if token in w2v.wv.vocab.keys():
        word2vec_vectors.append(torch.FloatTensor(w2v[token]))
    else:
        word2vec_vectors.append(torch.zeros(dim))
SENTENCE.vocab.set_vectors(SENTENCE.vocab.stoi, word2vec_vectors, dim)
assert all(w2v["我"] == SENTENCE.vocab.vectors[SENTENCE.vocab.stoi["我"]].numpy())

# iter
iter_test = Iterator(dataset_test, batch_size=5, shuffle=False, sort=None)

# batch
batch = next(iter(iter_test))

# pack pad
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence
pk = pack_padded_sequence(input=batch.input, lengths=batch.input_len, enforce_sorted=False)
pd = pad_packed_sequence(pk, padding_value=0, )

# 把output转换成诗
#%%
func_itos = np.vectorize(lambda x: SENTENCE.vocab.itos[x])
array_poem_input = func_itos(batch.input.numpy().T)
array_poem_output = func_itos(batch.output.numpy().T)
array_split_poem = np.concatenate([array_poem_input, array_poem_output], axis=1)
def func_poem_str(row):
    poem = "".join(filter(lambda word: word not in ("</s>", "<START>"), row))
    poem = poem.split("<EOP>")[0]
    return poem
poems = np.apply_along_axis(func1d=func_poem_str, axis=1, arr=array_split_poem)
