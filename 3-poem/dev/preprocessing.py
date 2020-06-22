import torch
import numpy as np
import re
from collections import Counter
import torchtext
import pandas as pd

raw_data = np.load("/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/tang.npz", allow_pickle=True)
data = raw_data["data"]
ix2word = raw_data['ix2word'].item()
word2ix = raw_data["word2ix"].item()

START_ix = word2ix["<START>"]
EOP_ix = word2ix["<EOP>"]
white_space_ix = word2ix['</s>']
special_symbols = ['、', '●', '〖', '〗', '○', '/', '：', 'Ｂ', '□', '「', '『', '』', '；', '［', '（', '」', '］']  # 有这些特殊符号的诗有噪音
special_symbols_ix = np.array([word2ix[word] for word in special_symbols])

def array_to_poem(array):
    poem = [ix2word[ix] for ix in array]
    poem = filter(lambda x: x not in ("</s>", "<START>", "<EOP>"), poem)
    poem = "".join(poem)
    return poem

# (50, 34, 66, 26): 五言八句，七言四句，七言八句，五言四句
c长度_to_有多少句   = {50: 8, 34: 4, 66: 8, 26: 4}
c长度_to_每句几个字 = {50: 5, 34: 7, 66: 7, 26: 5}
y允许的长度 = (50, 34, 66, 26)
preprocessed_data_list = []
space_split_poem_list = []  # 用空格分开的诗
for array_example in data:
    example = dict()
    real_length = np.sum(array_example != white_space_ix)  # 去掉空白符后的实际长度。长度里包括<START>和<EOP>
    has_special_symbol = len(np.intersect1d(special_symbols_ix, array_example)) > 0  # 是否有特殊符号
    if not real_length in y允许的长度 or has_special_symbol:
        continue
    example["sentence_num"] = c长度_to_有多少句[real_length]
    example["sentence_len"] = c长度_to_每句几个字[real_length]
    poem = array_to_poem(array_example)
    example["input"] = "<START> " + " ".join(poem[:example["sentence_len"]+1])
    example["input_len"] = example["sentence_len"] + 2
    example["output"] = " ".join(poem[example["sentence_len"]+1:]) + " <EOP>"
    example["output_len"] = (example["sentence_len"]+1) * (example["sentence_num"]-1) + 1
    assert example["input_len"] + example["output_len"] == real_length
    preprocessed_data_list.append(example)
    space_split_poem = f"<START> {' '.join(poem)} <EOP>"
    space_split_poem_list.append(space_split_poem)

with open("poem_for_embedding.txt", "w") as f:
    f.write("\n".join(space_split_poem_list))

df = pd.DataFrame(preprocessed_data_list)
df.to_csv("poem.tsv", sep="\t", index=False)
# self._START_ix = self.word2ix["<START>"]  # 诗起始符的ix
# self._EOP_ix = self.word2ix["<EOP>"]  # 诗结束符的ix
# self._white_space_ix = self.word2ix['</s>']  # 补位的空白符的ix
# self._special_symbols = ['、', '●', '〖', '〗', '○', '/', '：', 'Ｂ', '□', '「', '『', '』', '；', '［', '（', '」',
#                          '］']  # 有这些特殊符号的诗有噪音
# self._special_symbols_ix = np.array([self.word2ix[word] for word in self._special_symbols])
#
# sentence_num = 4  # 只保留七言四句诗
#
# self.data = []
# for array_example in self._raw_data:
#     real_length = np.sum(array_example != self._white_space_ix)  # 去掉空白符后的实际长度。长度里包括<START>和<EOP>
#     has_special_symbol = len(np.intersect1d(self._special_symbols_ix, array_example)) > 0  # 是否有特殊符号
#     conditions = [  # 满足以下条件的才要
#         real_length in (34,),  # (50, 34, 66, 26): 五言八句，七言四句，七言八句，五言四句。这是最多的四种类型，其他的不留
#         not has_special_symbol,  # 有特殊符号的不留
#     ]
#     if not all(conditions):
#         continue
#     else:
#         # new_array: np.ndarray = np.ones_like(array_example) * self._white_space_ix
#         # new_array[:real_length] = array_example[-real_length:]  # 原本空白符padding在前面，改成padding在后面
#         new_array: np.ndarray = array_example[-real_length + 1: -1]  # 只取正文部分，<START>和<EOP>都不要
#
#         input = new_array.reshape(sentence_num, -1)[0]  # 把第一句作为input
#         input_ones = np.ones(len(input) + 1) * self._START_ix  # 以下三行是在开始加个<START>
#         input_ones[1:] = input[:]
#         input = input_ones
#
#         output = new_array.reshape(sentence_num, -1)[1:].reshape(-1)  # 把后三句作为output
#         output_ones = np.ones(len(output) + 1) * self._EOP_ix  # 以下三行是在结尾加个<EOP>
#         output_ones[:-1] = output[:]
#         output = output_ones
#
#         example = Example(input=input, output=output, real_length=real_length,
#                           has_special_symbol=has_special_symbol)
#         self.data.append(example)
