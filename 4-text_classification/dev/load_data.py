#%%
import torch
import numpy as np
import re
from collections import Counter

raw_data = np.load("/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/tang.npz", allow_pickle=True)
data = raw_data["data"]
ix2word = raw_data['ix2word'].item()
word2ix = raw_data["word2ix"].item()
#%%
example = data[0]
example_poem = [ix2word[ix] for ix in example]

s所有符号 = set(word2ix.keys())
s所有汉字 = re.findall(r'[\u4e00-\u9fff]+', "".join(s所有符号))
s所有汉字 = set("".join(s所有汉字))
s所有非汉字符号 = s所有符号-s所有汉字
# {'、', '？', '䌽', '<START>', '䜩', '㳠', '！', '●', '〖', '〗', '○', '/', '：', 'Ｂ', '\ue829', '䦆', '□', '。', '\ue85a', '「', '，', '䶮', '『', '䴔', '<EOP>', '』', '䲡', '］', '</s>', '䴙', '；', '［', '䜣', '㖞', '（', '䴖', '」', '䌷'}

s所有非汉字符号 = {'、', '？',  '<START>',  '！', '●', '〖', '〗', '○', '/', '：', 'Ｂ',  '□', '。',  '「', '，',  '『',  '<EOP>', '』',  '］', '</s>',  '；', '［',  '（', '」', }
# {'、', '？', '<START>', '！', '●', '〖', '〗', '○', '/', '：', 'Ｂ', '□', '。', '「', '，', '『', '<EOP>', '』', '</s>', '；', '［', '（', '」', '］'}
f非汉字符号的ix = {word2ix[word] for word in s所有非汉字符号}
#  包含各个符号的诗的数量：
b包含各个符号的诗的数量 = {word: len(set(np.where(data==word2ix[word])[0])) for word in s所有非汉字符号}
# 』:   0
# （:   13
# /:   3
# </s>: 54248
# ！:   37
# 『:   0
# 」:   17
# ？:   683
# ］:   2
# 〗:   7
# <EOP>: 54251
# ，:   56814
# <START>: 57580
# ；:   27
# 。:   57580
# Ｂ:   1
# ●:   9
# ○:   4
# 「:   14
# 、:   2
# ：:   18
# □:   355
# ［:   2
# 〖:   7
#%% 筛查有特殊符号的诗
def array_to_poem(array):
    poem = [ix2word[ix] for ix in array]
    poem = filter(lambda x: x not in ("</s>",), poem)
    poem = "".join(poem)
    return poem

def find_poem_by_word(word):
    ix形式的poem = data[np.unique(np.where(data==word2ix[word])[0])]
    return [array_to_poem(array) for array in ix形式的poem]

y有特殊符号的诗 = {word: find_poem_by_word(word) for word, count in b包含各个符号的诗的数量.items() if count<20}
print(sum([count for count in b包含各个符号的诗的数量.values() if count<20])) # 未去重有99首，去掉后更少。数据集共57580首

# 没有特殊符号的诗
k空白符 = word2ix["</s>"]
s诗的长度 = Counter(np.sum(data!=k空白符, axis=1))
# top 10:
# [(50, 15267),
#  (34, 11661),
#  (66, 8406),
#  (26, 4313),
#  (125, 3332),
#  (74, 2701),
#  (98, 1489),
#  (122, 899),
#  (62, 826),
#  (38, 810)]

# 一首诗的长度
def poem_length(array):
    return np.sum(array!=k空白符)


