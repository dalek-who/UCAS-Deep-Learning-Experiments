from torch.utils.data import Dataset, DataLoader, random_split
import os
from pathlib import Path
from collections import namedtuple
from torchvision import transforms
import numpy as np


def array_to_poem(array, ix2word):
    poem = [ix2word[ix] for ix in array]
    poem = filter(lambda x: x not in ("</s>",), poem)
    poem = "".join(poem)
    return poem

# 训练集>=2000张，测试集>=500张
Example = namedtuple("Example", ("input", "output", "real_length", "has_special_symbol"))

class PoemDataset(Dataset):
    def __init__(self, root):
        super(self.__class__, self).__init__()
        self.root = Path(root)
        raw_data = np.load("/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/tang.npz", allow_pickle=True)
        self._raw_data: np.ndarray = raw_data["data"]
        self.ix2word: dict = raw_data['ix2word'].item()
        self.word2ix: dict = raw_data["word2ix"].item()

        self._START_ix = self.word2ix["<START>"]  # 诗起始符的ix
        self._EOP_ix = self.word2ix["<EOP>"]  # 诗结束符的ix
        self._white_space_ix = self.word2ix['</s>']  # 补位的空白符的ix
        self._special_symbols = ['、',  '●', '〖', '〗', '○', '/', '：', 'Ｂ', '□',  '「',  '『',  '』', '；', '［', '（', '」', '］']  # 有这些特殊符号的诗有噪音
        self._special_symbols_ix = np.array([self.word2ix[word] for word in self._special_symbols])

        sentence_num = 4  # 只保留七言四句诗

        self.data = []
        for array_example in self._raw_data:
            real_length = np.sum(array_example != self._white_space_ix)  # 去掉空白符后的实际长度。长度里包括<START>和<EOP>
            has_special_symbol = len(np.intersect1d(self._special_symbols_ix, array_example)) > 0  # 是否有特殊符号
            conditions = [ # 满足以下条件的才要
                real_length in (34,),  # (50, 34, 66, 26): 五言八句，七言四句，七言八句，五言四句。这是最多的四种类型，其他的不留
                not has_special_symbol,  # 有特殊符号的不留
            ]
            if not all(conditions):
                continue
            else:
                # new_array: np.ndarray = np.ones_like(array_example) * self._white_space_ix
                # new_array[:real_length] = array_example[-real_length:]  # 原本空白符padding在前面，改成padding在后面
                new_array: np.ndarray = array_example[-real_length + 1: -1]  # 只取正文部分，<START>和<EOP>都不要

                input = new_array.reshape(sentence_num, -1)[0]  # 把第一句作为input
                input_ones = np.ones(len(input)+1) * self._START_ix  # 以下三行是在开始加个<START>
                input_ones[1:] = input[:]
                input = input_ones

                output = new_array.reshape(sentence_num, -1)[1:].reshape(-1)  # 把后三句作为output
                output_ones = np.ones(len(output)+1) * self._EOP_ix  # 以下三行是在结尾加个<EOP>
                output_ones[:-1] = output[:]
                output = output_ones

                example = Example(input=input, output=output, real_length=real_length,
                                  has_special_symbol=has_special_symbol)
                self.data.append(example)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item) -> Example:
        return self.data[item]

    # def _preprocess_example(self, array_example) -> Example: # 预处理原始数据
    #     real_length = np.sum(array_example != self._white_space_ix)  # 去掉空白符后的实际长度。长度包括<START>和<END>
    #     has_special_symbol = len(np.intersect1d(self._special_symbols_ix, array_example))>0  # 是否有特殊符号
    #     # new_array: np.ndarray = np.ones_like(array_example) * self._white_space_ix
    #     # new_array[:real_length] = array_example[-real_length:]  # 原本空白符padding在前面，改成padding在后面
    #     new_array: np.ndarray = array_example[-real_length+1: -1]
    #     input = new_array.reshape(4, -1)[0]
    #     output = new_array.reshape(4, -1)[1:].reshape(1, -1)
    #     example =  Example(input=input, output=output, real_length=real_length, has_special_symbol=has_special_symbol)
    #     return example
    #
    # def _filter(self, example: Example) -> bool : # 过滤掉不合适的example
    #     conditions = [
    #         example.real_length in (34, ),  # (50, 34, 66, 26): 五言八句，七言四句，七言八句，五言四句。这是最多的四种类型，其他的不留
    #         not example.has_special_symbol,  # 有特殊符号的不留
    #     ]
    #     result = all(conditions)
    #     return result

if __name__=="__main__":
    root = "/data/users/wangyuanzheng/projects/ucas_DL/3-poem/data/tang.npz"
    dataset = PoemDataset(root)
    ix2word = dataset.ix2word
    d = dataset[0]
    print(array_to_poem(d.input, ix2word))
    print(array_to_poem(d.output, ix2word))
    # for i in range(len(dataset)):
    #     data = dataset[i]
    #     if data.real_length == 74:
    #         print(array_to_poem(data.padded_data, dataset.ix2word))
    #         break

# [(50, 15267), 5*8
#  (34, 11661), 7*4
#  (66, 8406),  7*8
#  (26, 4313),  5*4
#  (125, 3332),
#  (74, 2701),
#  (98, 1489),
#  (122, 899),
#  (62, 826),
#  (38, 810)]

# class CatDogDataset(Dataset):
#     label_dict = {"dog":0, "cat": 1}
#     Example = namedtuple("Example", ["img", "label", "name"])
#
#     def __init__(self, root, transform=None):
#         super(CatDogDataset, self).__init__()
#         self.root = Path(root)
#         self.img_name_list = os.listdir(root)
#         self.transform = transform
#         self._cache = dict()
#
#     def __len__(self):
#         return len(self.img_name_list)
#
#     def __getitem__(self, item):
#         if item not in self._cache:
#             img_name = self.img_name_list[item]
#             label = self.label_dict[img_name.split(".")[0]]
#             img = Image.open(self.root / img_name)
#             if self.transform is not None:
#                 img = self.transform(img)
#             example = self.Example(img, label, img_name)
#             self._cache[item] = example
#         else:
#             example = self._cache[item]
#         return example
