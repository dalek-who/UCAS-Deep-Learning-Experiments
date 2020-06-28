from typing import List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random
from torchtext.vocab import Vocab


class TextCNN(nn.Module):
    def __init__(self, emb_dim: int, filter_sizes: list, num_each_filter: int, dropout: float, num_labels: int, vocab: Vocab):
        super(self.__class__, self).__init__()
        assert len(filter_sizes)==len(set(filter_sizes)), filter_sizes
        self.vocab: Vocab = vocab
        self.embedding = nn.Embedding(len(vocab), emb_dim).from_pretrained(vocab.vectors, freeze=False)
        self.conv_kernels = nn.ModuleDict({
            f"conv_{ks}": nn.Conv2d(in_channels=1, out_channels=num_each_filter, kernel_size=(ks, emb_dim))
            for ks in filter_sizes
        })
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(len(filter_sizes) * num_each_filter, num_labels)

    def forward(self, x):
        # x: [batch_size, seq_len]

        emb: Tensor = self.embedding(x)
        # emb = [batch_size, seq_len, emb_dim]
        emb = emb.unsqueeze(1)
        # emb = [batch_size, 1, seq_len, emb_dim]  # 2d Conv需要有channel维。TextCNN一种Embedding层就是一维
        del x

        conv_list: List[Tensor] = [F.relu(conv_kernel(emb)) for conv_kernel in self.conv_kernels.values()]
        # conved_list[i]: [batch_size, num_each_filter, seq_len - filter_sizes[i] + 1, 1]
        del emb
        conv_list: List[Tensor] = [conv.squeeze(3) for conv in conv_list]
        # conved_list[i]: [batch_size, num_each_filter, seq_len - filter_sizes[i] + 1]

        pooled_list: List[Tensor] = [F.max_pool1d(conv, kernel_size=conv.shape[2]) for conv in conv_list]
        # pooled_list[i]: [batch_size, num_each_filter, 1]
        del conv_list
        pooled_list: List[Tensor] = [pooled.squeeze(2) for pooled in pooled_list]
        # pooled_list[i]: [batch_size, num_each_filter]

        cat = self.dropout(torch.cat(pooled_list, dim=1))
        # cat: [batch_size, num_each_filter * len(filter_sizes)]
        del pooled_list

        return self.classifier(cat)


