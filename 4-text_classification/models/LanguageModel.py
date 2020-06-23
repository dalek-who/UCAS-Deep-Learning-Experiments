import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random
from torchtext.vocab import Vocab


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hid_dim, rnn_num_layers, linear_hid_dim, dropout, vocab):
        super(self.__class__, self).__init__()
        self.vocab: Vocab = vocab
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim).from_pretrained(vocab.vectors)
        self.rnn = nn.GRU(embed_dim, rnn_hid_dim, rnn_num_layers)
        self.linear = nn.Linear(rnn_hid_dim, linear_hid_dim)
        self.output_layer = nn.Linear(linear_hid_dim, vocab_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_len, trg, teacher_forcing_ratio):
        # src: [src_seq_len, batch_size]
        # src_len: [batch_size]
        # trg: [trg_seq_len, batch_size]
        # teacher_forcing_ratio: scalar

        device = next(self.parameters()).device
        embedded = self.dropout1(self.embedding(src))
        # embedded: [src_seq_len, batch_size, embed_dim]

        packed_embedded = pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        _, hidden = self.rnn(packed_embedded)
        # hidden: [rnn_num_layers, batch_size, rnn_hid_dim]

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        vocab_size = self.vocab_size

        # input = trg[0, :]
        # input: [batch_size]

        # tensor to store decoder outputs, [trg_seq_len, batch_size, vocab_size]
        outputs = torch.zeros(trg_len, batch_size, vocab_size).to(device)
        rnn_output = hidden[-1].squeeze(0)
        # outputs: [trg_seq_len, batch_size, vocab_size]
        # rnn_output: [batch_size, rnn_hid_dim]

        for t in range(trg_len):
            outputs[t] = self.output_layer(self.dropout2(F.relu(self.linear(rnn_output))))
            # get the highest predicted token from our predictions
            top1 = rnn_output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio.detach().item()
            input = trg[t] if teacher_force else top1
            input = input.unsqueeze(0)
            # input: [1, batch_size]

            embedded = self.dropout1(self.embedding(input))
            # embedded: [1, batch_size, embed_dim]

            rnn_output, hidden = self.rnn(embedded, hidden)
            # rnn_output: [1, batch_size, rnn_hid_dim]
            # hidden: [rnn_num_layers, batch_size, rnn_hid_dim]

            rnn_output = rnn_output.squeeze(0)
            # rnn_output: [batch_size, rnn_hid_dim]

        return outputs

