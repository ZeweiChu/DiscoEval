import torch
import model_utils

import torch.nn as nn
import torch.nn.functional as F
import code
class encoder_base(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_init, log,
                 *args, **kwargs):
        super(encoder_base, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if embed_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embed_init))
            log.info(
                "{} initialized with pretrained word embedding".format(
                    type(self)))


class lstm(encoder_base):
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 embed_init, log, *args, **kwargs):
        super(lstm, self).__init__(vocab_size, embed_dim, embed_init, log)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, inputs, mask):
        input_vecs = self.embed(inputs.long())
        outputs, _ = model_utils.get_rnn_vecs(
            input_vecs, mask, self.lstm, bidir=True)
        outputs = outputs * mask.unsqueeze(-1)
        sent_vec = outputs.sum(1) / mask.sum(1, keepdim=True)
        return sent_vec

    def get_vecs(self, inputs, mask):
        input_vecs = self.embed(inputs.long())
        outputs, _ = model_utils.get_rnn_vecs(
            input_vecs, mask, self.lstm, bidir=True)
        outputs = outputs * mask.unsqueeze(-1)
        sent_vec = outputs.sum(1) / mask.sum(1, keepdim=True)
        return sent_vec, outputs


class gru(encoder_base):
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 embed_init, log, *args, **kwargs):
        super(gru, self).__init__(vocab_size, embed_dim, embed_init, log)
        self.lstm = nn.GRU(
            embed_dim, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, inputs, mask):
        input_vecs = self.embed(inputs.long())
        outputs, _ = model_utils.get_rnn_vecs(
            input_vecs, mask, self.lstm, bidir=True)
        # code.interact(local=locals())
        outputs = outputs * mask.unsqueeze(-1)
        sent_vec = outputs.sum(1) / mask.sum(1, keepdim=True)
        return sent_vec

    def get_vecs(self, inputs, mask):
        input_vecs = self.embed(inputs.long())
        outputs, _ = model_utils.get_rnn_vecs(
            input_vecs, mask, self.lstm, bidir=True)
        outputs = outputs * mask.unsqueeze(-1)
        sent_vec = outputs.sum(1) / mask.sum(1, keepdim=True)
        return sent_vec, outputs


class wordavg(encoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init, log,
                 *args, **kwargs):
        super(wordavg, self).__init__(vocab_size, embed_dim, embed_init, log)
        pass

    def forward(self, inputs, mask):
        input_vecs = self.embed(inputs.long()) * mask.unsqueeze(-1)
        sum_vecs = input_vecs.sum(1)
        avg_vecs = sum_vecs / mask.sum(1, keepdim=True)
        return avg_vecs

    def get_vecs(self, inputs, mask):
        input_vecs = self.embed(inputs.long()) * mask.unsqueeze(-1)
        sum_vecs = input_vecs.sum(1)
        avg_vecs = sum_vecs / mask.sum(1, keepdim=True)
        return avg_vecs, input_vecs


class gru_attn(encoder_base):
    ''' The sentence representation is the weighted sum of gru vectors 
    '''
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 embed_init, log, *args, **kwargs):
        super(gru_attn, self).__init__(vocab_size, embed_dim, embed_init, log)
        self.lstm = nn.GRU(
            embed_dim, hidden_size, bidirectional=True, batch_first=True)
        
        self.weight_embed = nn.Embedding(vocab_size, hidden_size*2)
    
    def attn(self, inp, mask, vecs, embed):
        weight = (embed(inp.long()) * vecs).sum(-1)  # bs x seq len
        weight.data.masked_fill_(
            (1 - mask).data.byte(), -float('inf'))
        return (vecs * F.softmax(weight, 1).unsqueeze(-1)).sum(1) # bs * (2*hidden_size)

    def forward(self, inputs, mask):
        input_vecs = self.embed(inputs.long())
        outputs, _ = model_utils.get_rnn_vecs(
            input_vecs, mask, self.lstm, bidir=True)
        sent_vec = self.attn(inputs, mask, outputs, self.weight_embed)
        return sent_vec

    def get_vecs(self, inputs, mask):
        input_vecs = self.embed(inputs.long())
        outputs, _ = model_utils.get_rnn_vecs(
            input_vecs, mask, self.lstm, bidir=True)
        outputs = outputs * mask.unsqueeze(-1)
        sent_vec = self.attn(inputs, mask, outputs, self.weight_embed)
        return sent_vec, outputs
