import torch
import model_utils
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class recurrent_decoder_base(nn.Module):
    def __init__(self, vocab_size, input_size, mlp_hidden_size,
                 mlp_layer, hidden_size, embed_dim, dropout, tie_weight,
                 word_dropout, log, embed_init, *args, **kwargs):
        super(recurrent_decoder_base, self).__init__()
        self.word_dropout = word_dropout
        self.tie_weight = tie_weight

        self.embed = nn.Embedding(vocab_size, embed_dim)
        if embed_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embed_init))
            log.info(
                "{} initialized with pretrained word embedding".format(
                    type(self)))
        if not tie_weight:
            self.hid2vocab = nn.Linear(hidden_size, vocab_size)

    def drop_word(self, inputs):
        """
        Do word dropout: with prob `word_dropout`, set the word to '<unk>'.
        """

        # Sample masks: elems with val 1 will be set to <unk>
        if self.word_dropout and self.training:
            mask = torch.from_numpy(
                np.random.binomial(
                    1, p=self.word_dropout, size=tuple(inputs.size()))
                .astype('uint8')
            )

            mask[0, :] = 0  # never drop bos
            mask = Variable(inputs.data.new(mask.size()).copy_(mask)).byte()

            # Set to <unk>
            new_inputs = Variable(
                inputs.data.new(inputs.size()).copy_(inputs.data.clone()))
            new_inputs.masked_fill_(mask, 0)

            return new_inputs
        else:
            return inputs

    def get_init_state(self, input_vecs):
        init_vecs = self.latent2hid(input_vecs)

        if isinstance(self.cell, nn.LSTM):
            init_vecs = tuple([h.unsqueeze(0).contiguous() for h in
                              torch.chunk(init_vecs, 2, -1)])
        elif isinstance(self.cell, nn.GRU):
            init_vecs = init_vecs.unsqueeze(0)
        return init_vecs

    def teacher_force(self, init_state, tgts, tgts_mask):
        pred = self.pred(init_state, tgts, tgts_mask)

        batch_size, seq_len, vocab_size = pred.size()

        pred = pred.contiguous().view(batch_size * seq_len, vocab_size)
        logloss = F.cross_entropy(
            pred, tgts[:, 1:].contiguous().view(-1).long(), reduce=False)

        logloss = (logloss.view(batch_size, seq_len) *
                   tgts_mask[:, 1:]).sum(-1) / tgts_mask[:, 1:].sum(-1)
        return logloss.mean()

    def forward(self, input_vecs, tgts, tgts_mask):
        init_vecs = self.get_init_state(input_vecs)
        return self.teacher_force(init_vecs, tgts, tgts_mask)

    def pred(self, init_state, tgts, tgts_mask):
        bs, sl = tgts_mask.size()
        inp_tgts = self.drop_word(tgts)
        tgts_embed = self.embed(inp_tgts.long())

        output_seq, _ = model_utils.get_rnn_vecs(
            tgts_embed, tgts_mask, self.cell, initial_state=init_state)
        # batch size x seq len x vocab size
        if not self.tie_weight:
            pred = self.hid2vocab(output_seq)[:, :-1, :]
        else:
            pred = torch.matmul(output_seq, self.embed.weight.t())[:, :-1, :]
        return pred


class lstm(recurrent_decoder_base):
    def __init__(self, vocab_size, input_size, mlp_hidden_size,
                 mlp_layer, hidden_size, embed_dim, dropout, tie_weight,
                 word_dropout, log, embed_init, *args, **kwargs):
        super(lstm, self).__init__(vocab_size, input_size, mlp_hidden_size,
            mlp_layer, hidden_size, embed_dim, dropout, tie_weight,
            word_dropout, log, embed_init, *args, **kwargs)

        self.latent2hid = model_utils.get_mlp(
            input_size, mlp_hidden_size,
            hidden_size * 2, mlp_layer, dropout)

        self.cell = nn.LSTM(
            embed_dim, hidden_size,
            bidirectional=False, batch_first=True)


class gru(recurrent_decoder_base):
    def __init__(self, vocab_size, input_size, mlp_hidden_size,
                 mlp_layer, hidden_size, embed_dim, dropout, tie_weight,
                 word_dropout, log, embed_init, *args, **kwargs):
        super(gru, self).__init__(vocab_size, input_size, mlp_hidden_size,
            mlp_layer, hidden_size, embed_dim, dropout, tie_weight,
            word_dropout, log, embed_init, *args, **kwargs)

        self.latent2hid = model_utils.get_mlp(
            input_size, mlp_hidden_size,
            hidden_size, mlp_layer, dropout)

        self.cell = nn.GRU(
            embed_dim, hidden_size,
            bidirectional=False, batch_first=True)
        if not tie_weight:
            self.hid2vocab = nn.Linear(hidden_size, vocab_size)


class gru_cat(recurrent_decoder_base):
    def __init__(self, vocab_size, input_size, mlp_hidden_size,
                 mlp_layer, hidden_size, embed_dim, dropout, tie_weight,
                 word_dropout, log, embed_init, *args, **kwargs):
        super(gru_cat, self).__init__(vocab_size, input_size, mlp_hidden_size,
            mlp_layer, hidden_size, embed_dim, dropout, tie_weight,
            word_dropout, log, embed_init, *args, **kwargs)

        self.latent2hid = model_utils.get_mlp(
            input_size, mlp_hidden_size,
            hidden_size, mlp_layer, dropout)

        self.cell = nn.GRU(
            embed_dim + input_size, hidden_size,
            bidirectional=False, batch_first=True)
        if not tie_weight:
            self.hid2vocab = nn.Linear(hidden_size, vocab_size)

    def teacher_force(self, vecs, init_state, tgts, tgts_mask):
        pred = self.pred(vecs, init_state, tgts, tgts_mask)

        batch_size, seq_len, vocab_size = pred.size()

        pred = pred.contiguous().view(batch_size * seq_len, vocab_size)
        logloss = F.cross_entropy(
            pred, tgts[:, 1:].contiguous().view(-1).long(), reduce=False)

        logloss = (logloss.view(batch_size, seq_len) *
                   tgts_mask[:, 1:]).sum(-1) / tgts_mask[:, 1:].sum(-1)
        return logloss.mean()

    def forward(self, input_vecs, tgts, tgts_mask):
        init_vecs = self.get_init_state(input_vecs)
        return self.teacher_force(input_vecs, init_vecs, tgts, tgts_mask)

    def pred(self, vecs, init_state, tgts, tgts_mask):
        bs, sl = tgts_mask.size()
        inp_vecs = vecs.unsqueeze(1).expand(-1, sl, -1)
        inp_tgts = self.drop_word(tgts)
        tgts_embed = self.embed(inp_tgts.long())

        output_seq, _ = model_utils.get_rnn_vecs(
            torch.cat([tgts_embed, inp_vecs], -1), tgts_mask,
            self.cell, initial_state=init_state)
        # batch size x seq len x vocab size
        if not self.tie_weight:
            pred = self.hid2vocab(output_seq)[:, :-1, :]
        else:
            pred = torch.matmul(output_seq, self.embed.weight.t())[:, :-1, :]
        return pred


class bag_of_words(nn.Module):
    def __init__(self, vocab_size, input_size, mlp_hidden_size,
                 mlp_layer, hidden_size, embed_dim, dropout, tie_weight,
                 word_dropout, log, embed_init, *args, **kwargs):
        super(bag_of_words, self).__init__()
        self.func = model_utils.get_mlp(
            input_size, mlp_hidden_size, hidden_size, mlp_layer, dropout)
        self.hid2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_vecs, tgts, tgts_mask):
        logits = F.log_softmax(self.hid2vocab(self.func(input_vecs)), -1)
        return -(torch.sum(logits * tgts, 1) / tgts.sum(1)).mean()
