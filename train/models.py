import os
import torch
import model_utils
import encoders
import decoders

import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from decorators import auto_init_pytorch
from torch.autograd import Variable


class base(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_init, experiment):
        super(base, self).__init__()
        self.expe = experiment
        self.eps = self.expe.config.eps
        self.use_cuda = self.expe.config.use_cuda

        self.encode = getattr(encoders, self.expe.config.encoder_type)(
            embed_dim=embed_dim,
            hidden_size=self.expe.config.ensize,
            embed_init=embed_init,
            vocab_size=vocab_size,
            log=experiment.log)

        if self.expe.config.encoder_type.lower() == "lstm" or \
                self.expe.config.encoder_type.lower() == "gru":
            ensize = 2 * self.expe.config.ensize
        else:
            ensize = embed_dim

        self.decode = getattr(decoders, self.expe.config.decoder_type)(
            input_size=ensize,
            mlp_hidden_size=self.expe.config.mhsize,
            mlp_layer=self.expe.config.mlplayer,
            hidden_size=self.expe.config.desize,
            dropout=self.expe.config.dp,
            embed_dim=embed_dim,
            tie_weight=self.expe.config.tw,
            word_dropout=self.expe.config.wd,
            embed_init=embed_init,
            vocab_size=vocab_size,
            log=experiment.log)

    def to_var(self, inputs):
        if self.use_cuda:
            if isinstance(inputs, Variable):
                inputs = inputs.cuda()
                inputs.volatile = self.volatile
                return inputs
            else:
                if not torch.is_tensor(inputs):
                    inputs = torch.from_numpy(inputs)
                return Variable(inputs.cuda(), volatile=self.volatile)
        else:
            if isinstance(inputs, Variable):
                inputs = inputs.cpu()
                inputs.volatile = self.volatile
                return inputs
            else:
                if not torch.is_tensor(inputs):
                    inputs = torch.from_numpy(inputs)
                return Variable(inputs, volatile=self.volatile)

    def to_vars(self, *inputs):
        return [self.to_var(inputs_) if inputs_.size else None
                for inputs_ in inputs]

    def optimize(self, loss):
        self.opt.zero_grad()
        loss.backward()
        if self.expe.config.gclip is not None:
            torch.nn.utils.clip_grad_norm(
                self.parameters(), self.expe.config.gclip)
        self.opt.step()

    def init_optimizer(self, opt_type, learning_rate, weight_decay):
        if opt_type.lower() == "adam":
            optimizer = torch.optim.Adam
        elif opt_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop
        elif opt_type.lower() == "sgd":
            optimizer = torch.optim.SGD
        elif opt_type.lower() == "adadelta":
            optimizer = torch.optim.Adadelta
        else:
            raise NotImplementedError("invalid optimizer: {}".format(opt_type))

        opt = optimizer(
            params=filter(
                lambda p: p.requires_grad, self.parameters()
            ),
            lr=learning_rate,
            weight_decay=weight_decay)

        return opt

    def lr_decay(self, epoch):
        lr = self.expe.config.lr / (1 + self.expe.config.lrd * epoch)
        self.expe.log.info("Updated learning rate: {}".format(lr))
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def save(self, test_bm, test_avg, todo_file, epoch, it, name="latest"):
        save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
        checkpoint = {
            "test_avg": test_avg,
            "test_bm": test_bm,
            "todo_file": todo_file,
            "epoch": epoch,
            "iteration": it,
            "state_dict": self.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "config": self.expe.config
        }
        torch.save(checkpoint, save_path)
        self.expe.log.info("model saved to {}".format(save_path))

    def load(self, checkpointed_state_dict=None, name="latest"):
        if checkpointed_state_dict is None:
            save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
            checkpoint = torch.load(save_path,
                                    map_location=lambda storage,
                                    loc: storage)
            self.load_state_dict(checkpoint['state_dict'])
            if checkpoint.get("opt_state_dict"):
                self.opt.load_state_dict(checkpoint.get("opt_state_dict"))
            self.expe.log.info("model loaded from {}".format(save_path))
            return checkpoint.get('epoch', 0), \
                checkpoint.get('iteration', 0), \
                checkpoint.get('test_bm', 0), \
                checkpoint.get('test_avg', 0), \
                checkpoint.get('todo_file', [])
        else:
            self.load_state_dict(checkpointed_state_dict)
            self.expe.log.info("model loaded!")

    @property
    def volatile(self):
        return not self.training


class basic_model(base):
    @auto_init_pytorch
    def __init__(self, vocab_size, embed_dim, embed_init, experiment,
                 *args, **kwargs):
        super(basic_model, self).__init__(
            vocab_size, embed_dim, embed_init, experiment)
        pass

    def forward(self, sent, mask, tgt, tgt_mask, tgt2, tgt_mask2, *args):
        self.train()
        sent, mask, tgt, tgt_mask, tgt2, tgt_mask2 = \
            self.to_vars(sent, mask, tgt, tgt_mask, tgt2, tgt_mask2)
        sent_vec = self.encode(sent, mask)
        logloss1 = self.decode(sent_vec, tgt, tgt_mask)
        logloss2 = self.decode(sent_vec, tgt2, tgt_mask2)

        logloss = logloss1 + logloss2
        return logloss, logloss1, logloss2, torch.zeros_like(logloss), \
            torch.zeros_like(logloss)

    def score_sts(self, sent1, mask1, sent2, mask2):
        self.eval()
        sent1, mask1, sent2, mask2 = self.to_vars(sent1, mask1, sent2, mask2)

        sent1_vec = self.encode(sent1, mask1)
        sent2_vec = self.encode(sent2, mask2)
        sent_cos_pos = F.cosine_similarity(sent1_vec, sent2_vec)
        return sent_cos_pos.data.cpu().numpy()


class pos_model(base):
    @auto_init_pytorch
    def __init__(self, vocab_size, embed_dim, embed_init, max_nsent, max_npara,
                 experiment, *args, **kwargs):
        super(pos_model, self).__init__(
            vocab_size, embed_dim, embed_init, experiment)

        if self.expe.config.encoder_type.lower() == "lstm" or \
                self.expe.config.encoder_type.lower() == "gru":
            ensize = 2 * self.expe.config.ensize
        else:
            ensize = embed_dim

        self.sent_id_pred = model_utils.get_mlp(
            input_size=ensize,
            hidden_size=self.expe.config.mhsize,
            output_size=max_nsent,
            n_layer=self.expe.config.mlplayer,
            dropout=self.expe.config.dp)

        self.para_id_pred = model_utils.get_mlp(
            input_size=ensize,
            hidden_size=self.expe.config.mhsize,
            output_size=max_npara,
            n_layer=self.expe.config.mlplayer,
            dropout=self.expe.config.dp)

    def forward(self, sent, mask, tgt, tgt_mask, tgt2, tgt_mask2,
                doc_id, para_id, pmask, sent_id, smask, *args):
        self.train()
        sent, mask, tgt, tgt_mask, tgt2, tgt_mask2, \
            para_id, pmask, sent_id, smask = \
            self.to_vars(sent, mask, tgt, tgt_mask, tgt2, tgt_mask2, para_id,
                         pmask, sent_id, smask)
        bs, sl = sent.size()

        sent_vec = self.encode(sent, mask)
        logloss1 = self.decode(sent_vec, tgt, tgt_mask)
        logloss2 = self.decode(sent_vec, tgt2, tgt_mask2)

        logloss = logloss1 + logloss2

        if self.expe.config.sratio:
            sent_id_logit = self.sent_id_pred(sent_vec)
            neg_smask = 1 - smask
            sent_id_logit.data.masked_fill_(
                neg_smask.data.byte(), -float('inf'))
            sent_id_loss = F.cross_entropy(sent_id_logit, sent_id.long())
        else:
            sent_id_loss = torch.zeros_like(logloss)

        if self.expe.config.pratio:
            para_id_logit = self.para_id_pred(sent_vec)
            neg_pmask = 1 - pmask
            para_id_logit.data.masked_fill_(
                neg_pmask.data.byte(), -float('inf'))
            para_id_loss = F.cross_entropy(para_id_logit, para_id.long())
        else:
            para_id_loss = torch.zeros_like(logloss)

        loss = self.expe.config.lratio * logloss + \
            self.expe.config.sratio * sent_id_loss + \
            self.expe.config.pratio * para_id_loss
        return loss, logloss1, logloss2, sent_id_loss, para_id_loss

    def score_sts(self, sent1, mask1, sent2, mask2):
        self.eval()
        sent1, mask1, sent2, mask2 = self.to_vars(sent1, mask1, sent2, mask2)

        sent1_vec = self.encode(sent1, mask1)
        sent2_vec = self.encode(sent2, mask2)
        sent_cos_pos = F.cosine_similarity(sent1_vec, sent2_vec)
        return sent_cos_pos.data.cpu().numpy()


class quantize_pos_model(base):
    @auto_init_pytorch
    def __init__(self, vocab_size, embed_dim, embed_init, max_nsent, max_npara,
                 experiment, *args, **kwargs):
        super(quantize_pos_model, self).__init__(
            vocab_size, embed_dim, embed_init, experiment)

        if self.expe.config.encoder_type.lower() == "lstm" or \
                self.expe.config.encoder_type.lower() == "gru":
            ensize = 2 * self.expe.config.ensize
        else:
            ensize = embed_dim

        self.sent_id_pred = model_utils.get_mlp(
            input_size=ensize,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.nb,
            n_layer=self.expe.config.mlplayer,
            dropout=self.expe.config.dp)

        self.para_id_pred = model_utils.get_mlp(
            input_size=ensize,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.nb,
            n_layer=self.expe.config.mlplayer,
            dropout=self.expe.config.dp)

        self.bins = np.arange(self.expe.config.nb) / self.expe.config.nb

    def quantize_pos(self, ids, ids_mask):
        quant_ids = np.digitize(ids / ids_mask.sum(1), self.bins) - 1
        return quant_ids

    def forward(self, sent, mask, tgt, tgt_mask, tgt2, tgt_mask2,
                doc_id, para_id, pmask, sent_id, smask, *args):
        self.train()

        sent, mask, tgt, tgt_mask, tgt2, tgt_mask2, para_id, sent_id = \
            self.to_vars(
                sent, mask, tgt, tgt_mask, tgt2, tgt_mask2,
                self.quantize_pos(para_id, pmask),
                self.quantize_pos(sent_id, smask))

        bs, sl = sent.size()

        sent_vec = self.encode(sent, mask)
        logloss1 = self.decode(sent_vec, tgt, tgt_mask)
        logloss2 = self.decode(sent_vec, tgt2, tgt_mask2)

        logloss = logloss1 + logloss2

        if self.expe.config.sratio:
            sent_id_logit = self.sent_id_pred(sent_vec)
            sent_id_loss = F.cross_entropy(sent_id_logit, sent_id.long())
        else:
            sent_id_loss = torch.zeros_like(logloss)

        if self.expe.config.pratio:
            para_id_logit = self.para_id_pred(sent_vec)
            para_id_loss = F.cross_entropy(para_id_logit, para_id.long())
        else:
            para_id_loss = torch.zeros_like(logloss)

        loss = self.expe.config.lratio * logloss + \
            self.expe.config.sratio * sent_id_loss + \
            self.expe.config.pratio * para_id_loss
        return loss, logloss1, logloss2, sent_id_loss, para_id_loss

    def score_sts(self, sent1, mask1, sent2, mask2):
        self.eval()
        sent1, mask1, sent2, mask2 = self.to_vars(sent1, mask1, sent2, mask2)

        sent1_vec = self.encode(sent1, mask1)
        sent2_vec = self.encode(sent2, mask2)
        sent_cos_pos = F.cosine_similarity(sent1_vec, sent2_vec)
        return sent_cos_pos.data.cpu().numpy()
