import torch
import torch.nn as nn


def get_mlp(input_size, hidden_size, output_size, n_layer, dropout):
    if n_layer == 0:
        proj = nn.Linear(input_size, output_size)
    else:
        proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout))

        for i in range(n_layer - 1):
            proj.add_module(
                str(len(proj)),
                nn.Linear(hidden_size, hidden_size))
            proj.add_module(str(len(proj)), nn.ReLU())
            proj.add_module(str(len(proj)), nn.Dropout(dropout))

        proj.add_module(
            str(len(proj)),
            nn.Linear(hidden_size, output_size))
    return proj


def get_rnn_vecs(
        inputs,
        mask,
        cell,
        bidir=False,
        initial_state=None,
        get_last=False):
    """
    Args:
    inputs: batch_size x seq_len x n_feat
    mask: batch_size x seq_len
    initial_state: batch_size x num_layers x hidden_size
    cell: GRU/LSTM/RNN
    """
    seq_lengths = torch.sum(mask, dim=-1).squeeze(-1)
    sorted_len, sorted_idx = seq_lengths.sort(0, descending=True)
    sorted_inputs = inputs[sorted_idx.long()]
    if sorted_len.dim() == 0:
        sorted_len = sorted_len.unsqueeze(0)
        sorted_idx = sorted_idx.unsqueeze(0)
        sorted_inputs = sorted_inputs.unsqueeze(0)
    packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
        sorted_inputs, sorted_len.long().cpu().data.numpy(), batch_first=True)
    if initial_state is not None:
        if isinstance(cell, torch.nn.LSTM):
            initial_state = \
                (initial_state[0].index_select(1, sorted_idx.long()),
                 initial_state[1].index_select(1, sorted_idx.long()))
        else:
            initial_state = \
                initial_state.index_select(1, sorted_idx.long())
    out, hid = cell(packed_seq, hx=initial_state)
    unpacked, unpacked_len = \
        torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True)
    _, original_idx = sorted_idx.sort(0, descending=False)
    output_seq = unpacked[original_idx.long()]
    if get_last:
        if isinstance(hid, tuple):
            if bidir:
                hid = tuple([torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
                            for h in hid])
            hid = \
                (hid[0].index_select(1, original_idx.long()),
                 hid[1].index_select(1, original_idx.long()))
        else:
            if bidir:
                hid = torch.cat(
                    [hid[0:hid.size(0):2], hid[1:hid.size(0):2]], 2)
            hid = hid.index_select(1, original_idx.long())
    return output_seq, hid


def binary_cross_entropy_with_logits(input, target, weight=None):
    r"""Function that measures Binary Cross Entropy between target and output
    logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Default: ``True``

    Examples::

         >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
         >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    return loss
