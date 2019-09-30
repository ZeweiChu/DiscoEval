from __future__ import absolute_import, division, unicode_literals


"""
Example of file for evaluating DiscoEval
"""
import logging
import sys
from collections import OrderedDict
from encoders import lstm, wordavg, gru
import pickle
from torch.autograd import Variable
import torch
import numpy as np

# Set PATHs
PATH_TO_DISCOEVAL = '..'
PATH_TO_DATA = '../data/'

sys.path.insert(0, PATH_TO_DISCOEVAL)
import discoeval 

def prepare(params, samples):
    return

def batcher(params, batch):
    vocab = params["vocab"]
    model = params["model"]
    max_batch_len = max([len(inp_d) for inp_d in batch])
    batch_sents = np.zeros((len(batch), max_batch_len)).astype("float32")
    batch_masks = np.zeros((len(batch), max_batch_len)).astype("float32")
    for i, inp in enumerate(batch):
        batch_sents[i, :len(inp)] = np.asarray([vocab.get(w.lower(), 0) for w in inp]).astype("float32")
        batch_masks[i, :max(len(inp), 1)] = 1.

    model.eval()
    to_var = params["to_var"]
    batch_sents, batch_masks = to_var(model, batch_sents), to_var(model, batch_masks)
    with torch.no_grad():
        embeddings = model.forward(batch_sents, batch_masks)
    
    return embeddings.data.cpu().numpy()

# Set params for SentEval
params_discoeval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16}
params_discoeval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def to_var(model, inputs):
    ''' convert the inputs to pytorch variable 
    '''
    if next(model.parameters()).is_cuda:
        if isinstance(inputs, Variable):
            inputs = inputs.cuda()
            inputs.volatile = not model.training
            return inputs
        else:
            if not torch.is_tensor(inputs):
                inputs = torch.from_numpy(inputs)
            return Variable(inputs.cuda(), volatile = not model.training)
    else:
        if isinstance(inputs, Variable):
            inputs = inputs.cpu()
            inputs.volatile = not model.training
            return inputs
        else:
            if not torch.is_tensor(inputs):
                inputs = torch.from_numpy(inputs)
            return Variable(inputs, volatile = not model.training)


if __name__ == "__main__":
    # Load model
    ckpt_file = sys.argv[1]
    task_index = int(sys.argv[2])

    saved_model = torch.load(ckpt_file)
    state_dict = saved_model["state_dict"]
    config = saved_model["config"]
    new_state_dict = OrderedDict([(k.replace("encode.", ""), state_dict[k]) for k in state_dict.keys() if k.startswith("encode")])

    vocab_file = "vocab-50000"
    
    def _build_vocab(vocab_file):
        vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2}
        with open(vocab_file, 'rb') as vf:
            vocab = pickle.load(vf)[1]
        return vocab
    vocab = _build_vocab(vocab_file)
    params_discoeval['vocab'] = vocab
  
    if config.encoder_type == "lstm":
        model = lstm(len(vocab), config.edim, config.ensize, None, None)
    elif config.encoder_type == "gru":
        model = gru(len(vocab), config.edim, config.ensize, None, None)
    else:
        model = wordavg(len(vocab), config.edim, None, None)

    model.load_state_dict(new_state_dict)
    model = model.cuda()
    params_discoeval['model'] = model
    params_discoeval['to_var'] = to_var


    se = discoeval.engine.SE(params_discoeval, batcher, prepare)
    transfer_tasks = [
        ['SParxiv', 'SProc', 'SPwiki'], 
        ['DCchat'], 
        ['DCwiki'], 
        ['BSOarxiv', 'BSOroc', 'BSOwiki'], 
        ['SSPabs', 'PDTB-E', 'PDTB-I', 'RST']]
    results = se.eval(transfer_tasks[task_index])
    print(results)

