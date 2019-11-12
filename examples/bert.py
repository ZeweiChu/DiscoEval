# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import torch
import code
import argparse

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import discoeval 

from transformers import BertConfig, BertTokenizer, BertModel 

# SentEval prepare and batcher
def prepare(params, samples):
    pass

def batcher(params, batch):
    layer = params["layer"]
    model = params["model"]
    tokenizer = params.tokenizer
    batch = [[token.lower() for token in sent] for sent in batch]
    batch = [" ".join(sent) if sent != [] else "." for sent in batch]
    batch = [["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"] for sent in batch]
    batch = [b[:512] for b in batch]
    seq_length = max([len(sent) for sent in batch])
    mask = [[1]*len(sent) + [0]*(seq_length-len(sent)) for sent in batch]
    segment_ids = [[0]*seq_length for _ in batch]
    batch = [tokenizer.convert_tokens_to_ids(sent) + [0]*(seq_length - len(sent)) for sent in batch]
    with torch.no_grad():
        batch = torch.tensor(batch).cuda()
        mask = torch.tensor(mask).cuda() # bs * seq_length
        segment_ids = torch.tensor(segment_ids).cuda()   	
        outputs, pooled_output, hidden_states, _ = model(batch, token_type_ids=segment_ids, attention_mask=mask)
        if layer == "avg":
            output = [o.data.cpu()[:, 0].numpy() for o in hidden_states]
            embeddings = np.mean(output, 0)
        elif layer == "pooler":
            embeddings = pooled_output.data.cpu().numpy()
        else: 
            layer = int(layer)
            output = hidden_states[layer]
            embeddings = output.data.cpu()[:, 0].numpy()
 
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_index", default=0, type=int, required=True,
                        help="which task to perform")
    parser.add_argument("--layer", default="avg", type=str, required=True,
                        help="which layer to evaluate on")
    parser.add_argument("--model_type", default="base", type=str, required=True, choices=["base", "large"],
                        help="the type of BERT model to evaluate on")
    args = parser.parse_args()

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    config = BertConfig.from_pretrained('bert-{}-uncased'.format(args.model_type))
    config.output_hidden_states = True
    config.output_attentions = True
    tokenizer = BertTokenizer.from_pretrained('bert-{}-uncased'.format(args.model_type))
    model = BertModel.from_pretrained('bert-{}-uncased'.format(args.model_type), config=config).cuda()
    model.eval()

    # Set params for SentEval
    params_discoeval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16, 
                        'tokenizer': tokenizer, "layer": args.layer, "model": model}
    params_discoeval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                     'tenacity': 5, 'epoch_size': 4}
    se = discoeval.engine.SE(params_discoeval, batcher, prepare)


    transfer_tasks = [
        ['SParxiv', 'SProc', 'SPwiki'], 
        ['DCchat'], 
        ['DCwiki'], 
        ['BSOarxiv', 'BSOroc', 'BSOwiki'], 
        ['SSPabs', 'PDTB-E', 'PDTB-I', 'RST']]
    results = se.eval(transfer_tasks[args.task_index])
    print(results)
