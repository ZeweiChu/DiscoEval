# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals
import os

from discoeval import utils
from discoeval.pdtb import PDTBEval
from discoeval.so import SPEval, BSOEval, DCEval, SSPEval
from discoeval.rst import RSTEval

class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['SParxiv', 'SProc', 'SPwiki', 
                           'DCchat', 'DCwiki', 
                           'BSOarxiv', 'BSOroc', 'BSOwiki', 
                           'SSPabs',
                           'PDTB-E', 'PDTB-I', 
                           'RST']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        if name == 'SParxiv':
            self.evaluation = SPEval(os.path.join(tpath, 'SP/arxiv'), "Sentence Position arXiv", nclasses=5, seed=self.params.seed)
        elif name == 'SProc':
            self.evaluation = SPEval(os.path.join(tpath, 'SP/rocstory'), "Sentence Position rocstory", nclasses=5, seed=self.params.seed)
        elif name == 'SPwiki':
            self.evaluation = SPEval(os.path.join(tpath, 'SP/wiki'), "Sentence Position wiki", nclasses=5, seed=self.params.seed)
        elif name == 'BSOarxiv':
            self.evaluation = BSOEval(os.path.join(tpath, 'BSO/arxiv'), "BSO arXiv", nclasses=2, seed=self.params.seed)
        elif name == 'BSOroc':
            self.evaluation = BSOEval(os.path.join(tpath, 'BSO/rocstory'), "BSO rocstory", nclasses=2, seed=self.params.seed)
        elif name == 'BSOwiki':
            self.evaluation = BSOEval(os.path.join(tpath, 'BSO/wiki'), "BSO wiki", nclasses=2, seed=self.params.seed)
        elif name == 'DCchat':
            self.evaluation = DCEval(os.path.join(tpath, 'DC/chat'), task_name="Discourse Coherence chat", nclasses=2, seed=self.params.seed)
        elif name == 'DCwiki':
            self.evaluation = DCEval(os.path.join(tpath, 'DC/wiki'), task_name="Discourse Coherence Wiki", nclasses=2, seed=self.params.seed)
        elif name == 'SSPabs':
            self.evaluation = SSPEval(os.path.join(tpath, 'SSP/abs'), "Sentence Section Prediction abstract", nclasses=2, seed=self.params.seed)
        elif name == 'PDTB-E':
            self.evaluation = PDTBEval(os.path.join(tpath, 'PDTB/Explicit'), seed=self.params.seed)
        elif name == 'PDTB-I':
            self.evaluation = PDTBEval(os.path.join(tpath, 'PDTB/Implicit'), seed=self.params.seed)
        elif name == 'RST':
            self.evaluation = RSTEval(os.path.join(tpath, 'RST'), seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
