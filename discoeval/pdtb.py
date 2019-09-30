# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
PDTB - Task
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import logging
import numpy as np

from discoeval.tools.validation import SplitClassifier


class PDTBEval(object):
    def __init__(self, task_path, seed=1111):
        self.seed = seed

        logging.debug('***** Transfer task : PDTB classification, task path: {} *****\n\n'.format(task_path))

        train = self.loadFile(os.path.join(task_path, 'train.txt'))
        valid = self.loadFile(os.path.join(task_path, 'valid.txt'))
        test = self.loadFile(os.path.join(task_path, 'test.txt'))
        self.data = {'train': train, 'valid': valid, 'test': test}
        self.labelset = []
        with open(os.path.join(task_path, 'labelset.txt')) as fin:
            for line in fin:
                self.labelset.append(line.strip())
        self.nclasses = len(self.labelset) 

    def do_prepare(self, params, prepare):
        samples = [sent for sents in self.data['train'][:2] for sent in sents] + [sent for sents in self.data['valid'][:2]  for sent in sents] + [sent for sents in self.data['test'][:2]  for sent in sents]

        return prepare(params, samples)

    def loadFile(self, fpath):
        input1, input2, labels = [], [], []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split("\t")
                input1.append(line[1].split())
                input2.append(line[2].split())
                labels.append(int(line[0]))
        logging.debug('Loaded {} instances\n'.format(len(labels)))
        return (input1, input2, labels)

    def run(self, params, batcher):
        self.X, self.y = {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input1, input2, mylabels = self.data[key]
            enc_input = []
            n_labels = len(mylabels)
            for ii in range(0, n_labels, params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                np.abs(enc1 - enc2))))
                if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = np.array(mylabels)


            logging.info("encoding X to be: {}".format(self.X[key].shape))

        config = {'nclasses': self.nclasses, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for PDTB\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
