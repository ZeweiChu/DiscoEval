# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
RST-DT - Task
'''
from __future__ import absolute_import, division, unicode_literals

import pickle
import os
import io
import copy
import logging
import numpy as np

from discoeval.tools.validation import SplitClassifier


class RSTEval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : RST-DT Task at {}*****\n\n'.format(taskpath))
        self.seed = seed

        train_sents, train_labels = self.loadFile(os.path.join(taskpath, 'RST_TRAIN.pkl'))
        valid_sents, valid_labels = self.loadFile(os.path.join(taskpath, 'RST_DEV.pkl'))
        test_sents, test_labels = self.loadFile(os.path.join(taskpath, 'RST_TEST.pkl'))

        logging.debug("#train: {}, #dev: {}, #test: {}".format(len(train_sents), len(valid_sents), len(test_sents)))
        self.labelset = set(train_labels + valid_labels + test_labels)
        self.samples = sum(train_sents, []) + sum(valid_sents, []) + sum(test_sents, [])

        logging.debug('***** Total instances loaded: {}*****'.format(len(train_sents + valid_sents + test_sents)))
        logging.debug('***** Total #label categories: {}****\n\n'.format(len(self.labelset)))
        self.data = {'train': (train_sents, train_labels),
                     'valid': (valid_sents, valid_labels),
                     'test': (test_sents, test_labels)
                     }

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        data = pickle.load(open(fpath, "rb"))
        sents = []
        labels = []
        for d in data:
            sents.append([d[1], d[2]])
            labels.append(d[0])
        return sents, labels

    def run(self, params, batcher):
        self.X, self.y = {}, {}
        dico_label = {k: v for v, k in enumerate(self.labelset)}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input, labels = self.data[key]

            logging.debug('split: {}, #data: {}'.format(key, len(input)))
            enc_input = []
            sent2vec = {}
            for ii in range(0, len(input), params.batch_size):
                batch = input[ii:ii + params.batch_size]

                encs1 = []
                encs2 = []
                if len(batch):
                    s2enc = []
                    for edus in batch:
                        for s in edus[0]:
                            if s not in sent2vec:
                                s2enc.append(s)
                        for s in edus[1]:
                            if s not in sent2vec:
                                s2enc.append(s)
                    if len(s2enc):
                        s2enc = sorted(s2enc, key=lambda x: len(x.split()))
                        encs1_ = []
                        for iii in range(0, len(s2enc), params.batch_size):
                            s2enc_ = [s.split() for s in s2enc[iii: iii + params.batch_size]]
                            encs1_.append(batcher(params, s2enc_))
                        for s, e in zip(s2enc, np.concatenate(encs1_)):
                            sent2vec[s] = e
                    encs1 = [np.stack([sent2vec[s] for s in ed[0]]).mean(0) for ed in batch]
                    encs2 = [np.stack([sent2vec[s] for s in ed[1]]).mean(0) for ed in batch]
                    enc1 = np.stack(encs1)
                    enc2 = np.stack(encs2)
                    enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                np.abs(enc1 - enc2))))
                if ii % (100*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / len(input)))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = np.array([dico_label[y] for y in labels])


            logging.info("encoding X to be: {}".format(self.X[key].shape))

        config = {'nclasses': len(dico_label), 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for RST-DT\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
