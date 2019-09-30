import os
import pickle

import numpy as np

from decorators import auto_init_args, lazy_execute


VOCAB_SIZE = 50000


class data_holder:
    @auto_init_args
    def __init__(self, train_data, test_data, sent_vocab,
                 doc_title_vocab, sec_title_vocab):
        self.inv_vocab = {i: w for w, i in sent_vocab.items()}


class data_processor:
    @auto_init_args
    def __init__(self, train_path, eval_path, experiment):
        self.expe = experiment

    def process(self):
        if self.expe.config.embed_file is None:
            vocab_fn = "vocab-" + str(VOCAB_SIZE) + "-no-pretrained"
        else:
            vocab_fn = "vocab-" + str(VOCAB_SIZE)

        W, sent_vocab, doc_title_vocab, sec_title_vocab = \
            self._build_pretrain_vocab(file_name=vocab_fn)
        self.expe.log.info(
            "vocab size - sent: {}, doc title: {}, sec title: {}"
            .format(len(sent_vocab), len(doc_title_vocab),
                    len(sec_title_vocab)))

        # get list of training paths
        train_data = []
        for path, subdirs, files in os.walk(self.train_path):
            for name in files:
                if name[-1] in "0123456789" and \
                        name.startswith("sentence_context.txt"):
                    train_data.append(os.path.join(path, name))
            break
        self.expe.log.info("#train file: {}".format(len(train_data)))

        def cal_stats(data):
            unk_count = 0
            total_count = 0
            leng = []
            for sent1, sent2 in zip(*data):
                leng.append(len(sent1))
                leng.append(len(sent2))
                for w in sent1 + sent2:
                    if w == 0:
                        unk_count += 1
                    total_count += 1
            return (unk_count, total_count, unk_count / total_count), \
                (len(leng), max(leng), min(leng), sum(leng) / len(leng))

        if self.eval_path is not None:
            eval_data = self._load_from_pickle(self.eval_path)
            new_data = dict()
            for year, data in sorted(eval_data.items()):
                self.expe.log.info(
                    "year: {}, #domain: {}".format(year, len(data)))
                new_data[year] = dict()
                for n, d in data.items():
                    data_idx = self._data_to_idx([d[0], d[1]], sent_vocab)
                    new_data[year][n] = [data_idx[0], data_idx[1], d[2]]

            for year, data in sorted(new_data.items()):
                for n, d in sorted(data.items()):
                    unk_stats, len_stats = cal_stats(d[:2])
                    self.expe.log.info("year {}, domain: {} #data: {}, "
                                       "max len: {}, min len: {}, "
                                       "avg len: {:.2f}"
                                       .format(year, n, *len_stats))
                    self.expe.log.info("#unk in year {}, domain {}: {}"
                                       .format(year, n, unk_stats))

            data = data_holder(
                train_data=sorted(train_data),
                test_data={y: new_data[y] for y in new_data},
                sent_vocab=sent_vocab,
                doc_title_vocab=doc_title_vocab,
                sec_title_vocab=sec_title_vocab)
        else:
            data = data_holder(
                train_data=sorted(train_data),
                test_data=None,
                sent_vocab=sent_vocab,
                doc_title_vocab=doc_title_vocab,
                sec_title_vocab=sec_title_vocab)

        return data, W

    def _data_to_idx(self, data, vocab):
        idx_pair1 = []
        idx_pair2 = []
        for d1, d2 in zip(*data):
            s1 = [vocab.get(w, 0) for w in d1]
            idx_pair1.append(s1)
            s2 = [vocab.get(w, 0) for w in d2]
            idx_pair2.append(s2)
        return np.array(idx_pair1), np.array(idx_pair2)

    def _load_glove_embedding(self, path):
        self.expe.log.info("loading GloVe embedding from: {}".format(path))
        with open(path, 'r', encoding='utf8') as fp:
            # word_vectors: word --> vector
            word_vectors = {}
            for line in fp:
                line = line.strip("\n").split(" ")
                word_vectors[line[0]] = np.array(
                    list(map(float, line[1:])), dtype='float32')
        vocab_embed = word_vectors.keys()
        embed_dim = word_vectors[next(iter(vocab_embed))].shape[0]

        return word_vectors, vocab_embed, embed_dim

    @lazy_execute("_load_from_pickle")
    def _build_pretrain_vocab(self):
        sent_vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2}
        doc_title_vocab, sec_title_vocab, sentence_vocab = \
            self._load_from_pickle(self.expe.config.vocab_file)

        doc_title_vocab = [x[0] for x in doc_title_vocab.most_common(VOCAB_SIZE)]
        doc_title_vocab = dict(zip(doc_title_vocab, range(1, len(doc_title_vocab) + 1)))
        doc_title_vocab["<unk>"] = 0
        sec_title_vocab = [x[0] for x in sec_title_vocab.most_common(VOCAB_SIZE)]
        sec_title_vocab = dict(zip(sec_title_vocab, range(1, len(sec_title_vocab) + 1)))
        sec_title_vocab["<unk>"] = 0

        if self.expe.config.embed_file is not None:
            word_vectors, vocab_embed, embed_dim = \
                self._load_glove_embedding(self.expe.config.embed_file)

            sentence_vocab = sentence_vocab.most_common()

            W = np.random.uniform(
                -np.sqrt(3.0 / embed_dim), np.sqrt(3.0 / embed_dim),
                size=(len(sent_vocab), embed_dim)).astype('float32')
            W = [W]
            n = 0
            for x in sorted(sentence_vocab, key=lambda x: -x[1]):
                if x[0] in word_vectors:
                    sent_vocab[x[0]] = len(sent_vocab)
                    W.append(word_vectors[x[0]][None, :])
                    n += 1
                    if len(sent_vocab) > VOCAB_SIZE:
                        break
            W = np.concatenate(W)
            self.expe.log.info(
                "{}/{} vocabs are initialized with GloVe embeddings."
                .format(n, len(sent_vocab)))
        else:
            for x in sentence_vocab.most_common(VOCAB_SIZE):
                sent_vocab[x[0]] = len(sent_vocab)
            W = None

        return W, sent_vocab, doc_title_vocab, sec_title_vocab

    def _load_from_pickle(self, file_name):
        self.expe.log.info("loading from {}".format(file_name))
        with open(file_name, "rb") as fp:
            data = pickle.load(fp)
        return data


class minibatcher:
    @auto_init_args
    def __init__(self, train_file, bow, batch_size, max_len,
                 max_nsent, max_npara, sent_vocab, doc_title_vocab,
                 sec_title_vocab, log):
        self.stop_flag = False
        self.sent_vocab_size = len(sent_vocab)
        self.doc_title_vocab_size = len(doc_title_vocab)
        self.sec_title_vocab_size = len(sec_title_vocab)

        self.n_sent = 0
        self.src_max_sent_len = 0
        self.src_min_sent_len = max_len
        self.src_sum_sent_len = 0

        self.prev_tgt_max_sent_len = 0
        self.prev_tgt_min_sent_len = max_len
        self.prev_tgt_sum_sent_len = 0

        self.next_tgt_max_sent_len = 0
        self.next_tgt_min_sent_len = max_len
        self.next_tgt_sum_sent_len = 0

        if bow:
            self._pad = self._pad_bow
        else:
            self._pad = self._pad_seq

    def _pad_seq(self, input_data, tgt_data, tgt_data2):
        assert len(input_data) == len(tgt_data)
        assert len(input_data) == len(tgt_data2)

        max_inp_len = max([len(inp_d) for inp_d in input_data])
        max_tgt_len = max([len(tgt_d) for tgt_d in tgt_data])
        max_tgt2_len = max([len(tgt_d) for tgt_d in tgt_data2])

        input_sents = \
            np.zeros((len(input_data), max_inp_len)).astype("float32")
        input_masks = \
            np.zeros((len(input_data), max_inp_len)).astype("float32")

        tgt_sents = \
            np.zeros((len(tgt_data), max_tgt_len + 2)).astype("float32")
        tgt_masks = \
            np.zeros((len(tgt_data), max_tgt_len + 2)).astype("float32")

        tgt_sents2 = \
            np.zeros((len(tgt_data2), max_tgt2_len + 2)).astype("float32")
        tgt_masks2 = \
            np.zeros((len(tgt_data2), max_tgt2_len + 2)).astype("float32")

        for i, (inp, tgt, tgt2) in enumerate(
                zip(input_data, tgt_data, tgt_data2)):

            input_sents[i, :len(inp)] = \
                np.asarray([self.sent_vocab.get(w, 0) for w in inp])\
                .astype("float32")
            input_masks[i, :len(inp)] = 1.

            tgt_sents[i, :len(tgt) + 2] = \
                np.asarray([1] + [self.sent_vocab.get(w, 0) for w in tgt] + [2])\
                .astype("float32")
            tgt_masks[i, :len(tgt) + 2] = 1.

            tgt_sents2[i, :len(tgt2) + 2] = \
                np.asarray([1] + [self.sent_vocab.get(w, 0) for w in tgt2] + [2])\
                .astype("float32")
            tgt_masks2[i, :len(tgt2) + 2] = 1.

        return [input_sents, input_masks, tgt_sents,
                tgt_masks, tgt_sents2, tgt_masks2]

    def _pad_bow(self, input_data, tgt_data, tgt_data2):
        assert len(input_data) == len(tgt_data)
        assert len(input_data) == len(tgt_data2)
        max_inp_len = max([len(inp_d) for inp_d in input_data])

        input_sents = \
            np.zeros((len(input_data), max_inp_len)).astype("float32")
        input_masks = \
            np.zeros((len(input_data), max_inp_len)).astype("float32")

        tgt_sents = \
            np.zeros((len(tgt_data), self.sent_vocab_size)).astype("float32")

        tgt_sents2 = \
            np.zeros((len(tgt_data2), self.sent_vocab_size)).astype("float32")

        for i, (inp, tgt, tgt2) in enumerate(
                zip(input_data, tgt_data, tgt_data2)):

            input_sents[i, :len(inp)] = \
                np.asarray([self.sent_vocab.get(w, 0) for w in inp])\
                .astype("float32")
            input_masks[i, :len(inp)] = 1.

            for w in tgt:
                tgt_sents[i, self.sent_vocab.get(w, 0)] += 1

            for w in tgt2:
                tgt_sents2[i, self.sent_vocab.get(w, 0)] += 1

        return [input_sents, input_masks, tgt_sents, input_masks,
                tgt_sents2, input_masks]

    def _pad_titles(self, doc_title, sec_title):
        assert len(doc_title) == len(sec_title)

        doc_title_bow = \
            np.zeros((len(doc_title), self.doc_title_vocab_size))\
            .astype("float32")
        sec_title_bow = \
            np.zeros((len(sec_title), self.sec_title_vocab_size))\
            .astype("float32")

        for i, (dt, st) in enumerate(zip(doc_title, sec_title)):
            for w in dt:
                doc_title_bow[i, self.doc_title_vocab.get(w, 0)] += 1

            for w in st:
                sec_title_bow[i, self.sec_title_vocab.get(w, 0)] += 1

        return [doc_title_bow, sec_title_bow]

    def __enter__(self):
        self.log.info("iterating {}".format(self.train_file))
        self.file = open(self.train_file, 'r')
        return self

    def __exit__(self, *args):
        self.log.info("#sents: {}, "
                      "src sent len stats - avg: {:.2f} | max: {} | min: {}"
                      .format(self.n_sent, self.src_sum_sent_len / self.n_sent,
                              self.src_max_sent_len, self.src_min_sent_len))
        self.log.info("#sents: {}, "
                      "prev sent len stats - avg: {:.2f} | max: {} | min: {}"
                      .format(self.n_sent,
                              self.prev_tgt_sum_sent_len / self.n_sent,
                              self.prev_tgt_max_sent_len,
                              self.prev_tgt_min_sent_len))
        self.log.info("#sents: {}, "
                      "next sent len stats - avg: {:.2f} | max: {} | min: {}"
                      .format(self.n_sent,
                              self.next_tgt_sum_sent_len / self.n_sent,
                              self.next_tgt_max_sent_len,
                              self.next_tgt_min_sent_len))
        self.file.close()

    def __iter__(self):
        return self

    def __next__(self):
        doc_ids, para_ids, sent_ids, sent_para_ids, pmask, \
            smask, curr_sents, prev_sents, next_sents, levels, \
            doc_titles, sent_titles = [[] for _ in range(12)]
        if self.stop_flag:
            raise StopIteration()
        for _ in range(self.batch_size):
            try:
                doc_id, para_id, sent_id, sent_para_id, n_para, \
                    n_sent, _, curr_sent, prev_sent, next_sent, \
                    level, doc_title, sec_title = \
                    next(self.file).strip().lower().split("\t")
                doc_ids.append(int(doc_id))
                para_ids.append(int(para_id))
                sent_ids.append(int(sent_id))
                sent_para_ids.append(int(sent_para_id))
                levels.append(int(level) - 1)

                doc_titles.append(
                    doc_title.lower().strip().split(" ")[:self.max_len])
                sent_titles.append(
                    sec_title.lower().strip().split(" ")[:self.max_len])

                # n_paras.append(n_para)
                # n_sents.append(n_sent)
                curr_sents.append(
                    curr_sent.lower().strip().split(" ")[:self.max_len])
                prev_sents.append(
                    prev_sent.lower().strip().split(" ")[:self.max_len])
                next_sents.append(
                    next_sent.lower().strip().split(" ")[:self.max_len])

                pm = np.zeros((1, self.max_npara))
                pm[0, : int(n_para)] = 1
                pmask.append(pm)

                sm = np.zeros((1, self.max_nsent))
                sm[0, : int(n_sent)] = 1
                smask.append(sm)

                self.n_sent += 1
                self.src_min_sent_len = min(self.src_min_sent_len,
                                            len(curr_sents[-1]))
                self.src_max_sent_len = max(self.src_max_sent_len,
                                            len(curr_sents[-1]))
                self.src_sum_sent_len += len(curr_sents[-1])

                self.next_tgt_min_sent_len = min(self.next_tgt_min_sent_len,
                                                 len(next_sents[-1]))
                self.next_tgt_max_sent_len = max(self.next_tgt_max_sent_len,
                                                 len(next_sents[-1]))
                self.next_tgt_sum_sent_len += len(next_sents[-1])

                self.prev_tgt_min_sent_len = min(self.prev_tgt_min_sent_len,
                                                 len(prev_sents[-1]))
                self.prev_tgt_max_sent_len = max(self.prev_tgt_max_sent_len,
                                                 len(prev_sents[-1]))
                self.prev_tgt_sum_sent_len += len(prev_sents[-1])

            except StopIteration:
                if len(pmask):
                    self.stop_flag = True
                    break
                else:
                    raise StopIteration()

        pmask = np.concatenate(pmask)
        smask = np.concatenate(smask)

        return [np.array(doc_ids).astype("int32"),
                np.array(para_ids).astype("int32"),
                pmask, np.array(sent_ids).astype("int32"),
                np.array(sent_para_ids).astype("int32"), smask,
                np.array(levels).astype("int32")] + \
            self._pad(curr_sents, prev_sents, next_sents) + \
            self._pad_titles(doc_titles, sent_titles)


class sts_minibatcher:
    @auto_init_args
    def __init__(self, data1, data2, batch_size,
                 shuffle, p_scramble, *args, **kwargs):
        self._reset()

    def __len__(self):
        return len(self.idx_pool)

    def _reset(self):
        self.pointer = 0
        idx_list = np.arange(len(self.data1))
        if self.shuffle:
            np.random.shuffle(idx_list)
        self.idx_pool = [idx_list[i: i + self.batch_size]
                         for i in range(0, len(self.data1), self.batch_size)]

    def _pad(self, data1, data2):
        assert len(data1) == len(data2)
        max_len1 = max([len(sent) for sent in data1])
        max_len2 = max([len(sent) for sent in data2])

        input_data1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        input_mask1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        tgt_data1 = \
            np.zeros((len(data1), max_len1 + 2)).astype("float32")
        tgt_mask1 = \
            np.zeros((len(data1), max_len1 + 2)).astype("float32")

        input_data2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        input_mask2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        tgt_data2 = \
            np.zeros((len(data2), max_len2 + 2)).astype("float32")
        tgt_mask2 = \
            np.zeros((len(data2), max_len2 + 2)).astype("float32")

        for i, (sent1, sent2) in enumerate(zip(data1, data2)):
            if np.random.choice(
                    [True, False],
                    p=[self.p_scramble, 1 - self.p_scramble]).item():
                sent1 = np.random.permutation(sent1)
                sent2 = np.random.permutation(sent2)

            input_data1[i, :len(sent1)] = \
                np.asarray(list(sent1)).astype("float32")
            input_mask1[i, :len(sent1)] = 1.

            tgt_data1[i, :len(sent1) + 2] = \
                np.asarray([1] + list(sent1) + [2]).astype("float32")
            tgt_mask1[i, :len(sent1) + 2] = 1.

            input_data2[i, :len(sent2)] = \
                np.asarray(list(sent2)).astype("float32")
            input_mask2[i, :len(sent2)] = 1.

            tgt_data2[i, :len(sent2) + 2] = \
                np.asarray([1] + list(sent2) + [2]).astype("float32")
            tgt_mask2[i, :len(sent2) + 2] = 1.
        return [input_data1, input_mask1, input_data2, input_mask2]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            raise StopIteration()

        idx = self.idx_pool[self.pointer]
        data1, data2 = self.data1[idx], self.data2[idx]
        self.pointer += 1
        return self._pad(data1, data2) + [idx]
