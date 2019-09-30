import argparse
import pickle

from tqdm import tqdm
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str)
parser.add_argument('--output', '-o', type=str)
args = parser.parse_args()

doc_title_vocab = Counter()
sec_title_vocab = Counter()
sentence_vocab = Counter()

with open(args.file) as fp:
    pbar = tqdm(fp, total=42168551)
    for line in pbar:
        strip_line = line.strip()
        if strip_line and not strip_line.startswith("<doc id="):
            d = strip_line.split("\t")
            doc_title, sec_title, sentences = d[1], d[2], d[3:]
            for w in doc_title.lower().split(" "):
                doc_title_vocab[w] += 1
            for w in sec_title.lower().split(" "):
                sec_title_vocab[w] += 1
            for s in sentences:
                for w in s.lower().split(" "):
                    sentence_vocab[w] += 1

print("len(doc_title_vocab)", len(doc_title_vocab))
print("len(sec_title_vocab)", len(sec_title_vocab))
print("len(sentence_vocab)", len(sentence_vocab))

print("top 10 in document title vocab")
print(doc_title_vocab.most_common(10))
print("top 10 in section title vocab")
print(sec_title_vocab.most_common(10))
print("top 10 in sentence vocab")
print(sentence_vocab.most_common(10))

with open(args.output, "wb+") as fp:
    pickle.dump([doc_title_vocab, sec_title_vocab, sentence_vocab], fp)
