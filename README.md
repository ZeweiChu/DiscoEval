# DiscoEval

This repository contains the code for DiscoEval
[Evaluation Benchmarks and Learning Criteria for Discourse-Aware Sentence Representations](https://arxiv.org/abs/1909.00142) (EMNLP 2019).

The structure of this repo:
- ```train```: the training code
- ```discoeval```: the Discourse Evaluation framework
- ```data```: the DiscoEval evaluation datasets

The pretrained models with different training signals can be downloaded from ```https://drive.google.com/file/d/1I0wFkNb2fmoC7kcj-FPxyVkHKyfX-MoX/view?usp=sharing```

The training data (generated from Wikipedia) can be downloaded from

```https://drive.google.com/open?id=1WPRJylC7PzLtYcg8-PMNX_ZUNNRkO3Bp```

Evaluation example code
```
train/discoeval_example.py
```

The code is tested under the following environment/versions:
- Python 3.6.2
- PyTorch 1.0.0
- numpy 1.16.0

Some code in this repo is adopted from [SentEval](https://github.com/facebookresearch/SentEval). 



## Reference

```
@inproceedings{mchen-discoeval-19,
  author    = {Mingda Chen and Zewei Chu and Kevin Gimpel},
  title     = {Evaluation Benchmarks and Learning Criteria for Discourse-Aware Sentence Representations},
  booktitle = {Proc. of {EMNLP}},
  year      = {2019}
}
```

