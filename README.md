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


## Experiments
||SP                           |BSO   |DC                                           |SSP |PDTB-E|PDTB-I|RST |AVG |
|------|-----------------------------|------|---------------------------------------------|----|------|------|----|----|
|baseline|47.3                         |63.8  |61.0                                         |77.8|36.5  |39.1  |56.7|54.6|
|SDT   |45.8                         |62.9  |60.3                                         |78.0|36.6  |39.1  |55.7|54.1|
|SPP   |48.4                         |65.3  |60.2                                         |78.4|38.1  |39.9  |56.4|55.2|
|NL    |46.9                         |64.0  |61.0                                         |78.9|37.6  |39.9  |56.5|55.0|
|SPP + NL|48.5                         |64.7  |59.9                                         |78.9|37.8  |40.5  |56.7|55.3|
|SDT + NL|46.1                         |63.0  |60.8                                         |78.1|36.7  |38.1  |56.2|54.1|
|SPP + SDT|46.5                         |63.9  |60.4                                         |77.6|35.2  |38.6  |56.3|54.1|
|ALL   |46.1                         |63.7  |60.0                                         |78.6|36.3  |37.6  |55.3|53.9|
|Skipthought|47.5                         |64.6  |55.2                                         |77.5|39.3  |40.2  |59.7|54.8|
|InferSent|45.8                         |62.9  |56.3                                         |62.2|37.3  |38.8  |52.3|50.8|
|DisSent|47.7                         |64.9  |54.8                                         |62.2|42.2  |40.7  |57.8|52.9|
|ELMo  |47.8                         |65.6  |60.7                                         |79.0|41.3  |41.8  |57.5|56.2|
|BERT base|53.1                         |68.5  |58.9                                         |80.3|41.9  |42.4  |58.8|57.7|
|BERT large|53.8                         |69.3  |59.6                                         |80.4|44.3  |43.6  |59.1|58.6|

You may notice some difference from the above table with our camera-ready version appeared on EMNLP 2019. 
The differences are: we removed the hidden states in SSP (previously 2000 by mistake), we regenerated the SP dataset (previously the sentence orders were shuffled, now the sentences are in the original order except the first sentence). 

## Reference

```
@inproceedings{mchen-discoeval-19,
  author    = {Mingda Chen and Zewei Chu and Kevin Gimpel},
  title     = {Evaluation Benchmarks and Learning Criteria for Discourse-Aware Sentence Representations},
  booktitle = {Proc. of {EMNLP}},
  year      = {2019}
}
```

