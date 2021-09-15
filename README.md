# SACE: Sense Aware Context Exploitation (SACE) Architecture
This is the source code for SACE, built with [BEM](https://github.com/facebookresearch/wsd-biencoders) modules.

SACE implements a selective attention layer upon the original gloss encoder and a sentence selector before the context encoder. A try-again mechanism is also implemented after the training process.

## Dependencies
To run this code, you'll need the following libraries:
* [Python 3](https://www.python.org/)
* [Pytorch 1.6.0](https://pytorch.org/)
* [Pytorch-Transformers 1.2.0](https://github.com/huggingface/transformers)
* [Transformers 3.2.0](https://github.com/huggingface/transformers)
* [Numpy 1.17.2](https://numpy.org/)
* [NLTK 3.5](https://www.nltk.org/)
* [tqdm](https://tqdm.github.io/)
* [apex 0.1](https://tqdm.github.io/)

We used the [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) for training and evaluating our model. Download the evaluation framework for convenient employment.

For cross-lingual datasets, we use [mwsd-datasets](https://github.com/SapienzaNLP/mwsd-datasets).

For WordNet Tagged Gloss (WNGT), we use [UFSAC](https://github.com/getalp/UFSAC).

## Train
Use the following code to train the base model. It takes 6 hours to finish the job using early stopping (3 epoch without updating).
```bash
python biencoder-context.py --gloss-bsz 400 --epoch 10 --gloss_max_length 32 --step_mul 50 --warmup 10000 --gloss_mode sense-pred --lr 1e-5 --word word --encoder-name roberta-base --train_mode roberta-base --context_len 2 --train_data semcor --same --sec_wsd
```

For the large model, run the following code.
```bash
python biencoder-context.py --gloss-bsz 150 --epoch 10 --gloss_max_length 32 --step_mul 50 --warmup 10000 --gloss_mode sense-pred --lr 1e-6 --word word --encoder-name roberta-large --train_mode roberta-large --context_len 2 --train_data semcor --same --sec_wsd
```

For the large model that trains with more training data (SemCor+WNGT+WNE), run the following code.
```bash
python biencoder-context.py --gloss-bsz 150 --epoch 10 --gloss_max_length 48 --step_mul 50 --warmup 10000 --gloss_mode sense-pred --lr 1e-6 --word non --encoder-name roberta-large --train_mode roberta-large --context_len 2 --train_data semcor-wngt --same
```

For the multilingual model that trains with more training data (SemCor+WNGT+WNE), run the following code.
```bash
python biencoder-context.py --gloss-bsz 400 --epoch 10 --gloss_max_length 48 --step_mul 50 --warmup 10000 --gloss_mode sense-pred --lr 5e-6 --word non --encoder-name xlmroberta-base --train_mode xlmroberta-base --context_len 2 --train_data semcor-wngt
```

## Evaluate
To evaluate the base model, run:
```bash
python biencoder-context.py --gloss-bsz 400 --epoch 10 --gloss_max_length 32 --step_mul 50 --warmup 10000 --gloss_mode sense-pred --lr 1e-5 --word word --encoder-name roberta-base --train_mode roberta-base --context_len 2 --train_data semcor --same --sec_wsd --eval
```

## License
This codebase is Attribution-NonCommercial 4.0 International licensed, as found in the [LICENSE](https://github.com/facebookresearch/wsd-biencoders/blob/master/LICENSE) file.


| Systems    | SE2  | SE3  | SE07  | SE13 | SE15  | ALL   | N    | V    | A    | R    |
| ---------- | ---- | ---- | ----- | ---- | ----- | ----- | ---- | ---- | ---- | ---- |
| SACEbase   | 80.9 | 79.1 | 74.7* | 82.4 | 84.6  | 80.9* | 83.2 | 71.1 | 85.4 | 87.9 |
| SACElarge  | 82.4 | 81.1 | 76.3* | 82.5 | 83.7  | 81.9* | 84.1 | 72.2 | 86.4 | 89.0 |
| SACElarge+ | 83.6 | 81.4 | 77.8  | 82.4 | 87.3* | 82.9* | 85.3 | 74.2 | 85.9 | 87.3 |
  
Please cite:  
{wang-wang-2021-word,
    title = "Word Sense Disambiguation: Towards Interactive Context Exploitation from Both Word and Sense Perspectives",
    author = "Wang, Ming  and
      Wang, Yinglin",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.406",
    doi = "10.18653/v1/2021.acl-long.406",
    pages = "5218--5229"
}