#!/bin/sh
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -q gpu
#BSUB -o roberta-base.out
#BSUB -e roberta-base.er
#BSUB -m gpu01
python biencoder-context.py --gloss-bsz 400 --epoch 10 --gloss_max_length 32 --step_mul 50 --warmup 10000 --gloss_mode sense-pred --lr 1e-5 --word word --encoder-name roberta-base --train_mode roberta-base --context_len 2 --train_data semcor --same --sec_wsd >> roberta-base.out
