#!/bin/bash
python train_mfcc.py kws1_v1_1 --num-heads 1
python train_mfcc.py kws2_v1_1 --num-heads 2
python train_mfcc.py kws3_v1_1 --num-heads 3 --batch-cut 2

python train_mfcc.py kws1_v2_1 --num-heads 1 --version 2
python train_mfcc.py kws2_v2_1 --num-heads 2 --version 2
python train_mfcc.py kws3_v2_1 --num-heads 3 --batch-cut 2 --version 2
