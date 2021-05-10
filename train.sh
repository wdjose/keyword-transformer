#!/bin/bash

python train.py kwt1_v1 --num-heads 1 --version 1
python train.py kwt2_v1 --num-heads 2 --version 1
python train.py kwt3_v1 --num-heads 3 --version 1

python train.py kwt1_v2 --num-heads 1 --version 2
python train.py kwt2_v2 --num-heads 2 --version 2
python train.py kwt3_v2 --num-heads 3 --version 2

python train.py kwt1_v3 --num-heads 1 --version 3
python train.py kwt2_v3 --num-heads 2 --version 3
python train.py kwt3_v3 --num-heads 3 --version 3
