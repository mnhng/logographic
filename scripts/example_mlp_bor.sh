#!/bin/bash
FILES="--training_set data/train.csv \
       --dev_set data/dev.csv \
       --test_set data/test.csv"

# BoR -> Cantonese phoneme
./train.py $FILES \
            --batch_size 128 \
            --decay 1e-4 \
            --experiment EXP1 \
            --iters 100 \
            --learning_rate 1e-3 \
            -s 0 -t 2
