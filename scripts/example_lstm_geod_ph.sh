#!/bin/sh
FILES="--training_set data/train.csv \
       --dev_set data/dev.csv \
       --test_set data/test.csv"

# GeoD, Mandarin, Korean, Vietnamese phonemes -> Cantonese phoneme
./train.py $FILES \
            --batch_size 128 \
            --decay 1e-4 \
            --experiment EXP2B \
            --hid_size 256 \
            --iters 100 \
            --learning_rate 3e-3 \
            --n_layers 2 \
            --idrop .2 \
            --rdrop .7 \
            -s 0 1 3 4 -t 2
