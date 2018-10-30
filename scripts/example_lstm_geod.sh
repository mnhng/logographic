#!/bin/sh
FILES="--training_set data/train.csv \
       --dev_set data/dev.csv \
       --test_set data/test.csv"

# GeoD -> Cantonese phoneme
./train.py $FILES \
            --batch_size 128 \
            --experiment EXP2A \
            --hid_size 256 \
            --iters 100 \
            --learning_rate 3e-3 \
            --n_layers 2 \
            --idrop .2 \
            --rdrop .7 \
            -s 0 -t 2
