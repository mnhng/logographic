#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import random
from ids_conv import get_converter
from utilities import code2ch


def main(args):
    converter = get_converter()
    random.seed(args.seed)
    lines = [l.strip().split('\t') for l in open(args.datapath, 'r')]
    for _ in range(args.n):
        print_sample(converter, *lines[random.randrange(len(lines))])


def print_sample(conv, codepoint, M, C, K, V):
    print('codepoint:', codepoint, '\tlogograph:', code2ch(codepoint))
    print('Mandarin:', M, '\tCantonese:', C, '\tKorean:', K, '\tVietnamese:', V)
    print('GeoD:', conv.to_seq(codepoint))
    print()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', '-d', required=True)
    parser.add_argument('-n', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
