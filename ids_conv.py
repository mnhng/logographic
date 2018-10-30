#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
from utilities import code2ch

from inventory import terminal_set

IDC = {'⿰': 2, '⿱': 2, '⿲': 3, '⿳': 3,
       '⿴': 2, '⿵': 2, '⿶': 2, '⿷': 2,
       '⿸': 2, '⿹': 2, '⿺': 2, '⿻': 2}


def parse_ids_rules(filepath):
    lines = [l.strip().split('\t') for l in open(filepath)]
    return {l[1]: l[2] for l in lines if len(l) > 1 and l[1] != l[2]}


class ids_converter(object):
    def __init__(self, rule_files, max_len, basic_units):
        self.rules = {}
        for fp in rule_files:
            self.rules.update(parse_ids_rules(fp))

        self.MAX_SEQ_LEN = max_len
        self.basic_units = basic_units

        self.enc_func = None

    def _decompose(self, char, verbose=False):
        if verbose:
            print(char)
        if char in self.basic_units:
            return [char]
        if char in '[GHUAJKTXV]':
            return []

        # if char in missing:
            # missing[char] = missing[char] + 1
            # return ['*']
            # return []

        components = self.rules[char]

        ret = []
        for part in components:
            ret += self._decompose(part, verbose)

        return ret

    def to_seq(self, codepoint):
        ch = code2ch(codepoint)
        # try:
        ret = self._decompose(ch)
        # except:
        #     print('failed', ch, codepoint)
        #     raise
        # assert len(ret) < self.MAX_SEQ_LEN
        return ret

    def to_ids_vec(self, codepoint):
        ret = np.zeros((1, len(self.basic_units)))
        for stroke in self.to_seq(codepoint):
            ret[0, self.basic_units[stroke]] += 1
        return ret

    def to_bag_of_strokes(self, codepoint):
        ret = np.zeros((1, len(self.basic_units)))
        strokes_and_ops = self.to_seq(codepoint)
        strokes_only = [st for st in strokes_and_ops if st not in IDC]
        for stroke in strokes_only:
            ret[0, self.basic_units[stroke]] += 1
        return ret

    def to_ids_seq(self, codepoint):
        ret = np.zeros((1, self.MAX_SEQ_LEN, len(self.basic_units)), dtype=int)
        for i, stroke in enumerate(reversed(self.to_seq(codepoint))):
            ret[0, self.MAX_SEQ_LEN - 1 - i, self.basic_units[stroke]] = 1
        return ret

    def set_encoding(self, otype):
        if otype == 'bor':
            self.enc_func = self.to_bag_of_strokes
        elif otype == 'idv':
            self.enc_func = self.to_ids_vec
        elif otype == 'idsq':
            self.enc_func = self.to_ids_seq

    def encode_lines(self, codepoints):
        return [np.vstack(self.enc_func(c) for c in codepoints)]


def get_converter():
    return ids_converter(['ids.txt', 'ids_update.txt'], 30, terminal_set)
