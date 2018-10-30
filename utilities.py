from __future__ import print_function, division
import numpy as np
import pickle
import os


def name_stem(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def get_column(_lines, index):
    return [l[index] for l in _lines]


def code2ch(codepoint):
    return chr(int(codepoint[2:], base=16))


def count_errors(_pred, _true):
    comp = [np.argmax(b, axis=1) != np.argmax(a, axis=1) for a, b in zip(_pred, _true)]
    errors = [100 * np.sum(x) / len(x) for x in comp]

    wer = np.mean(errors)  # WER
    ser = np.mean(np.logical_or.reduce(comp)) * 100  # SER

    return [ser, wer] + errors
