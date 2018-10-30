from __future__ import print_function
import numpy as np
from utilities import get_column


def phoneme_vec(size, position_map, phoneme):
    ret = np.zeros(size)
    if len(phoneme) == 0:
        return ret.reshape((1, size))

    ret[position_map[phoneme]] = 1

    return ret.reshape((1, size))


class SyllableMap:
    def __init__(self, symbol2int, int2symbol, sizes):
        self.symbol2int = symbol2int
        self.int2symbol = int2symbol
        self.sizes = sizes
        self.offsets = np.cumsum(self.sizes)

    def encode(self, onset, nucleus, coda):
        return [phoneme_vec(self.sizes[0], self.symbol2int[0], onset),
                phoneme_vec(self.sizes[1], self.symbol2int[1], nucleus),
                phoneme_vec(self.sizes[2], self.symbol2int[2], coda)]

    def encode_lines(self, syllables):
        rows = [self.encode(*s.split(',')) for s in syllables]
        return [np.vstack(get_column(rows, idx)) for idx in range(3)]

    def decode(self, word_vectors):
        subsyl_vectors = np.split(word_vectors, self.offsets, axis=1)
        assert subsyl_vectors[-1].shape[1] == 0
        subsyl_vectors = subsyl_vectors[:-1]
        indices = [x.argmax(axis=1) for x in subsyl_vectors]

        subsyl_symbols = []
        for v, m in zip(indices, self.int2symbol):
            subsyl_symbols.append(np.vectorize(m.__getitem__)(v))

        return [' '.join(x) for x in zip(*subsyl_symbols)]
