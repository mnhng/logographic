import abc
import json
import os
import time

from encoding import language_xsampa_maps, language_direct_maps
from encoding import xsampa_consonant, xsampa_vowel
from mapping import SyllableMap


class BaseExp(metaclass=abc.ABCMeta):
    _xsampa_dims = [len(xsampa_consonant),
                    len(xsampa_vowel),
                    len(xsampa_consonant)]
    IN_DIMS_DICT = {0: [-1],
                    1: _xsampa_dims,
                    2: _xsampa_dims,
                    3: _xsampa_dims,
                    4: _xsampa_dims}
    OUT_DIMS_DICT = {1: [24, 21, 4], 2: [20, 22, 7],
                     3: [17, 20, 9], 4: [32, 55, 13]}

    def __init__(self, ids_conv, **kwargs):
        self.ids_conv = ids_conv
        self.IN_DIMS_DICT[0] = [len(self.ids_conv.basic_units)]
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.encoders = []
        for i in self.sources:
            if i == 0:
                enc = self.ids_conv
            else:
                enc = SyllableMap(*language_xsampa_maps[i], sizes=self._xsampa_dims)
            self.encoders.append(enc)

        decoder_maps = language_direct_maps[self.target]
        # decoder_maps = language_xsampa_maps[self.target]
        sizes = [len(x) for x in decoder_maps[0]]
        self.decoder = SyllableMap(*decoder_maps, sizes=sizes)

        attempt = 0
        while attempt < 3:
            timestamp = time.strftime('%m%d-%H%M%S', time.localtime())
            exp_dir = os.path.join('output', type(self).__name__, timestamp)
            try:
                os.makedirs(exp_dir)
                break
            except OSError:
                attempt += 1
                import random
                time.sleep(random.randrange(0, 15))
        assert attempt < 3, 'Failed to create %s' % exp_dir
        self.folder = exp_dir

        with open(os.path.join(self.folder, 'params.json'), 'w') as fh:
            out = json.dumps(kwargs, sort_keys=True, indent=4,
                             separators=(',', ': '))
            print(out, file=fh)

    @abc.abstractmethod
    def load_dataset(self, filepath): pass

    @abc.abstractmethod
    def initialize_model(self, in_dims, out_dims): pass

    def decode(self, vectors):
        return self.decoder.decode(vectors)
