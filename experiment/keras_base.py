import abc
import numpy as np
import os

from keras.callbacks import CSVLogger

from utilities import name_stem, count_errors
from utilities import get_column

from .common_base import BaseExp


class KerasBaseExp(BaseExp):
    def train(self, *filepaths):
        inp_mats = []
        out_mats = []
        for fp in filepaths:
            _x, _y = self.load_dataset(fp)
            inp_mats.append(_x)
            out_mats.append(_y)

        x_true = [np.vstack(mat) for mat in zip(*inp_mats)]
        y_true = [np.vstack(mat) for mat in zip(*out_mats)]
        return self._train_impl(x_true, y_true)

    def _train_impl(self, x_true, y_true):
        in_dims = sum([self.IN_DIMS_DICT[s] for s in self.sources], [])
        self.initialize_model(in_dims, self.OUT_DIMS_DICT[self.target])

        print([mat.shape for mat in x_true])
        print([mat.shape for mat in y_true])

        stat_file = os.path.join(self.folder, 'loss.csv')
        self.model.fit(x_true, y_true, epochs=self.iters,
                       batch_size=self.batch_size,
                       shuffle=True,
                       callbacks=[CSVLogger(stat_file)],
                       verbose=2)

        self.model.save(os.path.join(self.folder, 'model.h5'), True)
        print(self.folder)

        return self

    def test(self, filepath):
        x_true, y_true = self.load_dataset(filepath)
        return self._test_impl(x_true, y_true, name_stem(filepath))

    def _test_impl(self, x_true, y_true, prefix):
        y_pred = self.model.predict(x_true)

        return count_errors(y_pred, y_true)

    @abc.abstractmethod
    def load_dataset(self, filepath):
        lines = [l.strip('\n').split('\t') for l in open(filepath)]

        data_in = []
        for idx, m in zip(self.sources, self.encoders):
            data_in += m.encode_lines(get_column(lines, idx))
        data_out = self.decoder.encode_lines(get_column(lines, self.target))

        return data_in, data_out


class exp_bag_of_strokes(KerasBaseExp):
    def load_dataset(self, filepath):
        self.ids_conv.set_encoding('bor')
        return super(exp_bag_of_strokes, self).load_dataset(filepath)


class exp_idsq(KerasBaseExp):
    def load_dataset(self, filepath):
        self.ids_conv.set_encoding('idsq')
        return super(exp_idsq, self).load_dataset(filepath)


class exp_idsq_deepsupervision(exp_idsq):
    def load_dataset(self, filepath):
        data_in, data_out = super(exp_idsq_deepsupervision, self).load_dataset(filepath)
        return data_in, data_out * 2  # two outputs

    def test(self, filepath):
        x_true, y_true = self.load_dataset(filepath)
        y_pred = self.model.predict(x_true)
        print('aux error', *count_errors(y_pred[:3], y_true[:3]))
        # remove aux_outputs for deep supervision
        y_true, y_pred = y_true[-3:], y_pred[-3:]

        return count_errors(y_pred, y_true)
