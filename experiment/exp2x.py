from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam

from .keras_base import KerasBaseExp
from .keras_base import exp_idsq
from .keras_base import exp_idsq_deepsupervision

from .blocks import fc_branch, final_type1
from .blocks import final_type1_aux


class lstm_model(KerasBaseExp):
    def initialize_model(self, in_dims, out_dims):
        assert len(in_dims) == 1

        layer = Input(shape=(self.ids_conv.MAX_SEQ_LEN, in_dims[0]))
        input_layer = [layer]

        for _ in range(self.n_layers - 1):
            layer = LSTM(self.hid_size, dropout=self.idrop, recurrent_dropout=self.rdrop,
                         return_sequences=True, return_state=False,
                         go_backwards=False, unroll=True)(layer)

        layer = LSTM(self.hid_size, dropout=self.idrop, recurrent_dropout=self.rdrop,
                     return_sequences=False, return_state=False,
                     go_backwards=False, unroll=True)(layer)

        self.model = Model(inputs=input_layer, outputs=final_type1(layer, out_dims))
        opt = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')
        self.model.summary()


class EXP2A(lstm_model, exp_idsq):
    pass


class early_fusion_dsp_model(KerasBaseExp):
    def initialize_model(self, in_dims, out_dims):
        input_layer = []
        aux_layer = []
        outputs = []
        for d in in_dims:
            if d < 200:  # TODO: fix this hack
                _l = Input(shape=(d, ))
                input_layer.append(_l)
                aux_layer.append(_l)
            else:
                _l = Input(shape=(self.ids_conv.MAX_SEQ_LEN, d))
                input_layer.append(_l)

                for _ in range(self.n_layers - 1):
                    _l = LSTM(self.hid_size, dropout=self.idrop, recurrent_dropout=self.rdrop,
                                 return_sequences=True, return_state=False,
                                 go_backwards=False,
                                 unroll=True)(_l)

                _l = LSTM(self.hid_size, dropout=self.idrop, recurrent_dropout=self.rdrop,
                             return_sequences=False, return_state=False,
                             go_backwards=False,
                             unroll=True)(_l)

                aux_layer.append(_l)
                outputs.extend(final_type1_aux(_l, out_dims))

        layer = fc_branch(Concatenate()(aux_layer), self.decay)
        outputs.extend(final_type1(layer, out_dims))

        self.model = Model(inputs=input_layer, outputs=outputs)
        opt = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')
        self.model.summary()


class EXP2B(early_fusion_dsp_model, exp_idsq_deepsupervision):
    pass
