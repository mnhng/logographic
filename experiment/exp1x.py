from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam

from .keras_base import KerasBaseExp
from .keras_base import exp_bag_of_strokes

from .blocks import fc_branch, final_type1


class mlp_type1(KerasBaseExp):
    def initialize_model(self, in_dims, out_dims):
        input_layer = [Input(shape=(d, )) for d in in_dims]

        if len(input_layer) > 1:
            layer = Concatenate()(input_layer)
        else:
            layer = input_layer[0]

        layer = fc_branch(layer, self.decay)

        self.model = Model(inputs=input_layer, outputs=final_type1(layer, out_dims))
        opt = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')


class EXP1(mlp_type1, exp_bag_of_strokes):
    pass
