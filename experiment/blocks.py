from keras.layers import Dense
from keras.layers.core import Dropout
from keras.layers.merge import Concatenate
from keras.regularizers import l2


def fc_branch(inputs, reg):
    fc1 = Dense(750, activation='relu', kernel_regularizer=l2(reg))(inputs)
    drop1 = Dropout(.5)(fc1)
    fc2 = Dense(500, activation='relu', kernel_regularizer=l2(reg))(drop1)
    drop2 = Dropout(.5)(fc2)
    fc3 = Dense(250, activation='relu', kernel_regularizer=l2(reg))(drop2)
    drop3 = Dropout(.2)(fc3)

    return drop3


def final_type0(h, dims):
    on = Dense(dims[0], activation='softmax', name='ON')(h)
    nu = Dense(dims[1], activation='softmax', name='NU')(h)
    cd = Dense(dims[2], activation='softmax', name='CD')(h)
    return [on, nu, cd]


def final_type1(h, dims):
    cd = Dense(dims[2], activation='softmax', name='CD')(h)
    h1 = Concatenate()([h, cd])
    nu = Dense(dims[1], activation='softmax', name='NU')(h1)
    h2 = Concatenate()([h, nu, cd])
    on = Dense(dims[0], activation='softmax', name='ON')(h2)
    return [on, nu, cd]


def final_type1_aux(h, dims):
    cd = Dense(dims[2], activation='softmax', name='CD_aux')(h)
    h1 = Concatenate()([h, cd])
    nu = Dense(dims[1], activation='softmax', name='NU_aux')(h1)
    h2 = Concatenate()([h, nu, cd])
    on = Dense(dims[0], activation='softmax', name='ON_aux')(h2)
    return [on, nu, cd]


def final_type2(h, dims):
    nu = Dense(dims[1], activation='softmax', name='NU')(h)
    h1 = Concatenate()([h, nu])
    cd = Dense(dims[2], activation='softmax', name='CD')(h1)
    h2 = Concatenate()([h, nu, cd])
    on = Dense(dims[0], activation='softmax', name='ON')(h2)
    return [on, nu, cd]
