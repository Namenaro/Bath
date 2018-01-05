# -*- coding: utf-8 -*

import numpy as np
from math import floor

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import TensorBoard
from keras.regularizers import l2, l1
from keras import optimizers
from keras import backend as K
from tensorflow.contrib.learn.python.learn.estimators.head import loss_only_head

from bath.sess_hist import SessionHistory

np.random.seed(6)
class PreservationReward:
    """
    Однослойный автокодировщик.
    """
    def __init__(self):
        self.model = None

    def evaluate_session(self, session_history):
        foveal_dataset = session_history.get_foveal_dataset()  # последовательнсть, в коорой мы ищем закон сохранения
        len_of_input_space = len(foveal_dataset[0])
        # ГИПЕРПАРАМЕТРЫ--------------------------
        num_epochs = 1000
        dropout_rate = 0.2
        code_len = floor(len_of_input_space/2)
        # ГИПЕРПАРАМЕТРЫ--------------------------
        self._create_model(len_of_input_space, hidden_len=code_len, dropout_rate=dropout_rate)
        loss_before = self.model.evaluate(x=foveal_dataset, y=foveal_dataset)
        self._fit_model(foveal_dataset, num_epochs=num_epochs)
        loss_after = self.model.evaluate(x=foveal_dataset, y=foveal_dataset)
        return self._main_formula_reinforcement(loss_before, loss_after)

    def _create_model(self, flatten_data_len, hidden_len, dropout_rate):
        input = Input(shape=(flatten_data_len,), name='input')
        dinput = Dropout(rate=dropout_rate)(input)
        hidden = Dense(hidden_len,
                       activation='sigmoid',
                       name='hidden'
                       )(dinput)
        dhidden = Dropout(rate=dropout_rate)(hidden)
        prediction = Dense(flatten_data_len,
                           activation='linear',
                           name='prediction'
                           )(dhidden)
        self.model = Model(inputs=[input], outputs=prediction)

    def _fit_model(self, dataset, num_epochs):
        sgd = optimizers.SGD(lr=0.1, momentum=0.9, decay=0.01, nesterov=True)
        self.model.compile(loss=losses.mean_squared_error,
                           optimizer=sgd
                           )
        self.model.fit({'input': dataset},
                       {'prediction': dataset},
                       epochs=num_epochs,
                       shuffle=True,
                       batch_size=1,
                       validation_data=({'input': dataset},{'prediction': dataset}))

    def _main_formula_reinforcement(self, loss_before, loss_after):
        # награда тем больше, чем лучше выученная реконструкция
        reconstruction_goodness = loss_before - loss_after
        # используемость информации от входа против заучивания среднего
        last_layer = self.model.get_layer( name='decoder')
        weights = last_layer.get_weights()[0]
        biases = last_layer.get_weights()[1]
        mean_w = np.mean(a=weights)
        mean_b = np.mean(a=biases)
        information_flow_measure = mean_w - mean_b
        reinforcement = reconstruction_goodness + information_flow_measure
        return reinforcement

