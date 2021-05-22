

import numpy as np
import pandas as pd

import tokenizers
import tensorflow.keras as keras


from transformers import TFRobertaModel, RobertaConfig, BertConfig
from sklearn.model_selection import StratifiedKFold
import tensorflow.keras.backend as K
import tensorflow as tf
class Keras_Conv1D(keras.layers.Conv1D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(keras.layers.Conv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs)


class Keras_Dropout(keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)
    def call(self, inputs, training=None):
        return super().call(inputs,training)
class Keras_Flatten(keras.layers.Flatten):
    def __init__(self, data_format=None, **kwargs):
        super().__init__(data_format=data_format, **kwargs)

class Keras_Activation(keras.layers.Activation):
    def __init__(self, activation, **kwargs):
        super().__init__(activation, **kwargs)
    def call(self,inputs):
        return super().call(inputs)
class Keras_Model(keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
class Roberta_Config(RobertaConfig):
    def __init__(self,pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    def Roberta(self,path):
        return RobertaConfig.from_pretrained(path)
class Stra_Kfold(StratifiedKFold):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)
class TFRoberta_Model(TFRobertaModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
    def call_model(self,path):
        return  super().from_pretrained(path,config=self.config)

