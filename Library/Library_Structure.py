'''
    Đây là module có chức năng:
        +Import đầy đủ các thư viện cho cả project
        +Xây dựng các class kế thừa thư viện có sẵn.
'''
import numpy as np
import pandas as pd

import tokenizers
import tensorflow.keras as keras

from transformers import TFRobertaModel, RobertaConfig, BertConfig
from sklearn.model_selection import StratifiedKFold
import tensorflow.keras.backend as K
import tensorflow as tf
class Keras_Conv1D(keras.layers.Conv1D):
    '''
        Class kế thừa keras.layers.Conv1D: nói đơn giản Conv1D có chức năng tổng hợp và thu nhỏ ma trận input.
        ngoài ra Conv1D thường được xử dụng khi xử lí văn bản còn Conv2D xử lí ảnh.
    '''
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
    '''
        Dropout: bỏ qua ngẫu nhiên các unit trong quá trình train nhằm giảm sự phụ thuộc vào nhau giữa các neural
        và giúp tránh tình trạng over-fitting.
    '''
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)
    def call(self, inputs, training=None):
        return super().call(inputs,training)
    
class Keras_Flatten(keras.layers.Flatten):
    '''
        Flatten giúp reshape lại ma trận thành mảng đơn. Giúp đơn giản output.
    '''
    def __init__(self, data_format=None, **kwargs):
        super().__init__(data_format=data_format, **kwargs)

class Keras_Activation(keras.layers.Activation):
    '''
        Activation là hàm kích hoạt. Hàm kích hoạt đóng vai trò là thành phần phi tuyến tại output của các nơ-ron.
        Hiểu đơn giản là nếu không có nó thì output cũng chỉ đơn giản là nhận input với các weight mà thôi.
    '''
    def __init__(self, activation, **kwargs):
        super().__init__(activation, **kwargs)
    def call(self,inputs):
        return super().call(inputs)
    
class Keras_Model(keras.models.Model):
    '''
        Nhóm các lớp lại để có thể dễ dàng trainning
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class Roberta_Config(RobertaConfig):
    '''
        Cấu hình roBERTa.
    '''
    def __init__(self,pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    def Roberta(self,path):
        return RobertaConfig.from_pretrained(path)
    
class Stra_Kfold(StratifiedKFold):
    '''
        Chia nhỏ data để train và đánh giá.
    '''
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)
        
class TFRoberta_Model(TFRobertaModel):
    '''
        Model RoBERTa.
    '''
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
    def call_model(self,path):
        return  super().from_pretrained(path,config=self.config)

