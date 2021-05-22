'''
    Module build model và độ đo.
'''
from Library import Library_Structure as library

class Model_RoBERTa(library.Roberta_Config):
    '''
        Class thiết lập, xây dựng model.
    '''
    def __init__(self, MAX_LEN, PATH, pad_token_id=1, bos_token_id=0, eos_token_id=2):
        super(library.Roberta_Config, self).__init__(
            pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        self.MAX_LEN = MAX_LEN
        self.PATH = PATH
        self.ids = library.keras.layers.Input(
            (MAX_LEN,), dtype=library.tf.int32)
        self.att = library.keras.layers.Input(
            (MAX_LEN,), dtype=library.tf.int32)
        self.tok = library.keras.layers.Input(
            (MAX_LEN,), dtype=library.tf.int32)

    def Build_model(self):
        '''
            Hàm build model.
        :return: Model
        '''
        config = super().Roberta(self.PATH + 'config-roberta-base.json')
        a = library.TFRoberta_Model(config)
        bert_model = a.call_model(self.PATH + 'pretrained-roberta-base.h5')
        x = bert_model(self.ids, attention_mask=self.att,
                       token_type_ids=self.tok)
        x1 = library.Keras_Dropout(0.1).call(x[0])
        x1 = library.Keras_Conv1D(1, 1)(x1)
        x1 = library.Keras_Flatten()(x1)
        x1 = library.Keras_Activation('softmax').call(x1)

        x2 = library.Keras_Dropout(0.1).call(x[0])
        x2 = library.Keras_Conv1D(1, 1)(x2)
        x2 = library.Keras_Flatten()(x2)
        x2 = library.Keras_Activation('softmax').call(x2)

        model = library.Keras_Model(
            inputs=[self.ids, self.att, self.tok], outputs=[x1, x2])
        optimizer = library.tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    @staticmethod
    def Jaccard(str1, str2):
        '''
            Hàm xây dựng độ đo dùng trong project. Đơn giản là giao các kết quả sau khi chia nhỏ để đánh giá.
        :param str1: Phần đánh giá 1.
        :param str2: Phần đánh giá 2.
        :return: Kết quả.
        '''
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        if (len(a) == 0) & (len(b) == 0):
            return 0.5
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
