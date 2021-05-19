from Library import library


class Model1():
    def __init__(self,  MAX_LEN,PATH):
        self.MAX_LEN=MAX_LEN
        self.PATH=PATH
        self.ids = library.keras.layers.Input((MAX_LEN,), dtype=library.tf.int32)
        self.att = library.keras.layers.Input((MAX_LEN,), dtype=library.tf.int32)
        self.tok = library.keras.layers.Input((MAX_LEN,), dtype=library.tf.int32)

    def buil(self):
        config = library.transformers.RobertaConfig.from_pretrained(self.PATH + 'config-roberta-base.json')
        bert_model = library.transformers.TFRobertaModel.from_pretrained(self.PATH + 'pretrained-roberta-base.h5', config=config)
        x = bert_model(self.ids, attention_mask=self.att, token_type_ids=self.tok)

        x1 = library.keras.layers.Dropout(0.1)(x[0])
        x1 = library.keras.layers.Conv1D(1, 1)(x1)
        x1 = library.keras.layers.Flatten()(x1)
        x1 = library.keras.layers.Activation('softmax')(x1)

        x2 = library.keras.layers.Dropout(0.1)(x[0])
        x2 = library.keras.layers.Conv1D(1, 1)(x2)
        x2 = library.keras.layers.Flatten()(x2)
        x2 = library.keras.layers.Activation('softmax')(x2)

        model = library.tf.keras.models.Model(inputs=[self.ids, self.att, self.tok], outputs=[x1, x2])
        optimizer = library.tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model
    @staticmethod
    def jaccard(str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        if (len(a) == 0) & (len(b) == 0): return 0.5
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
