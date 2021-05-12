import numpy as np
import pandas as pd
import tokenizers
import tensorflow.keras as keras
import transformers
from sklearn.model_selection import StratifiedKFold
import tensorflow.keras.backend as K
import tensorflow as tf
# !pip install transformers

# !pip3 install sentencepiece
# !pip3 install tf_sentencepiece
print('TF version', tf.__version__)
print("Eager execution: {}".format(tf.executing_eagerly()))
MAX_LEN = 96
PATH = '/content/drive/MyDrive/NLP/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab=PATH+'vocab-roberta-base.json',
    merges=PATH+'merges-roberta-base.txt',
    lowercase=True,
    add_prefix_space=True
)
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = pd.read_csv('/content/drive/MyDrive/NLP/train.csv').fillna('')
test = pd.read_csv('/content/drive/MyDrive/NLP/test.csv').fillna('')
train.head()


class Process_data():
    def __init__(self, ct1, ct2, MAX_LEN):
        self.ct1 = ct1
        self.ct2 = ct2
        self.MAX_LEN = MAX_LEN
        self.input_ids = np.ones((ct1, MAX_LEN), dtype='int32')
        self.attention_mask = np.zeros((ct1, MAX_LEN), dtype='int32')
        self.token_type_ids = np.zeros((ct1, MAX_LEN), dtype='int32')
        self.start_tokens = np.zeros((ct1, MAX_LEN), dtype='int32')
        self.end_tokens = np.zeros((ct1, MAX_LEN), dtype='int32')
        self.input_ids_t = np.ones((ct2, MAX_LEN), dtype='int32')
        self.attention_mask_t = np.zeros((ct2, MAX_LEN), dtype='int32')
        self.token_type_ids_t = np.zeros((ct2, MAX_LEN), dtype='int32')

    @staticmethod
    def FIND_OVERLAP(k):
        text1 = " " + " ".join(train.loc[k, 'text'].split())
        text2 = " ".join(train.loc[k, 'selected_text'].split())
        idx = text1.find(text2)
        chars = np.zeros((len(text1)))
        chars[idx:idx + len(text2)] = 1
        if text1[idx - 1] == ' ':
            chars[idx - 1] = 1
        enc = tokenizer.encode(text1)
        return idx, chars, enc

    @staticmethod
    def ID_OFFSETS(enc=None):
        offsets = []
        idx = 0
        for t in enc.ids:
            w = tokenizer.decode([t])
            offsets.append((idx, idx + len(w)))
            idx += len(w)
        return offsets

    def START_END_TOKENS(self, offsets, chars=None, k=None, enc=None):
        toks = []
        for i, (a, b) in enumerate(offsets):
            sm = np.sum(chars[a:b])
            if sm > 0:
                toks.append(i)

        s_tok = sentiment_id[train.loc[k, 'sentiment']]
        self.input_ids[k, :len(enc.ids) + 5] = [0] + \
            enc.ids + [2, 2] + [s_tok] + [2]
        self.attention_mask[k, :len(enc.ids) + 5] = 1
        if len(toks) > 0:
            self.start_tokens[k, toks[0] + 1] = 1
            self.end_tokens[k, toks[-1] + 1] = 1

    def INPUT_IDS(self, k):
        text1 = " " + " ".join(test.loc[k, 'text'].split())
        enc = tokenizer.encode(text1)
        s_tok = sentiment_id[test.loc[k, 'sentiment']]
        self.input_ids_t[k, :len(enc.ids) + 5] = [0] + \
            enc.ids + [2, 2] + [s_tok] + [2]
        self.attention_mask_t[k, :len(enc.ids) + 5] = 1

    def Set(self):
        return self.input_ids


a = Process_data(train.shape[0], test.shape[0], MAX_LEN)

for k in range(train.shape[0]):
    idx, chars, enc = a.FIND_OVERLAP(k)
    offsets = a.ID_OFFSETS(enc)
    a.START_END_TOKENS(offsets, chars, k, enc)
for k in range(test.shape[0]):
    a.INPUT_IDS(k)


class Model1(tf.keras.Model):
    def __init__(self,  MAX_LEN, PATH):
        super(Model1, self).__init__()
        self.MAX_LEN = MAX_LEN
        self.PATH = PATH
        self.ids = keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        self.att = keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        self.tok = keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        # self.Dropout=keras.layers.Dropout(0.1)
        # self.Conv1D=keras.layers.Conv1D(1, 1)
        # self.Flatten=keras.layers.Flatten()
        # self.Activation=keras.layers.Activation('softmax')

    def buil(self):
        config = transformers.RobertaConfig.from_pretrained(
            self.PATH + 'config-roberta-base.json')
        bert_model = transformers.TFRobertaModel.from_pretrained(
            self.PATH + 'pretrained-roberta-base.h5', config=config)
        x = bert_model(self.ids, attention_mask=self.att,
                       token_type_ids=self.tok)

        # x1 = self.Dropout(x[0])
        # x1 = self.Conv1D(x1)
        # x1 = self.Flatten(x1)
        # x1 = self.Activation(x1)

        # x2 = self.Dropout(x[0])
        # x2 = self.Conv1D(x2)
        # x2 = self.Flatten(x2)
        # x2 = self.Activation(x2)
        x1 = tf.keras.layers.Dropout(0.1)(x[0])
        x1 = tf.keras.layers.Conv1D(1, 1)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Activation('softmax')(x1)

        x2 = tf.keras.layers.Dropout(0.1)(x[0])
        x2 = tf.keras.layers.Conv1D(1, 1)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Activation('softmax')(x2)

        model = tf.keras.models.Model(
            inputs=[self.ids, self.att, self.tok], outputs=[x1, x2])
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    @staticmethod
    def jaccard(str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        if (len(a) == 0) & (len(b) == 0):
            return 0.5
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))


class Train(Model1):
    def __init__(self, MAX_LEN, PATH, input_ids, input_ids_t, attention_mask, attention_mask_t, token_type_ids, token_type_ids_t, start_tokens, end_tokens):
        super().__init__(MAX_LEN, PATH)
        self.DISPLAY = 1
        self.VER = 'v0'
        self.jac = []
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.input_ids_t = input_ids_t
        self.token_type_ids = token_type_ids
        self.token_type_ids_t = token_type_ids_t
        self.attention_mask_t = attention_mask_t
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
        self.oof_start = np.zeros((input_ids.shape[0], MAX_LEN))
        self.oof_end = np.zeros((input_ids.shape[0], MAX_LEN))
        self.preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
        self.preds_end = np.zeros((input_ids_t.shape[0], MAX_LEN))

    def Acu(self):
        print('>>>> OVERALL 5Fold CV Jaccard =', np.mean(self.jac))

    def Train_model(self):
        for fold, (idxT, idxV) in enumerate(self.skf.split(self.input_ids, train.sentiment.values)):
            print('#' * 25)
            print('### FOLD %i' % (fold + 1))
            print('#' * 25)
            K.clear_session()
            model = super().buil()

            sv = tf.keras.callbacks.ModelCheckpoint(
                '%s-roberta-%i.h5' % (self.VER, fold), monitor='val_loss', verbose=1, save_best_only=True,
                save_weights_only=True, mode='auto', save_freq='epoch')

            model.fit([self.input_ids[idxT, ], self.attention_mask[idxT, ], self.token_type_ids[idxT, ]],
                      [self.start_tokens[idxT, ], self.end_tokens[idxT, ]],
                      epochs=3, batch_size=32, verbose=self.DISPLAY, callbacks=[sv],
                      validation_data=([self.input_ids[idxV, ], self.attention_mask[idxV, ], self.token_type_ids[idxV, ]],
                                       [self.start_tokens[idxV, ], self.end_tokens[idxV, ]]))

            print('Loading model...')
            model.load_weights('%s-roberta-%i.h5' % (self.VER, fold))

            print('Predicting OOF...')
            self.oof_start[idxV, ], self.oof_end[idxV, ] = model.predict(
                [self.input_ids[idxV, ], self.attention_mask[idxV, ], self.token_type_ids[idxV, ]], verbose=self.DISPLAY)

            print('Predicting Test...')
            preds = model.predict(
                [self.input_ids_t, self.attention_mask_t, self.token_type_ids_t], verbose=self.DISPLAY)
            self.preds_start += preds[0] / self.skf.n_splits
            self.preds_end += preds[1] / self.skf.n_splits

            # DISPLAY FOLD JACCARD
            all = []
            for k in idxV:
                a = np.argmax(self.oof_start[k, ])
                b = np.argmax(self.oof_end[k, ])
                if a > b:
                    # IMPROVE CV/LB with better choice here
                    st = train.loc[k, 'text']
                else:
                    text1 = " " + " ".join(train.loc[k, 'text'].split())
                    enc = tokenizer.encode(text1)
                    st = tokenizer.decode(enc.ids[a - 1:b])
                all.append(super().jaccard(st, train.loc[k, 'selected_text']))
            self.jac.append(np.mean(all))
            print('>>>> FOLD %i Jaccard =' % (fold + 1), np.mean(all))
            print()


bui = Train(MAX_LEN, PATH, a.input_ids, a.input_ids_t, a.attention_mask,
            a.attention_mask_t, a.token_type_ids, a.token_type_ids_t, a.start_tokens, a.end_tokens)
bui.Train_model()
bui.Acu()
