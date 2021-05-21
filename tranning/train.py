# from Library import library
from build_model.build import Model_RoBERTa
from load_data import load


class Train(Model_RoBERTa,load.library.Stra_Kfold):
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
        self.skf = load.library.Stra_Kfold(5,True,777)
        # self.skf = load.library.StratifiedKFold(
        #     n_splits=5, shuffle=True, random_state=777)
        self.oof_start = load.library.np.zeros((input_ids.shape[0], MAX_LEN))
        self.oof_end = load.library.np.zeros((input_ids.shape[0], MAX_LEN))
        self.preds_start = load.library.np.zeros((input_ids_t.shape[0], MAX_LEN))
        self.preds_end = load.library.np.zeros((input_ids_t.shape[0], MAX_LEN))

    def Acu(self):
        print('>>>> OVERALL 5Fold CV Jaccard =', load.library.np.mean(self.jac))

    def Train_model(self):
        for fold, (idxT, idxV) in enumerate(self.skf.split(self.input_ids, load.train.sentiment.values)):
            print('#' * 25)
            print('### FOLD %i' % (fold + 1))
            print('#' * 25)
            load.library.K.clear_session()
            model = super().build_model()

            sv = load.library.tf.keras.callbacks.ModelCheckpoint(
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
                a = load.library.np.argmax(self.oof_start[k, ])
                b = load.library.np.argmax(self.oof_end[k, ])
                if a > b:
                    # IMPROVE CV/LB with better choice here
                    st = load.train.loc[k, 'text']
                else:
                    text1 = " " + " ".join(load.train.loc[k, 'text'].split())
                    enc = load.tokenizer.encode(text1)
                    st = load.tokenizer.decode(enc.ids[a - 1:b])
                all.append(super().jaccard(
                    st, load.train.loc[k, 'selected_text']))
            self.jac.append(load.library.np.mean(all))
            print('>>>> FOLD %i Jaccard =' % (fold + 1), load.library.np.mean(all))
            print()


bui = Train(load.MAX_LEN, load.PATH, load.a.input_ids, load.a.input_ids_t, load.a.attention_mask,
            load.a.attention_mask_t, load.a.token_type_ids, load.a.token_type_ids_t, load.a.start_tokens, load.a.end_tokens)
bui.Train_model()
bui.Acu()
