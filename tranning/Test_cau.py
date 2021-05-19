from load_data import load
from Library import library
from build_model import build
class Test(build.Model1):
    def __init__(self,MAX_LEN,PATH,str1,path,sentiment):
        super().__init__(MAX_LEN,PATH)
        self.MAX_LEN=MAX_LEN
        self.str1=str1
        self.path=path
        self.sentiment=sentiment
        self.model=super().buil()
        self.input_ids_t = library.np.ones((1, MAX_LEN), dtype='int32')
        self.attention_mask_t = library.np.zeros((1, MAX_LEN), dtype='int32')
        self.token_type_ids_t = library.np.zeros((1, MAX_LEN), dtype='int32')
        self.preds_start = library.np.zeros(( self.input_ids_t.shape[0], MAX_LEN))
        self.preds_end = library.np.zeros(( self.input_ids_t.shape[0], MAX_LEN))
        self.VER = 'v0';
        self.DISPLAY = 1;
        self.skf = library.StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    def TEST_1text(self):
        # INPUT_IDS
        text1 = " " + " ".join(self.str1.split())
        enc = load.tokenizer.encode(text1)
        s_tok = load.sentiment_id[self.sentiment]
        self.input_ids_t[0, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        self.attention_mask_t[0, :len(enc.ids) + 5] = 1
        self.model.load_weights(self.path)
        print('Predicting Test...')
        preds = self.model.predict([self.input_ids_t, self.attention_mask_t, self.token_type_ids_t], verbose=self.DISPLAY)
        self.preds_start += preds[0] / self.skf.n_splits
        self.preds_end += preds[1] / self.skf.n_splits
    def KQ(self):
        all = []
        a = library.np.argmax(self.preds_start)
        b = library.np.argmax(self.preds_end)
        if a > b:
            print(a)
            st = self.str1
        else:
            text1 = " " + " ".join(self.str1.split())
            enc = load.tokenizer.encode(text1)
            st = load.tokenizer.decode(enc.ids[a - 1:b])
        all.append(st)
        print(all)
test="I just downloaded a ton of stunning, BEAUTIFUL wallpaper..."
path='/content/drive/MyDrive/NLP/doAN/backup/v0-roberta-4.h5'
sentiment='positive'
a=Test(load.MAX_LEN,load.PATH,test,path,sentiment)
a.TEST_1text()
a.KQ()
