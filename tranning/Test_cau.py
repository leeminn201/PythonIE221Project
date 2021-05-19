from load_data import load
from build_model import build
import pandas as pd
import random as r
class Test(build.Model1):
    def __init__(self,MAX_LEN,PATH,path):
        super().__init__(MAX_LEN,PATH)
        self.MAX_LEN=MAX_LEN
        self.path=path
        self.model=super().buil()
        self.VER = 'v0';
        self.DISPLAY = 1;
        self.skf = load.library.StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
        self.df = pd.DataFrame(columns=["textID", "text", "sentiment", "selected_text"])
        self.arr = {
            0: 'neutral',
            1: 'positive',
            2: 'negative'
        }
    def TEST_1text(self,test):
        # INPUT_IDS
        ct=test.shape[0]
        input_ids_t = load.library.np.ones((ct, self.MAX_LEN), dtype='int32')
        attention_mask_t = load.library.np.zeros((ct, self.MAX_LEN), dtype='int32')
        token_type_ids_t = load.library.np.zeros((ct, self.MAX_LEN), dtype='int32')
        preds_start = load.library.np.zeros(( input_ids_t.shape[0], self.MAX_LEN))
        preds_end = load.library.np.zeros(( input_ids_t.shape[0], self.MAX_LEN))
        for k in range(test.shape[0]):
            text1 = " " + " ".join(test.loc[k,'text'].split())
            enc = load.tokenizer.encode(text1)
            s_tok = load.sentiment_id[test.loc[k,'sentiment']]
            input_ids_t[0, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
            attention_mask_t[0, :len(enc.ids) + 5] = 1
        self.model.load_weights(self.path)
        print('Predicting Test...')
        preds = self.model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=self.DISPLAY)
        preds_start += preds[0] / self.skf.n_splits
        preds_end += preds[1] / self.skf.n_splits
        all = []
        for k in range(input_ids_t.shape[0]):
            a = load.library.np.argmax(preds_start)
            b = load.library.np.argmax(preds_end)
            if a > b:
                print(a)
                st = test.loc[k,'text']
            else:
                text1 = " " + " ".join(test.loc[k,'text'].split())
                enc = load.tokenizer.encode(text1)
                st = load.tokenizer.decode(enc.ids[a - 1:b])
            all.append(st)
        return all
    def TEXT(self,p):
        while True:
            name = input("Nhập text cảm xúc : ")
            n = int(input("[0:'neutral',1:'positive',2:'negative']=  "))
            while n != 0 and n != 1 and n != 2:
                n = int(input("[0:'neutral',1:'positive',2:'negative']=  "))
            i = r.randint(1000, 9999)
            data = {
                "textID": i,
                "text": name,
                "sentiment": self.arr[n]
            }
            self.df = self.df.append(data, ignore_index=True)
            k = int(input("Nếu bạn ko cần text nữa thì chọn 0 = "))
            if k == 0:
                break
        # df[["textID", "text", "sentiment", "selected_text"]].to_csv('D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/Dataset/submission.csv', index=False)
    @staticmethod
    def TEXT_CSV():
        k = input("Nhập đường dẫn link = ")
        test = pd.read_csv(k).fillna('')
        return test
    def Text_speed_1cau(self,str1,sentiment):
        input_ids_t = load.library.np.ones((1, self.MAX_LEN), dtype='int32')
        attention_mask_t = load.library.np.zeros((1, self.MAX_LEN), dtype='int32')
        token_type_ids_t = load.library.np.zeros((1, self.MAX_LEN), dtype='int32')
        preds_start = load.library.np.zeros((input_ids_t.shape[0], self.MAX_LEN))
        preds_end = load.library.np.zeros((input_ids_t.shape[0], self.MAX_LEN))
        text1 = " " + " ".join(str1.split())
        enc = load.tokenizer.encode(text1)
        s_tok = load.sentiment_id[sentiment]
        input_ids_t[0, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        attention_mask_t[0, :len(enc.ids) + 5] = 1
        self.model.load_weights(self.path)
        print('Predicting Test...')
        preds = self.model.predict([input_ids_t, attention_mask_t, token_type_ids_t],
                                   verbose=self.DISPLAY)
        preds_start += preds[0] / self.skf.n_splits
        preds_end += preds[1] / self.skf.n_splits
        a = load.library.np.argmax(preds_start)
        b = load.library.np.argmax(preds_end)
        if a > b:
            st = str1
        else:
            text1 = " " + " ".join(str1.split())
            enc = load.tokenizer.encode(text1)
            st = load.tokenizer.decode(enc.ids[a - 1:b])
        print("text: {}   sentiment: {}   selected_text:  {}".format(str1,sentiment,st))
    @staticmethod
    def KQ(test,all):
        test['selected_text'] = all
        # test[['selected_text']].to_csv('D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/Dataset/submission.csv', index=False)
        # pd.set_option('max_colwidth', 60)
        # test.sample(1)
        print(test['selected_text'])
path='D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/backup/v0-roberta-4.h5'
a=Test(load.MAX_LEN,load.PATH,path)
# df=a.TEXT()
# all=a.TEST_1text(df)
# a.KQ(df,all)
# name = input("Nhập text cảm xúc : ")
# arr = {
#             0: 'neutral',
#             1: 'positive',
#             2: 'negative'
#         }
# n = int(input("[0:'neutral',1:'positive',2:'negative']=  "))
# a.Text_speed_1cau(name,arr[n])
df=a.TEXT_CSV()
all=a.TEST_1text(df)
a.KQ(df,all)

# for fold in range(0,5):
#   model.load_weights('/content/drive/MyDrive/NLP/backup/%s-roberta-%i.h5'%(VER,fold))
#   print('Predicting Test...')
#   preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
#   preds_start += preds[0]/skf.n_splits
#   preds_end += preds[1]/skf.n_splits