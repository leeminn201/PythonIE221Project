'''
    Module kiểm tra trực tiếp project.
'''
from Load_Data import Processing_data as load
from Build_Model import Build as build
import pandas as pd
import random as r
import os
class Test(build.Model_RoBERTa):
    '''
        Class test project.
    '''
    def __init__(self,MAX_LEN,PATH,path):
        super().__init__(MAX_LEN,PATH)
        self.path=path
        self.model=super().Build_model()
        self.VER = 'v0'
        self.DISPLAY = 1
        self.skf = load.library.Stra_Kfold(5, True, 777)
        self.arr = {
            0: 'neutral',
            1: 'positive',
            2: 'negative'
        }
#Test model
    def Test_Model(self,test):
        '''
            Đưa câu test vào model.
        :param test: câu test
        :return:
        '''
        # INPUT_IDS
        ct = test.shape[0]
        input_ids_t = load.library.np.ones((ct, self.MAX_LEN), dtype='int32')
        attention_mask_t = load.library.np.zeros((ct, self.MAX_LEN), dtype='int32')
        token_type_ids_t = load.library.np.zeros((ct, self.MAX_LEN), dtype='int32')

        for k in range(test.shape[0]):
            # INPUT_IDS
            text1 = " " + " ".join(test.loc[k, 'text'].split())
            enc = load.tokenizer.encode(text1)
            s_tok = load.sentiment_id[test.loc[k, 'sentiment']]
            input_ids_t[k, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
            attention_mask_t[k, :len(enc.ids) + 5] = 1
        self.model.load_weights(self.path)
        preds_start = load.library.np.zeros((input_ids_t.shape[0], self.MAX_LEN))
        preds_end = load.library.np.zeros((input_ids_t.shape[0], self.MAX_LEN))
        print('Predicting Test...')
        preds = self.model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=self.DISPLAY)
        preds_start += preds[0]/self.skf.n_splits
        preds_end += preds[1]/self.skf.n_splits
        all = []
        for k in range(input_ids_t.shape[0]):
            a = load.library.np.argmax(preds_start[k,])
            b = load.library.np.argmax(preds_end[k,])
            if a > b:
                st = test.loc[k, 'text']
            else:
                text1 = " " + " ".join(test.loc[k, 'text'].split())
                enc = load.tokenizer.encode(text1)
                st = load.tokenizer.decode(enc.ids[a - 1:b])
            all.append(st)
        return all
#Load dữ liệu người dùng nhập
    def Text(self):
        '''
            Xử lí câu input.
        '''
        df = pd.DataFrame(columns=["textID", "text", "sentiment"])
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
            df = df.append(data, ignore_index=True)
            k = int(input("Nếu bạn ko cần text nữa thì chọn 0 = "))
            if k == 0:
                break
        return df
        # df[["textID", "text", "sentiment", "selected_text"]].to_csv('D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/Dataset/submission.csv', index=False)
#Đưa file CSV để load
    @staticmethod
    def Text_CSV():
        '''
            Input file test dạng csv.
        '''
        k = input("Nhập đường dẫn link = ")
        test = pd.read_csv(k).fillna('')
        return test,k
#Test 1 câu không cần lưu chỉ test để xem kết quả
    def Text_Speed_Sentences(self,str1,sentiment):
        '''
            Test 1 câu nhưng không lưu kết quả.
        :param str1: Câu test.
        :param sentiment: Sentiment của câu test.
        '''
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
        #Nếu load 5 fold với 5 model đã train thì kết quả tốt hơn
        # for fold in range(0,model_ct):
        #     self.model.load_weights('/content/drive/MyDrive/NLP/backup/%s-roberta-%i.h5'%(self.VER,fold))
        #     print('Predicting Test...')
        #     preds = self.model.predict([input_ids_t, attention_mask_t, token_type_ids_t],
        #                            verbose=self.DISPLAY)
        #     preds_start += preds[0] / self.skf.n_splits
        #     preds_end += preds[1] / self.skf.n_splits
        self.model.load_weights(self.path)
        preds = self.model.predict([input_ids_t, attention_mask_t, token_type_ids_t],verbose=self.DISPLAY)
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
#Kết quả lưu vào file csv minh tạo để lưu khi người dùng nhập
    @staticmethod
    def Result(test,all):
        '''
            Kết quả lưu vào file csv minh tạo để lưu khi người dùng nhập
        '''
        test['selected_text'] = all
        if not os.path.isfile('D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/Dataset/submission.csv'):
            test.to_csv('D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/Dataset/submission.csv', index=False)
        else:
            with open('D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/Dataset/submission.csv', 'a', encoding="utf-8",newline='') as f:
                test.to_csv(f, index=False, header=f.tell()==0)
        print(test[['text','selected_text']])
#Kết quả lưu vào file CSV khi đưa vào để test,sample_submission.csv lưu toàn bộ dữ liệu được test từ file csv (nhiều file csv)
    @staticmethod
    def Result_CSV(test,all,link_file=None):
        '''
        Kết quả lưu vào file CSV khi đưa vào để test.
        Sample_submission.csv lưu toàn bộ dữ liệu được test từ một/nhiều file csv
        '''
        test['selected_text'] = all
        if not os.path.isfile('D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/Dataset/sample_submission.csv'):
            test.to_csv('D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/Dataset/sample_submission.csv', index=False)
        else:
            with open('D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/Dataset/sample_submission.csv', 'a', encoding="utf-8",newline='') as f:
                test.to_csv(f, index=False, header=f.tell()==0)
        test.to_csv(link_file, index=False)
        print(test['selected_text'])
path='D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/backup/v0-roberta-4.h5'
a = Test(load.MAX_LEN, load.PATH, path)
#Test với 1 câu nhanh ko cần lưu vô csv:
name = input("Nhập text cảm xúc : ")
arr = {
    0: 'neutral',
    1: 'positive',
    2: 'negative'
}
n = int(input("[0:'neutral',1:'positive',2:'negative']=  "))
a.Text_Speed_Sentences(name, arr[n])


#Test với 1 hoặc nhiều câu lưu vô file D:\UIT LEARN\Năm 3 Kì 2\Python\do_an\doAN\Dataset\submission.csv đây là file chính lưu dữ liệu người dùng đưa vào
df = a.Text()  # người dùng nhập dư liệu từ bàn phím (cần xữ lí try catch khi người dùng nhập sai hoặc ràng buộc)
all = a.Test_Model(df)  # đưa dữ liệu vào và bắt đầu test xuất ra kq
a.Result(df, all)

#Test với 1 file csv bất kì nhung phải có header là "text", "sentiment" sai định dạng cút
df, link = a.Text_CSV()
all = a.Test_Model(df)
a.Result_CSV(df, all, link)
