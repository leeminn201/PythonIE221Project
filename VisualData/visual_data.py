from Library import library as l

from LoadData import load_data as load

from tranning.train import Train

from build_model import build


class Visual(load):
    def __init__(self, ct1, ct2, MAX_LEN):
        super.__init__(self, ct1, ct2, MAX_LEN)
        self.Test = load.test
        self.Train = load.train
        self.all = []
        self.start = []
        self.end = []
        self.start_pred = []
        self.end_pred = []

    def visualTrain(self):
        for k in range(super().input_ids.shape[0]):
            a = l.np.argmax(self.preds_start_train[k,])
            b = l.np.argmax(self.preds_end_train[k,])
            self.start.append(l.np.argmax(self.start_tokens[k]))
            self.end.append(l.np.argmax(self.end_tokens[k]))        
            if a>b:
                st = self.Train.loc[k,'text']
                self.start_pred.append(0)
                self.end_pred.append(len(st))
            else:
                text1 = " "+" ".join(self.Train.loc[k,'text'].split())
                enc = load.tokenizer.encode(text1)
                st = load.tokenizer.decode(enc.ids[a-2:b-1])
                self.start_pred.append(a)
                self.end_pred.append(b)
            all.append(st)
        self.Train['start'] = self.start
        self.Train['end'] = self.end
        self.Train['start_pred'] = self.start_pred
        self.Train['end_pred'] = self.end_pred
        self.Train['selected_text_pred'] = all
        self.Train.sample(10)
        return self.Train

    def summary(self):
        return train.describe()

    def visualTest(self):
        all = []

        for k in range(self.input_ids_t.shape[0]):
            a = l.np.argmax(self.preds_start[k,])
            b = l.np.argmax(self.preds_end[k,])
            if a>b: 
                st = self.Test.loc[k,'text']
            else:
                text1 = " "+" ".join(self.Test.loc[k,'text'].split())
                enc = load.tokenizer.encode(text1)
                st = load.tokenizer.decode(enc.ids[a-2:b-1])
            all.append(st)

        self.Test['selected_text'] = all
        self.Test[['textID','selected_text']].to_csv('submission.csv',index=False)
        self.Test.sample(10)
        return self.Test


    def metric_tse(self, df,col1,col2):
        return df.apply(lambda x: build.jaccard(x[col1],x[col2]),axis=1)
    
    def add_train(self):
        self.Train = self.Train.replace({'sentiment': {'negative': -1, 'neutral': 0, 'positive': 1}})
        self.Train['len_text'] = self.Train['text'].str.len()
        self.Train['len_selected_text'] = self.Train['selected_text'].str.len()
        self.Train['diff_num'] = self.Train['end']-self.Train['start']
        self.Train['share'] = self.Train['len_selected_text']/self.Train['len_text']

        self.Train['selected_text_pred'] = self.Train['selected_text_pred'].map(lambda x: x.lstrip(' '))
        self.Train['len_selected_text_pred'] = self.Train['selected_text_pred'].str.len()
        self.Train['diff_num_pred'] = self.Train['end_pred']-self.Train['start_pred']
        self.Train['share_pred'] = self.Train['len_selected_text_pred']/self.Train['len_text']
        # len_equal
        self.Train['len_equal'] = 0
        self.Train.loc[(self.Train['start'] == self.Train['start_pred']) & (self.Train['end'] == self.Train['end_pred']), 'len_equal'] = 1
        # Độ đo
        self.Train['metric'] = Train.metric_tse(self.Train,'selected_text','selected_text_pred')
        # res
        self.Train['res'] = 0
        self.Train.loc[self.Train['metric'] == 1, 'res'] = 1
        return self.Train

    def rep_3chr(self, text):
        # Kiểm tra xem có từ kí tự lặp lại ko
        chr3 = 0
        for word in text.split():
            for c in set(word):
                if word.rfind(c+c+c) > -1:
                    chr3 = 1                
        return chr3
        
    def all_of_Train(self):
        self.Train['text_chr3'] = self.Train['text'].apply(self.rep_3chr)
        self.Train['selected_text_chr3'] = self.Train['selected_text'].apply(self.rep_3chr)
        self.Train['selected_text_pred_chr3'] = self.Train['selected_text_pred'].apply(self.rep_3chr)
        
        col_interesting = ['sentiment', 'len_text', 'text_chr3', 'selected_text', 'len_selected_text', 'diff_num', 'share', 
                            'selected_text_chr3', 'selected_text_pred', 'len_selected_text_pred', 'diff_num_pred', 'share_pred',
                            'selected_text_pred_chr3', 'len_equal', 'metric', 'res']
        self.Train[col_interesting].head(10)
        return self.Train

    def plot_word_cloud(x, col):
        corpus=[]
        for k in x[col].str.split():
            for i in k:
                corpus.append(i)
        l.plt.figure(figsize=(12,8))
        word_cloud = l.WordCloud(
                                background_color='black',
                                max_font_size = 80
                                ).generate(" ".join(corpus[:50]))
        l.plt.imshow(word_cloud)
        l.plt.axis('off')
        l.plt.show()
        return corpus[:50]

    def wordCloud(self): 
        print('Word cloud các từ xuất hiện trong tập train')
        train_all = self.plot_word_cloud(self.Train, 'text')
        train_all

        print('Word cloud các từ xuất hiện trong tập test')
        test_all = self.plot_word_cloud(self.Test, 'text')
        test_all

        print('Word cloud cho các từ được chọn trong tập train')
        train_selected_text = self.plot_word_cloud(self.Train, 'selected_text')
        return

    #Metric
    def histogram(self):
        print('Độ thị tỉ lệ dự đoán')
        self.Train[['metric']].hist(bins=10)

        train_outlier = self.Train[self.Train['res'] == 0].reset_index(drop=True)
        train_outlier
        print('Dự đoán các t/h ngoại lệ của tập train')
        train_outlier[['metric']].hist(bins=10)
        return 
