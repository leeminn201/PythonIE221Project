
# _Tìm hiểu Information Extraction và RoBERTa qua TWEET SENTIMENT EXTRACTION sử dụng thư viện TENSORFLOW._
Link Github: <https://github.com/leminhuit201/PythonIE221Project>
Tài liệu: <https://www.kaggle.com/c/tweet-sentiment-extraction/overview>

## Giới thiệu về thư viện tensorflow
### Tại sao lại là tensorflow mà không phải các thư viện khác?
* Tensorflow là thư viện mã nguồn mở cho nhiệm vụ đào tạo và phát triển các mô hình máy học: Mọi người có thể phát triển và đóng góp vào thư viện hiện có.
* Được phát triển bởi nhóm GoogleBrain: Một thư viện được phát triển và hậu thuẫn bởi một tổ chức lớn như Google đảm bảo thư viện sẽ được duy trì và cải tiến theo thời gian.
* Tensorflow có cộng đồng người dùng lớn: Ngay từ khi ra mắt vào năm 2015, tensorflow lập tức chiếm vị trí dẫn đầu về số lượng người dùng. Nơi mà skikit-learn đã thống trị trong suốt nhiều năm trước đó. Trải qua nhiều năm phát triển, bảng xếp hạng các thư viện cho việc thực thi máy học có nhiều biến động. Tuy nhiên, tensorflow vẫn nghiễm nhiên dẫn đầu và bỏ xa các thư viện còn lại. 
Thống kê bảng xếp hạng các thư viện phổ biến nhất cho nhiệm vụ máy học từ năm 2013 đến năm 2019: <https://www.youtube.com/watch?v=744f60NyAgc>

* Tensorflow miễn phí: Người dùng chỉ cần cài đặt thư viện vào máy là có thể sử dụng mà không cần trả phí . Nếu sử dụng google colab, thư viện này thậm chí còn được tích hợp sẵn.
* Cú pháp dễ đọc, dễ học và nguồn tài liệu phong phú: Các hàm xử lý của tensorflow được đặt tên theo 1 quy chuẩn nhất định bằng tiếng Anh, giúp người mới tiếp cận dễ đọc và nhớ tên các hàm, cũng như có thể áp dụng như các chương trình mã python thông thường. Bên cạnh đó, nếu có thắc mắc về cách sử dụng của một hàm hay một module nào đó, người dùng hoàn toàn có thể kiểm tra tài liệu hướng dẫn được viết chi tiết cho từng hàm. Có thể tìm thấy nguồn hướng dẫn khổng lồ của tensorflow tại: <https://www.tensorflow.org/api_docs/python/tf>
* Tensorflow có thể dễ dàng huấn luyện chương trình trên CPU và GPU cho nhiệm vụ tính toán: Hiện nay, với các mô hình học sâu, việc tính toán song song với GPU giúp tiết kiệm rất nhiều thời gian, và tensorflow hỗ trợ chúng tôi làm việc đó. Kết hợp với GPU được cung cấp miễn phí từ Google Colab chúng tôi có thể tiết kiệm rất nhiều thời gian cho việc huấn luyện mô hình( mất khoảng hơn 1 tiếng cho bài toán trích xuất thông tin này thay vì hơn 12 tiếng nếu sử dụng CPU của máy tính)

** Với những lí do chính trên, cộng thêm nhiều lợi ích khác mà tensorflow cung cấp, nhóm chúng tôi quyết định sử dụng tensorflow làm thư viện hỗ trợ chính cho nhiệm vụ giải quyết bài toán này, mọi người hoàn toàn có thể tìm hiểu và sử dụng các thư viện hỗ trợ khác như PyTorch, Skikit-learn,... **


### Information Extraction (IE) là một lĩnh vực về xử lý trích xuất thông tin có cấu trúc trong xử lý ngôn ngữ tự nhiên.
## Khuyến khích sử dụng Google Colab vì có hỗ trợ chạy bằng GPU. 
## Cùng bắt đầu nào
### Bước 1: Install một số module vì Colab không hỗ trợ sẵn.

    !pip install transformers
    !pip install sentencepiece
    
###   Bước 2: Import thư viện, module sẽ sử dụng trong đồ án.
    
    import pandas as pd, numpy as np
    import math
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from sklearn.model_selection import StratifiedKFold
    from transformers import *
    import tokenizers
    from tensorflow.keras import layers

### Bước 3: Cấu hình tham số cho thuật toán mã hóa BBPE.
Nếu bạn chưa hiểu vè BBPE: <https://huggingface.co/transformers/tokenizer_summary.html>

    MAX_LEN = 96
    PATH = '/content/drive/MyDrive/roberta-base/'
    tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab = PATH+'vocab.json', 
    merges = PATH+'merges.txt', 
    lowercase = True,
    add_prefix_space = True)
### Bước 4: Tạo một số biến sẽ dùng và import data train/test.
###### Trong đó: EPOCHS là số lần tất cả data (đã chia nhỏ) được train, BATCH_SIZE là size của mỗi phần khi chia nhỏ. 
###
   
    EPOCHS = 3 
    BATCH_SIZE = 32 
    PAD_ID = 1
    SEED = 88888
    LABEL_SMOOTHING = 0.1
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
    train = pd.read_csv('/content/drive/MyDrive/Dataset/Python /train.csv').fillna('')
    train.head()
### Bước 5:  Thiết lập các ma trận đầu vào cho RoBERTa.
    ct = train.shape[0]
    input_ids = np.ones((ct,MAX_LEN),dtype='int32')
    #Tạo attention_mask nhằm fit các sequence quá nhỏ
    attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
    token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
    start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
    end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
    for k in range(train.shape[0]):
    
    # FIND OVERLAP
    text1 = " "+" ".join(train.loc[k,'text'].split())
    text2 = " ".join(train.loc[k,'selected_text'].split())
    #Tìm từ selected_text trong text
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx+len(text2)]=1
    if text1[idx-1]==' ': chars[idx-1] = 1 
    #Tokenizer
    enc = tokenizer.encode(text1) 
    
    # ID_OFFSETS
    offsets = []; idx=0
    for t in enc.ids:
        w = tokenizer.decode([t])
        #Decode rồi đưa vào offset
        offsets.append((idx,idx+len(w)))
        idx += len(w)
    # START END TOKENS
    toks = []
    for i,(a,b) in enumerate(offsets):
        sm = np.sum(chars[a:b])
        if sm>0: toks.append(i) 
### Xử lí đầu vào (tập train).        
    s_tok = sentiment_id[train.loc[k,'sentiment']]
    input_ids[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]
    attention_mask[k,:len(enc.ids)+3] = 1
    if len(toks)>0:
        start_tokens[k,toks[0]+2] = 1
        end_tokens[k,toks[-1]+2] = 1
### Xử lí đầu vào (tập test).
    test = pd.read_csv('/content/drive/MyDrive/Dataset/Python /test.csv').fillna('')
    ct = test.shape[0]
    input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
    attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
    token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')
    
    for k in range(test.shape[0]):
        
    # INPUT_IDS
    text1 = " "+" ".join(test.loc[k,'text'].split())
    enc = tokenizer.encode(text1)                
    s_tok = sentiment_id[test.loc[k,'sentiment']]
    input_ids_t[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]
    attention_mask_t[k,:len(enc.ids)+3] = 1
### Bước 6: Xây dựng Model.
    def build_model():
    #Đầu vào model.
        ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
### Data train model có sẵn và sử dụng lại những lần đã train.
    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)
#####  Dropout: bỏ qua ngẫu nhiên các unit trong quá trình train nhằm giảm sự phụ thuộc vào nhau giữa các neural và giúp tránh tình trạng over-fitting.
###
##### Conv1D: nói đơn giản Conv1D có chức năng tổng hợp và thu nhỏ ma trận input ngoài ra Conv1D thường được xử dụng khi xử lí văn bản còn Conv2D xử lí ảnh.   
###
##### Flatten giúp reshape lại ma trận thành mảng đơn. Giúp đơn giản output.
###
##### Activation là hàm kích hoạt. Hàm kích hoạt đóng vai trò là thành phần phi tuyến tại output của các nơ-ron. Hiểu đơn giản là nếu không có nó thì output cũng chỉ đơn giản là nhận input với các weight mà thôi.
####
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x1 = tf.keras.layers.Conv1D(1,1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(1,1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

### Xây dựng hàm độ đo Jaccard.
    
    def jaccard(str1, str2): 
        a = set(str1.lower().split()) 
        b = set(str2.lower().split())
        if (len(a)==0) & (len(b)==0): return 0.5
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    
### Bước 7: Train Model sử dụng tập train và cả những lần train trước.
    
    jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
    oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
    oof_end = np.zeros((input_ids.shape[0],MAX_LEN))
    preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))
    
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)
    for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):

    print('#'*25)
    print('### FOLD %i'%(fold+1))
    print('#'*25)
    
    K.clear_session()
    model = build_model()
        
    sv = tf.keras.callbacks.ModelCheckpoint(
        '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')
        
    model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 
        epochs=3, batch_size=32, verbose=DISPLAY, callbacks=[sv],
        validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], 
        [start_tokens[idxV,], end_tokens[idxV,]]))
    
    print('Loading model...')
    model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    
    print('Predicting OOF...')
    oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)
    
    print('Predicting Test...')
    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
    preds_start += preds[0]/skf.n_splits
    preds_end += preds[1]/skf.n_splits
### Trả về độ chính xác chi tiết.    
    # DISPLAY FOLD JACCARD
    all = []
    for k in idxV:
        a = np.argmax(oof_start[k,])
        b = np.argmax(oof_end[k,])
        if a>b: 
            st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
        else:
            text1 = " "+" ".join(train.loc[k,'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a-1:b])
        all.append(jaccard(st,train.loc[k,'selected_text']))
    jac.append(np.mean(all))
    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
    print()
###
### Các bạn có thể đọc thêm để hiểu rõ hơn bài blog này.
https://www.phamduytung.com/blog/2019-05-05-deep-learning-dropout/
https://huggingface.co/transformers/model_doc/auto.html
https://qastack.vn/stats/295397/what-is-the-difference-between-conv1d-and-conv2d
https://aicurious.io/posts/2019-09-23-cac-ham-kich-hoat-activation-function-trong-neural-networks/
https://machinelearningcoban.com/2018/07/06/deeplearning/
https://qastack.vn/programming/43237124/what-is-the-role-of-flatten-in-keras
### Bài Blog đầu tiên của mình -  giải thích chi tiết một chương trình code thuộc chủ đề bài toán Information đã kết thúc.
### Mong rằng các bạn sau khi đọc sẽ hiểu thêm phần nào về chủ đề IE nói chung và bài toán này nói riêng.
### Cảm ơn các bạn đã đọc.
###### Mail liên hệ: 18521106@gm.uit.edu.vn
# roBERTa

Link về paper để đọc và tham khảo: <https://arxiv.org/abs/1907.11692>

-   RoBERTa được xây dựng dựa trên chiến thuật mask ngôn ngữ của BERT, trong đó hệ thống sẽ học được cách dự đoán một số phần văn bản được chủ ý giấu đi trong số rất nhiều các văn bản không chú thích khác. RoBERTa được thực hiện trên PyTorch, với khả năng thay đổi một số siêu tham số chính trong BERT, trong đó bao gồm mục tiêu tiền huấn luyện câu tiếp theo của BERT, cũng như việc huấn luyện theo nhóm nhỏ và tốc độ học. Nhờ vậy, RoBERTa có thể cải thiện mục tiêu mô hình hóa các ngôn ngữ đã được mask so với BERT, qua đó cải thiện hiệu quả các tác vụ downstream (tức các tác vụ trong đó luồng dữ liệu đi từ mạng về thiết bị đầu cuối). Ngoài ra, các nhà nghiên cứu cũng đã thử huấn luyện RoBERTa trên qui mô lớn, nhiều dữ liệu hơn so với BERT, trên một khoảng thời gian dài hơn. Việc huấn luyện này sử dụng song song cả các bộ dữ liệu NLP không chú thích và CC-News – một bộ dữ liệu mới, lấy nguồn từ các bài báo công khai trên mạng.
-   Những thay đổi trên về thiết kế đã cho hiệu quả cao trong các tác vụ MNLI, QNLI, RTE, STS-B, và RACE, cũng như cải thiện đáng kể trên bảng xếp hạng GLUE. Cụ thể, với điểm số là 88,5, RoBERTa hiện đang dẫn đầu bảng xếp hạng GLUE, đồng hạng với XLNet-Large. Những kết quả này đã cho thấy tầm quan trọng của một số thiết kế chưa từng được tìm hiểu trong huấn luyện BERT, đồng thời giúp chỉ rõ các ảnh hưởng riêng biệt gây ra bởi các yếu tố như kích thước dữ liệu sử dụng, thời gian huấn luyện, và mục tiêu tiền huấn luyện.
-   Kết quả nghiên cứu cho thấy rằng, quá trình huấn luyện BERT có thể gây ra những hiệu quả đáng kể trong nhiều tác vụ NLP khác nhau, qua đó chứng minh rằng, phương thức này mang tính cạnh tranh cao trong số rất nhiều các phương thức khác. Nói rộng hơn, thì nghiên cứu đã chỉ ra tiềm năng của các kỹ thuật huấn luyện tự giám sát, giúp đuổi kịp, hay thậm chí là vượt qua các cách thức truyền thống, có giám sát trước đây. Ngoài ra, RoBERTa cũng là một đóng góp của Facebook trong quá trình cải thiện công nghệ trong các hệ thống tự giám sát, được đặc biệt phát triển để ít lệ thuộc vào việc đánh nhãn dữ liệu hơn – một quá trình vô cùng lâu dài và tốn tài nguyên. Các nhà nghiên cứu cũng rất hi vọng rằng, cộng đồng công nghệ sẽ tiếp tục phát triển và cải thiện, đem lĩnh vực NLP đi xa hơn nữa, với bộ mô hình và code của RoBERTa.

## Build roBERTa Model

Chúng tôi sử dụng mô hình cơ sở roBERTa "pre-trained" và thêm phần đầu câu trả lời câu hỏi tùy chỉnh. Các mã thông báo đầu tiên được nhập vào bert_model và chúng tôi sử dụng đầu ra đầu tiên của BERT, tức là x [0] bên dưới. Đây là các nhúng của tất cả các mã thông báo đầu vào và có hình dạng (batch_size, MAX_LEN, 768). Tiếp theo, chúng tôi áp dụng tf.keras.layers.Conv1D (filter = 1, kernel_size = 1) và biến đổi các nhúng thành hình dạng (batch_size, MAX_LEN, 1). Sau đó, chúng tôi làm phẳng điều này và áp dụng softmax, vì vậy đầu ra cuối cùng của chúng tôi từ x1 có hình dạng (batch_size, MAX_LEN). Đây là một trong những mã hóa nóng của các chỉ báo mã thông báo bắt đầu (cho văn bản được chọn). Và x2 là các chỉ số mã thông báo kết thúc.

![alt](https://pic3.zhimg.com/80/v2-dcb83ca651acc1c93c9c8f982ee4b67e_1440w.jpg)

## Các bạn có thể tham khảo cách build model bằng tensorflow

    def build_model():
      ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
      att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
      tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

      config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
      bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
      x = bert_model(ids,attention_mask=att,token_type_ids=tok)

      x1 = tf.keras.layers.Dropout(0.1)(x[0])
      x1 = tf.keras.layers.Conv1D(1,1)(x1)
      x1 = tf.keras.layers.Flatten()(x1)
      x1 = tf.keras.layers.Activation('softmax')(x1)

      x2 = tf.keras.layers.Dropout(0.1)(x[0])
      x2 = tf.keras.layers.Conv1D(1,1)(x2)
      x2 = tf.keras.layers.Flatten()(x2)
      x2 = tf.keras.layers.Activation('softmax')(x2)

      model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
      optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
      model.compile(loss='categorical_crossentropy', optimizer=optimizer)

      return model

## Model trên pytorch <https://pytorch.org/hub/pytorch_fairseq_roberta/>
