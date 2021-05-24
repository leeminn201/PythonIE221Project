
# roBERTa
Link về paper để đọc và tham khảo: <https://arxiv.org/abs/1907.11692>
- RoBERTa được xây dựng dựa trên chiến thuật mask ngôn ngữ của BERT, trong đó hệ thống sẽ học được cách dự đoán một số phần văn bản được chủ ý giấu đi trong số rất nhiều các văn bản không chú thích khác. RoBERTa được thực hiện trên PyTorch, với khả năng thay đổi một số siêu tham số chính trong BERT, trong đó bao gồm mục tiêu tiền huấn luyện câu tiếp theo của BERT, cũng như việc huấn luyện theo nhóm nhỏ và tốc độ học. Nhờ vậy, RoBERTa có thể cải thiện mục tiêu mô hình hóa các ngôn ngữ đã được mask so với BERT, qua đó cải thiện hiệu quả các tác vụ downstream (tức các tác vụ trong đó luồng dữ liệu đi từ mạng về thiết bị đầu cuối). Ngoài ra, các nhà nghiên cứu cũng đã thử huấn luyện RoBERTa trên qui mô lớn, nhiều dữ liệu hơn so với BERT, trên một khoảng thời gian dài hơn. Việc huấn luyện này sử dụng song song cả các bộ dữ liệu NLP không chú thích và CC-News – một bộ dữ liệu mới, lấy nguồn từ các bài báo công khai trên mạng.
- Những thay đổi trên về thiết kế đã cho hiệu quả cao trong các tác vụ MNLI, QNLI, RTE, STS-B, và RACE, cũng như cải thiện đáng kể trên bảng xếp hạng GLUE. Cụ thể, với điểm số là 88,5, RoBERTa hiện đang dẫn đầu bảng xếp hạng GLUE, đồng hạng với XLNet-Large. Những kết quả này đã cho thấy tầm quan trọng của một số thiết kế chưa từng được tìm hiểu trong huấn luyện BERT, đồng thời giúp chỉ rõ các ảnh hưởng riêng biệt gây ra bởi các yếu tố như kích thước dữ liệu sử dụng, thời gian huấn luyện, và mục tiêu tiền huấn luyện.
- Kết quả nghiên cứu cho thấy rằng, quá trình huấn luyện BERT có thể gây ra những hiệu quả đáng kể trong nhiều tác vụ NLP khác nhau, qua đó chứng minh rằng, phương thức này mang tính cạnh tranh cao trong số rất nhiều các phương thức khác. Nói rộng hơn, thì nghiên cứu đã chỉ ra tiềm năng của các kỹ thuật huấn luyện tự giám sát, giúp đuổi kịp, hay thậm chí là vượt qua các cách thức truyền thống, có giám sát trước đây. Ngoài ra, RoBERTa cũng là một đóng góp của Facebook trong quá trình cải thiện công nghệ trong các hệ thống tự giám sát, được đặc biệt phát triển để ít lệ thuộc vào việc đánh nhãn dữ liệu hơn – một quá trình vô cùng lâu dài và tốn tài nguyên. Các nhà nghiên cứu cũng rất hi vọng rằng, cộng đồng công nghệ sẽ tiếp tục phát triển và cải thiện, đem lĩnh vực NLP đi xa hơn nữa, với bộ mô hình và code của RoBERTa.
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
