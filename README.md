
# ROBERTA
Link về paper để đọc và tham khảo: <https://arxiv.org/abs/1907.11692>
- RoBERTa xây dựng dựa trên chiến lược "masking" ngôn ngữ của BERT, trong đó hệ thống học cách dự đoán các phần văn bản ẩn có chủ ý trong các ví dụ ngôn ngữ không có chú thích khác. RoBERTa, được triển khai trong PyTorch, sửa đổi các siêu tham số chính trong BERT, bao gồm việc xóa mục tiêu đào tạo trước câu tiếp theo của BERT và đào tạo với các đợt học nhỏ và tỷ lệ học tập lớn hơn nhiều. Điều này cho phép RoBERTa cải thiện mục tiêu mô hình ngôn ngữ được che dấu so với BERT và dẫn đến hiệu suất tác vụ hạ nguồn tốt hơn. Chúng tôi cũng khám phá việc đào tạo RoBERTa theo thứ tự dữ liệu lớn hơn BERT, trong một khoảng thời gian dài hơn. Chúng tôi đã sử dụng các bộ dữ liệu NLP không có chú thích hiện có cũng như CC-News, một bộ tiểu thuyết được rút ra từ các bài báo công khai.
- Sau khi thực hiện những thay đổi thiết kế này, mô hình của chúng tôi đã mang lại hiệu suất hiện đại trên các nhiệm vụ MNLI, QNLI, RTE, STS-B và RACE và cải thiện hiệu suất khá lớn trên tiêu chuẩn GLUE. Với số điểm 88,5, RoBERTa đã đạt vị trí hàng đầu trên bảng xếp hạng GLUE, phù hợp với thành tích của người dẫn đầu trước đó, XLNet-Large. Những kết quả này làm nổi bật tầm quan trọng của các lựa chọn thiết kế chưa được khám phá trước đây trong đào tạo BERT và giúp gỡ bỏ những đóng góp tương đối của kích thước dữ liệu, thời gian đào tạo và mục tiêu đào tạo trước.
## Build roBERTa Model

Chúng tôi sử dụng mô hình cơ sở roBERTa "pre-trained" và thêm phần đầu câu trả lời câu hỏi tùy chỉnh. Các mã thông báo đầu tiên được nhập vào bert_model và chúng tôi sử dụng đầu ra đầu tiên của BERT, tức là x [0] bên dưới. Đây là các nhúng của tất cả các mã thông báo đầu vào và có hình dạng (batch_size, MAX_LEN, 768). Tiếp theo, chúng tôi áp dụng tf.keras.layers.Conv1D (filter = 1, kernel_size = 1) và biến đổi các nhúng thành hình dạng (batch_size, MAX_LEN, 1). Sau đó, chúng tôi làm phẳng điều này và áp dụng softmax, vì vậy đầu ra cuối cùng của chúng tôi từ x1 có hình dạng (batch_size, MAX_LEN). Đây là một trong những mã hóa nóng của các chỉ báo mã thông báo bắt đầu (cho văn bản được chọn). Và x2 là các chỉ số mã thông báo kết thúc.

![alt](https://pic3.zhimg.com/80/v2-dcb83ca651acc1c93c9c8f982ee4b67e_1440w.jpg)
## Các bạn có thể tham khảo cách build model bằng tensoflow

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
