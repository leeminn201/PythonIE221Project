
# ROBERTA
Link về paper để đọc và tham khảo: <https://arxiv.org/abs/1907.11692>
## Build roBERTa Model

We use a pretrained roBERTa base model and add a custom question answer head. First tokens are input into bert_model and we use BERT's first output, i.e. x[0] below. These are embeddings of all input tokens and have shape (batch_size, MAX_LEN, 768). Next we apply tf.keras.layers.Conv1D(filters=1, kernel_size=1) and transform the embeddings into shape (batch_size, MAX_LEN, 1). We then flatten this and apply softmax, so our final output from x1 has shape (batch_size, MAX_LEN). These are one hot encodings of the start tokens indicies (for selected_text). And x2 are the end tokens indicies.

![alt](https://pic3.zhimg.com/80/v2-dcb83ca651acc1c93c9c8f982ee4b67e_1440w.jpg)
## Các bạn có thể tham khảo cách buil model bằng tensoflow

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
