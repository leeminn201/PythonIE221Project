
# ROBERTA

## Build roBERTa Model

We use a pretrained roBERTa base model and add a custom question answer head. First tokens are input into bert_model and we use BERT's first output, i.e. x[0] below. These are embeddings of all input tokens and have shape (batch_size, MAX_LEN, 768). Next we apply tf.keras.layers.Conv1D(filters=1, kernel_size=1) and transform the embeddings into shape (batch_size, MAX_LEN, 1). We then flatten this and apply softmax, so our final output from x1 has shape (batch_size, MAX_LEN). These are one hot encodings of the start tokens indicies (for selected_text). And x2 are the end tokens indicies.

![alt](https://pic3.zhimg.com/80/v2-dcb83ca651acc1c93c9c8f982ee4b67e_1440w.jpg)
