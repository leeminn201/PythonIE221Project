

from doAN.libary import libary

MAX_LEN = 96
PATH = 'D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/PythonIE221Project/tf-roberta/'
tokenizer = libary.tokenizers.ByteLevelBPETokenizer(
    vocab=PATH + 'vocab-roberta-base.json',
    merges=PATH + 'merges-roberta-base.txt',
    lowercase=True,
    add_prefix_space=True
)
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = libary.pd.read_csv(
    'D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/PythonIE221Project/Dataset/train.csv').fillna('')
test = libary.pd.read_csv(
    'D:/UIT LEARN/Năm 3 Kì 2/Python/do_an/doAN/PythonIE221Project/Dataset/test.csv').fillna('')
print(train.head())


class Process_data(libary.matrix):
    def __init__(self, ct1, ct2, MAX_LEN):
        self.ct1 = ct1
        self.ct2 = ct2
        self.MAX_LEN = MAX_LEN
        self.input_ids = libary.np.ones((ct1, MAX_LEN), dtype='int32')
        self.attention_mask = libary.np.zeros((ct1, MAX_LEN), dtype='int32')
        self.token_type_ids = libary.np.zeros((ct1, MAX_LEN), dtype='int32')
        self.start_tokens = libary.np.zeros((ct1, MAX_LEN), dtype='int32')
        self.end_tokens = libary.np.zeros((ct1, MAX_LEN), dtype='int32')
        self.input_ids_t = libary.np.ones((ct2, MAX_LEN), dtype='int32')
        self.attention_mask_t = libary.np.zeros((ct2, MAX_LEN), dtype='int32')
        self.token_type_ids_t = libary.np.zeros((ct2, MAX_LEN), dtype='int32')

    @staticmethod
    def FIND_OVERLAP(k):
        text1 = " " + " ".join(train.loc[k, 'text'].split())
        text2 = " ".join(train.loc[k, 'selected_text'].split())
        idx = text1.find(text2)
        chars = libary.np.zeros((len(text1)))
        chars[idx:idx + len(text2)] = 1
        if text1[idx - 1] == ' ':
            chars[idx - 1] = 1
        enc = tokenizer.encode(text1)
        return idx, chars, enc

    @staticmethod
    def ID_OFFSETS(enc=None):
        offsets = []
        idx = 0
        for t in enc.ids:
            w = tokenizer.decode([t])
            offsets.append((idx, idx + len(w)))
            idx += len(w)
        return offsets

    def START_END_TOKENS(self, offsets, chars=None, k=None, enc=None):
        toks = []
        for i, (a, b) in enumerate(offsets):
            sm = libary.np.sum(chars[a:b])
            if sm > 0:
                toks.append(i)

        s_tok = sentiment_id[train.loc[k, 'sentiment']]
        self.input_ids[k, :len(enc.ids) + 5] = [0] + \
            enc.ids + [2, 2] + [s_tok] + [2]
        self.attention_mask[k, :len(enc.ids) + 5] = 1
        if len(toks) > 0:
            self.start_tokens[k, toks[0] + 1] = 1
            self.end_tokens[k, toks[-1] + 1] = 1

    def INPUT_IDS(self, k):
        text1 = " " + " ".join(test.loc[k, 'text'].split())
        enc = tokenizer.encode(text1)
        s_tok = sentiment_id[test.loc[k, 'sentiment']]
        self.input_ids_t[k, :len(enc.ids) + 5] = [0] + \
            enc.ids + [2, 2] + [s_tok] + [2]
        self.attention_mask_t[k, :len(enc.ids) + 5] = 1


a = Process_data(train.shape[0], test.shape[0], MAX_LEN)

for k in range(train.shape[0]):
    idx, chars, enc = a.FIND_OVERLAP(k)
    offsets = a.ID_OFFSETS(enc)
    a.START_END_TOKENS(offsets, chars, k, enc)
for k in range(test.shape[0]):
    a.INPUT_IDS(k)
