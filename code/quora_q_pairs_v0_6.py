'''
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7

According to experiments by kagglers, Theano backend with GPU may give bad LB scores while
        the val_loss seems to be fine, so try Tensorflow backend first please
'''

########################################
## import packages
########################################
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

import keras
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Bidirectional, Embedding, Dropout, Activation
from keras.layers.merge import concatenate, dot
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from multiprocessing import Pool

# import importlib
# import sys
# importlib.reload(sys)
# sys.setdefaultencoding('utf-8')

TELE = True#True#False
if TELE == True:
    from guardian import ContextGuardian
    guard = ContextGuardian(
        addr="ws://warden.q-phantom.com/telegram",
        project="mytest2",#"TradeS",
        send_from="qphantom_log",
        send_to=274937121 #这里写上自己的telegram账户id，可以添加@userinfobot这个bot查询
    )


# %%
########################################
## set directories and parameters
########################################
BASE_DIR = '../input/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 250#np.random.randint(175, 275)#255#
num_dense = 140#np.random.randint(100, 150)#140#
rate_drop_lstm = 0.2#0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.2#0.15 + np.random.rand() * 0.25
act = 'relu'#'relu'

# num_lstm = np.random.randint(175, 275)
# num_dense = np.random.randint(100, 150)
# rate_drop_lstm = 0.15 + np.random.rand() * 0.25
# rate_drop_dense = 0.15 + np.random.rand() * 0.25
# act = 'relu'


re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'DeepBiRNN_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

print('## {} ##'.format(STAMP))
guard.message('## {} ##'.format(STAMP))

# %%
########################################
## index word vectors
########################################
print('Indexing word vectors')

import time
time_begin = time.time()

t1 = time.time()
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))
t2 = time.time()
print('######## time = {:.2f} second #########'.format(t2-t1))
type(word2vec)
word2vec['office'].shape
word2vec['1'].shape
len(word2vec.vocab)


# %%
########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # remove_stopwords=False, stem_words=False
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)


t1 = time.time()

texts_1 = []
texts_2 = []
labels = []
# f= codecs.open(TRAIN_DATA_FILE, encoding='utf-8')
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    row_l = list(list(row) for row in reader)
    with Pool(6) as p:
        texts_1 = p.map(text_to_wordlist, [v[3] for v in row_l])
    with Pool(6) as p:
        texts_2 = p.map(text_to_wordlist, [v[4] for v in row_l])
    labels = [int(v[5]) for v in row_l]
print('Found %s texts in train.csv' % len(texts_1))
t2 = time.time()
print('######## text_to_wordlist time = {:.2f} second #########'.format(t2-t1))


t1 = time.time()

test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    row_l = list(list(row) for row in reader)
    with Pool(8) as p:
        test_texts_1 = p.map(text_to_wordlist, [v[1] for v in row_l])
    with Pool(8) as p:
        test_texts_2 = p.map(text_to_wordlist, [v[2] for v in row_l])
    test_ids = [v[0] for v in row_l]
print('Found %s texts in test.csv' % len(test_texts_1))
t2 = time.time()
print('######## text_to_wordlist time = {:.2f} second #########'.format(t2-t1))

type(texts_1), len(texts_1)
# %%
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

t1 = time.time()
sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
t2 = time.time()
print('######## texts_to_sequences time = {:.2f} second #########'.format(t2-t1))

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)




# %%
########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
# word_index 纯粹是基于quora, 跟google embedding没关系
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

# %%
########################################
## sample train/validation data
########################################
#np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344

time_end = time.time()
print('######## text preprocess time = {:.2f} second #########'.format(time_end - time_begin))
guard.message('######## text preprocess time = {:.2f} second #########'.format(time_end - time_begin))
# %%
########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
# lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
# lstm_layer = LSTM(num_lstm, return_sequences=True)
# lstm_layer = LSTM(MAX_SEQUENCE_LENGTH, return_sequences=True)
# lstm_layer = LSTM(MAX_SEQUENCE_LENGTH, return_sequences=True, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
bi_lstm_layer1 = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True))
bi_lstm_layer2 = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True))
# bi_lstm_layer3 = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True))
bi_lstm_layer_last = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=False))

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = bi_lstm_layer1(embedded_sequences_1)
x1 = bi_lstm_layer2(x1)
# x1 = bi_lstm_layer3(x1)
x1 = bi_lstm_layer_last(x1)


sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = bi_lstm_layer1(embedded_sequences_2)
y1 = bi_lstm_layer2(y1)
# y1 = bi_lstm_layer3(y1)
y1 = bi_lstm_layer_last(y1)

# v_similarity = dot([x1, y1], axes = 1, normalize = True)
# preds = Dense(1, activation='sigmoid')(v_similarity)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

# %%
########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

# %%
########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
#model.summary()
print(STAMP)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.count_log = 0
    # def on_batch_end(self, batch, logs={}):
    #     self.losses.append(logs.get('loss'))
    def on_epoch_end(self, epoch, logs={}):
        self.count_log += 1
        if TELE == True:
            # guard.message("Loop %s of 100" % (self.count_log))
            guard.message("Loop {} of 100, loss {:.4f}, val_loss {:.4f}, acc {:.4f}, val_acc {:.4f}"\
                        .format(self.count_log, logs.get('loss'), logs.get('val_loss'), logs.get('acc'), logs.get('val_acc')))

history = LossHistory()

early_stopping =EarlyStopping(monitor='val_loss', patience=3)#val_acc, monitor='val_loss', patience=3
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

hist = model.fit([data_1_train, data_2_train], labels_train, \
        validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
        epochs=200, batch_size=600, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint, tensorboard, history]) # batch_size=2048,

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

# %%
########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2], batch_size=2500, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=2500, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)

if TELE == True:
    guard.close()
