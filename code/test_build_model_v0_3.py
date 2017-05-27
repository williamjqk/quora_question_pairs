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

import tensorflow as tf
import keras
# from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Bidirectional, Embedding, Dropout, Activation
from keras.layers.merge import concatenate, dot
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

BASE_DIR = '../input/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 50#255#np.random.randint(175, 275)
num_dense = 120#np.random.randint(100, 150)
rate_drop_lstm = 0.0#0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.0#0.15 + np.random.rand() * 0.25
act = 'relu'#'relu'

re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'testRNN_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

nb_words = MAX_NB_WORDS#300000#min(MAX_NB_WORDS, len(word_index))+1
# %%
#######################################
# define the model structure
#######################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        #weights=[embedding_matrix],
        trainable=False,
        name='embedding1',
        input_length=MAX_SEQUENCE_LENGTH)
# bi_lstm_layer1 = Bidirectional(LSTM(num_lstm, return_sequences=True, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm), name='bilstm1')
# bi_lstm_layer2 = Bidirectional(LSTM(num_lstm, return_sequences=True, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm), name='bilstm2')
# bi_lstm_layer3 = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm), name='bilstm3')
bi_lstm_layer1 = Bidirectional(LSTM(num_lstm, return_sequences=True, name='lstm1'), name='biRNN1')
bi_lstm_layer2 = Bidirectional(LSTM(num_lstm, return_sequences=True, name='lstm2'), name='biRNN2')
bi_lstm_layer3 = Bidirectional(LSTM(num_lstm, return_sequences=False, name='lstm3'), name='biRNN3')
with tf.name_scope('flow1_input_layer'):
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='input1')
with tf.name_scope('flow1_embedding_layer'):
    embedded_sequences_1 = embedding_layer(sequence_1_input)
with tf.name_scope('flow1_rnn_layer1'):
    x1 = bi_lstm_layer1(embedded_sequences_1)
with tf.name_scope('flow1_rnn_layer2'):
    x1 = bi_lstm_layer2(x1)
with tf.name_scope('flow1_rnn_layer3'):
    x1 = bi_lstm_layer3(x1)

with tf.name_scope('flow2_input_layer'):
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='input2')
with tf.name_scope('flow2_embedding_layer'):
    embedded_sequences_2 = embedding_layer(sequence_2_input)
with tf.name_scope('flow2_rnn_layer1'):
    y1 = bi_lstm_layer1(embedded_sequences_2)
with tf.name_scope('flow2_rnn_layer2'):
    y1 = bi_lstm_layer2(y1)
with tf.name_scope('flow2_rnn_layer3'):
    y1 = bi_lstm_layer3(y1)

with tf.name_scope('name_dense_layer'):
    merged = concatenate([x1, y1])
    # merged = Dropout(rate_drop_dense)(merged)
    # merged = BatchNormalization()(merged)
    merged = Dense(num_dense, activation=act)(merged)
    # merged = Dropout(rate_drop_dense)(merged)
    # merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)


########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
# %%
# np.random.randint(1,50,(100,30))
data_1_train = np.random.randint(1,num_lstm,(100,MAX_SEQUENCE_LENGTH))#np.random.randint()
data_2_train = np.random.randint(1,num_lstm,(100,MAX_SEQUENCE_LENGTH))
labels_train = np.random.randint(0,2,(100,1))

data_1_val = np.random.randint(1,num_lstm,(100,MAX_SEQUENCE_LENGTH))#np.random.randint()
data_2_val = np.random.randint(1,num_lstm,(100,MAX_SEQUENCE_LENGTH))
labels_val = np.random.randint(0,2,(100,1))

import datetime
now = datetime.datetime.now()
# logdir = "../logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
logdir = "/logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
dir1 = os.path.dirname(os.getcwd())
logdir = dir1 + logdir
print("tensorboard --logdir={}".format(logdir))
tb_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)
# os.system("tensorboard --logdir={}".format(logdir))
# %%
hist = model.fit([data_1_train, data_2_train], labels_train,
        validation_data=([data_1_val, data_2_val], labels_val),
        epochs=2, batch_size=20, shuffle=True, callbacks=[tb_callback]) # batch_size=2048,
