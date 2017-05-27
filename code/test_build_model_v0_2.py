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
from keras.layers import Dense, Input, GRU, LSTM, Bidirectional, TimeDistributed, Embedding, Dropout, Activation,Flatten,RepeatVector,Permute, Lambda
from keras import backend as K
from keras.layers.merge import concatenate, dot, multiply
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.topology import Layer
from multiprocessing import Pool

BASE_DIR = '../input/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 50#255#np.random.randint(175, 275)
num_dense = 30#np.random.randint(100, 150)
rate_drop_lstm = 0.2#0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.2#0.15 + np.random.rand() * 0.25
act = 'relu'#'relu'

re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'testRNN_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

nb_words = MAX_NB_WORDS#300000#min(MAX_NB_WORDS, len(word_index))+1
# %%
# np.random.randint(1,50,(100,30))
data_1_train = np.random.randint(1,num_lstm,(100,MAX_SEQUENCE_LENGTH))#np.random.randint()
data_2_train = np.random.randint(1,num_lstm,(100,MAX_SEQUENCE_LENGTH))
labels_train = np.random.randint(0,2,(100,1))

data_1_val = np.random.randint(1,num_lstm,(100,MAX_SEQUENCE_LENGTH))#np.random.randint()
data_2_val = np.random.randint(1,num_lstm,(100,MAX_SEQUENCE_LENGTH))
labels_val = np.random.randint(0,2,(100,1))
# %%
#######################################
# define the model structure
#######################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        #weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

bi_lstm_layer1 = Bidirectional(GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True))
bi_lstm_layer2 = Bidirectional(GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True))

atten_layer = Sequential()
atten_layer.add(Dense(1, activation='tanh', input_shape=(MAX_SEQUENCE_LENGTH, 2*num_lstm))) # for bidirectional lstm
atten_layer.add(Flatten())
atten_layer.add(Activation('softmax'))
atten_layer.add(RepeatVector(num_lstm))
atten_layer.add(Permute([2, 1]))

time_dis = TimeDistributed(Dense(num_lstm))


sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = bi_lstm_layer1(embedded_sequences_1)
x1 = bi_lstm_layer2(x1)
atten_x1 = atten_layer(x1)
x1 = time_dis(x1)
x1 = multiply([x1, atten_x1])
x1 = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(num_lstm,))(x1)


sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = bi_lstm_layer1(embedded_sequences_2)
y1 = bi_lstm_layer2(y1)
atten_y1 = atten_layer(y1)
y1 = time_dis(y1)
y1 = multiply([y1, atten_y1])
y1 = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(num_lstm,))(y1)

v_similarity = dot([x1, y1], axes = 1, normalize = True)
preds1 = Dense(1, activation='sigmoid')(v_similarity)
#
merged = concatenate([x1, y1])
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds2 = Dense(1, activation='sigmoid')(merged)

stack1 = concatenate([preds1, preds2])
stack1 = Dense(30, activation=act)(stack1)
stack1 = Dropout(rate_drop_dense)(stack1)
stack1 = BatchNormalization()(stack1)
stack1 = Dense(30, activation=act)(stack1)
stack1 = Dropout(rate_drop_dense)(stack1)
stack1 = BatchNormalization()(stack1)
preds = Dense(1, activation='sigmoid')(stack1)


########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])

import datetime
now = datetime.datetime.now()
logdir = "../logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
tb_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)

early_stopping =EarlyStopping(monitor='val_loss', patience=1)#val_acc, monitor='val_loss', patience=3
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

# hist = model.fit([data_1_train, data_2_train], labels_train,
#         validation_data=([data_1_val, data_2_val], labels_val),
#         epochs=10, batch_size=20, shuffle=True, callbacks=[early_stopping, model_checkpoint]) # , callbacks=[tb_callback], batch_size=2048,

model.load_weights(bst_model_path)
# bst_val_score = min(hist.history['val_loss'])
