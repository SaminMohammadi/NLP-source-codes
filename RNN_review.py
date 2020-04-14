from __future__ import print_function, division 
 
import os 
import sys
from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, Input, Embedding, GlobalMaxPooling1D, Dense, Bidirectional
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score

df = pd.read_csv("./data/toxic_comments/train.csv", encoding='utf8')
comments = df["comment_text"].fillna("DUMMY COMMENT").values
possible_labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
labels = df[possible_labels].values

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIMENSION = 100
tokenizer = Tokenizer(num_words= MAX_NUM_WORDS, lower=True, oov_token= 'UNKOWN')
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
word2ind = tokenizer.word_index

words_embeddings = {}
with open(f"./data/glove.6B.{EMBEDDING_DIMENSION}d.txt", encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embeddings = np.asarray(values[1:], dtype= 'float32')
        words_embeddings[word] = embeddings


num_words = min(len(words_embeddings), MAX_NUM_WORDS)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIMENSION))
for word,i in word2ind.items():
    if i< MAX_NUM_WORDS:
        embedding = words_embeddings.get(word)
        if embedding is not None:
            embedding_matrix[i] = embedding

embeding_layer = Embedding(
    num_words, 
    EMBEDDING_DIMENSION, 
    input_length= MAX_SEQUENCE_LENGTH,
    weights = [embedding_matrix],
    trainable=False)

input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embeding_layer(input_)
x = Bidirectional(LSTM(15,return_sequences=True))(x)
x = GlobalMaxPooling1D()(x)
output = Dense(len(possible_labels), activation='sigmoid')(x)

model = Model(inputs=input_, outputs=output)
model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
r = model.fit(x=data, y=labels, batch_size=120, epochs=2, validation_split=0.2)
aucs=[]
p = model.predict(data)
for i in range(6):
    auc = roc_auc_score(labels[:,i], p[:,i])
    aucs.append(auc)
print(np.mean(aucs))

