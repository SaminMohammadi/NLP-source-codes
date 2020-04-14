from __future__ import print_function

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Embedding, Input, LSTM, GlobalMaxPool1D, Dense, Bidirectional
from sklearn.metrics import roc_auc_score
import numpy as np

DIMENSION_LENGTH = 100
SEQUENCE_LENGTH = 50
MAX_VOCAB_NUMBER = 20000

print ('Reading keywords and labels ...')
df = pd.read_csv("./data/manual_corpus_with_id.csv")
keywords = df["keyword"].fillna("DUMMY KEYWORDS").values 
possible_labels = ['INF', 'NAV', 'RES'] # three binary columns 
labels = df[possible_labels].values
print ('Min:', min(len(str(s)) for s in keywords))
print ('Max:', max(len(str(s)) for s in keywords))

print('Tokenizing & padding ...')
tokenizer = Tokenizer(MAX_VOCAB_NUMBER, lower=True, oov_token='UNKOWN')
tokenizer.fit_on_texts(keywords)
sequences = tokenizer.texts_to_sequences(keywords)

data = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)
word2index = tokenizer.word_index 
print ('Nomebr of Found tokens:%s' %len(word2index))


words_embeddings = {}
print ('Loading embeddings ...')
with open(f'./data/glove.6B.{DIMENSION_LENGTH}d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embedding = values[1:]
        words_embeddings[word] = embedding 

num_words = min(len(word2index)+1, MAX_VOCAB_NUMBER)
input_matrix = np.zeros((num_words, DIMENSION_LENGTH) )
for word, i in word2index.items():
    embedding = words_embeddings.get(word)
    if embedding is not None:
        if i< num_words:
            input_matrix[i] = embedding

# Model building
embedding_layer = Embedding(
    num_words,
    DIMENSION_LENGTH,
    input_length=SEQUENCE_LENGTH,
    weights=[input_matrix],
    trainable=False
)

input_ = Input(shape=(SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Bidirectional(LSTM(128,return_sequences = True))(x)
x =  GlobalMaxPool1D()(x)
output = Dense (len(possible_labels), activation='sigmoid')(x)

model = Model(inputs=input_, outputs=output)
model.compile(optimizer = 'adam', metrics=['accuracy'], loss='binary_crossentropy')
h = model.fit(data, labels, epochs=10, batch_size=500, validation_split=0.2)

###################################################
p = model.predict(data)
aucs = []
for i in range(len(possible_labels)):
    auc = roc_auc_score(labels[:,i], p[:,i])
    aucs.append(auc)
    print (auc)
print(np.mean(aucs))
###################################################
print ('Reading test keywords and labels ...')
df = pd.read_csv("./data/train_pirvu.csv")
keywords = df["keyword"].fillna("DUMMY KEYWORDS").values 
possible_labels = ['INF', 'NAV', 'TRA'] # three binary columns 
labels = df[possible_labels].values
print ('Min:', min(len(str(s)) for s in keywords))
print ('Max:', max(len(str(s)) for s in keywords))

print('Tokenizing & padding ...')
sequences = tokenizer.texts_to_sequences(keywords)

data = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)
word2index = tokenizer.word_index 
print ('Nomebr of Found tokens:%s' %len(word2index))


num_words = min(len(word2index)+1, MAX_VOCAB_NUMBER)
input_matrix = np.zeros((num_words, DIMENSION_LENGTH) )
for word, i in word2index.items():
    embedding = words_embeddings.get(word)
    if embedding is not None:
        if i< num_words:
            input_matrix[i] = embedding


p = model.predict(data)
aucs = []
for i in range(len(possible_labels)):
    auc = roc_auc_score(labels[:,i], p[:,i])
    aucs.append(auc)
    print (auc)
print(np.mean(aucs))

