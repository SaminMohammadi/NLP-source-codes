from __future__ import print_function, division

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D, Input
from keras.models import Model, Sequential
from sklearn.metrics import roc_auc_score 


MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE= 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

print('Loading word vectors...')
word2vec = {}
with open(os.path.join('./data/glove.6B.%sd.txt' % EMBEDDING_DIM), encoding='utf8')  as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print ('Found %s word vectors.' % len(word2vec))



print('Loading in comments...')

train = pd.read_csv('./data/toxic_comments/train.csv')
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult","identity_hate"]
targets = train[possible_labels].values

print ("Max sentence length:", max(len(s) for s in sentences))
print ("Min sentence length:", min(len(s) for s in sentences))
s = sorted(len(s) for s in sentences)
print("Median sequence length:", s[len(s) // 2])

tokenizer = Tokenizer(num_words= MAX_VOCAB_SIZE, lower=True, oov_token= 'UNKOWN')
tokenizer.fit_on_texts(sentences)
tokenizer.num_words = MAX_VOCAB_SIZE
sequences = tokenizer.texts_to_sequences(sentences)
# print ("sequences : ", sequences); exit()

word2idx = tokenizer.word_index
print("Number of unique words:%s" %len(word2idx))

# sequences should be padded
data = pad_sequences(sequences, maxlen= MAX_SEQUENCE_LENGTH) # 1. Pre-padding is default 2. Tokenizer indexes start from 1 and 0 is reserved for padding 
# IMPORTANT: Default for padding and truncating is 'pre', because the last words are more important when we are going to take into account the prediction after the last word
print ("Shaoe of data tensor: ", data.shape)

print ("Filling pre-trained embeddings...")
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1) # +1 , because UNKOWN is also counted in addition to first #words
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector


# constructing V*D matrix for embedding layer 
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False
)

print('Builing the model...')

input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))

x = embedding_layer(input_)
print ('Embedding_in:', x.shape)

x = Conv1D(128, kernel_size=3, activation='relu')(x)
print ('Conv1D_in:', x.shape)

x = MaxPooling1D(pool_size=3)(x)
print ('MaxPooling_in:', x.shape)

x = Conv1D(128, kernel_size=3, activation = 'relu')(x)
print ('Conv1D_in:', x.shape)

x = MaxPooling1D(pool_size=3)(x)
print ('MaxPooling_in:', x.shape)

x = Conv1D(128, kernel_size=3, activation = 'relu')(x)
print ('Conv1D_in:', x.shape)

x = GlobalMaxPooling1D()(x)
print ('MaxPooling_in:', x.shape)

x = Dense(128, activation='relu')(x)
print ('Dense1_in:', x.shape)

output = Dense(len(possible_labels), activation='sigmoid')(x)
print ('Dense1_in:', x.shape)
model = Model(input_, output)

'''
# paralle modeling 
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(possible_labels),activation='sigmoid'))
'''
model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

print('Training model...')
r = model.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.plot(r.history['accuracy'], label='accuracy')
plt.legend() 
plt.show()

p = model.predict(data)
aucs = []
for i in range(6):
    auc = roc_auc_score(targets[:,i], p[:,i])
    aucs.append(auc)
print(np.mean(aucs))
