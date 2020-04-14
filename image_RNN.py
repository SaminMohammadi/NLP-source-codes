from __future__ import print_function, division

from keras.datasets import mnist
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, GlobalMaxPool1D, Dense, LSTM
from keras.layers import Concatenate 
import numpy as np
from sklearn.metrics import roc_auc_score



(x_train, y_train) , (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[10])
plt.show()

DIMENSION = x_train[0].shape[1]
SEQUENCE_LENGTH = x_train[0].shape[0]

NUM_INPUTS = x_train.shape[0]

from keras.utils import to_categorical

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

#### 1 ####
input_ = Input(shape=(SEQUENCE_LENGTH,DIMENSION))
embedding_layer = Embedding(
    NUM_INPUTS,
    DIMENSION,
    input_length = (SEQUENCE_LENGTH,DIMENSION)
)

#x = embedding_layer(input_)
x = Bidirectional(LSTM(128, return_sequences = True))(input_)
x = GlobalMaxPool1D()(x)
x = Dense(10, activation='sigmoid')(x)

#### 2 ####
DIMENSION_ = x_train[0].shape[0]
SEQUENCE_LENGTH_ = x_train[0].shape[1]

input__ = Input(shape=(SEQUENCE_LENGTH_,DIMENSION_))
embedding_layer = Embedding(
    NUM_INPUTS,
    DIMENSION_,
    input_length = (SEQUENCE_LENGTH_,DIMENSION_)
)

#x_ = embedding_layer(input__)
x_ = Bidirectional(LSTM(128, return_sequences = True))(input__)
x_ = GlobalMaxPool1D()(x_)
x_ = Dense(10, activation='sigmoid')(x_)

x__ = Concatenate()([x,x_])

output = Dense(10, activation='sigmoid')(x__)

model = Model(inputs= [input_, input__] , outputs = output)
model.compile(optimizer = 'adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')


x_train_ = x_train.transpose((0,2,1))#.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test_ = x_test.transpose((0,2,1))#.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

r = model.fit([x_train, x_train_], y_train, batch_size=120, epochs=2, validation_split=0.2)

p = model.predict([x_test, x_test_])
print(p.shape)
roc_auc_score(to_categorical(y_test), p)



