from __future__ import print_function, division
from builtins import range, input

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Bidirectional, LSTM, GRU
import numpy as np
import matplotlib.pyplot as plt

T = 8
D = 2
M = 3

X=np.random.randn(1,T,D)

input_ = Input(shape=(T,D))
rnn = Bidirectional(LSTM(M, return_state=False, return_sequences=False))
x = rnn(input_)
model = Model(inputs=input_, outputs=x)
o = model.predict(X)
print("o:", o)
