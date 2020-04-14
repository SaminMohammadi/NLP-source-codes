from __future__ import print_function, division
from builtins import range

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import LSTM, Dense, Input, Embedding
from keras.models import Model

M = 28
SEQUENCE_LENGTH = 100
DIMENTION = 100
MAX_VOCAB_SIZE = 20000
BATCH_SIZE = 120
EPOCH = 2

input_text = []
target_text = []
with open("./data/robert_frost.txt") as f:
    for line in f:
        line = line.rstrip()
        if not line:
            continue


        input_line = '<sos> ' + line
        target_line = line + ' <eos>'
        input_text.append(input_line)
        target_text.append(target_line)

all_lines = input_text + target_text

tokenizer = Tokenizer(num_words= MAX_VOCAB_SIZE , filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(all_lines)
word2ind = tokenizer.word_index

input_sequences = tokenizer.texts_to_sequences(input_text)
target_sequences = tokenizer.texts_to_sequences(target_text)

max_sequence_length_from_data = max(len(s) for s in input_sequences)
print('number of unique words found: %s' % len(word2ind))
print('max length of sequences %s' % max_sequence_length_from_data)
assert('<sos>' in word2ind)
assert('<eos>' in word2ind)

max_sequence_len = min (max_sequence_length_from_data, SEQUENCE_LENGTH)
input_sequences = pad_sequences(input_sequences, maxlen= max_sequence_length_from_data)
target_sequences = pad_sequences(target_sequences, maxlen= max_sequence_length_from_data)
print('input sequence shape:', input_sequences.shape)

word_embeddins = {}
print ("loading in embeddings...")
with open ("./data/glove.6B.%sd.txt" % DIMENTION, encoding='utf8') as f:
    for line in f:
        items = line.split()
        word = items[0]
        embedding = items [1:]
        word_embeddins[word] = embedding
print ('number of words with embeddings: %s' % len(word_embeddins))

num_words = min(MAX_VOCAB_SIZE, len(word2ind)+1)
embedding_matrix = np.zeros(shape=(num_words, DIMENTION))
for word, i in word2ind.items():
    embedding_vector = word_embeddins.get(word)
    if embedding_vector is not None:
        if i< num_words:
            embedding_matrix[i] = embedding_vector

### one-hot-targets 
one_hot_targets = np.zeros((len(input_sequences), max_sequence_len, num_words))
for i, sequence in enumerate(target_sequences):
    for j, word in enumerate(sequence):
        if word>0 :
            one_hot_targets[i,j,word] = 1


### Encoder ###
embedding_layer = Embedding(
    num_words,
    DIMENTION,
    #input_length = max_sequence_len,
    weights = [embedding_matrix]
    #trainable = False
)

input_encoder = Input(shape=(max_sequence_len,))
initial_h = Input(shape=(M,))
initial_c = Input(shape=(M,))
x = embedding_layer(input_encoder)
lstm = LSTM(M, return_sequences=True, return_state=True)
x, _, _ = lstm(x, initial_state=[initial_h, initial_c])
dense = Dense(num_words, activation='softmax')
h1 = dense(x)
model = Model(inputs=[input_encoder, initial_h, initial_c], outputs=h1)

model.compile(
    loss= 'categorical_crossentropy',
    optimizer='adam',
    metrics = ['accuracy']
)
z = np.zeros((len(input_sequences), M))
model.fit([input_sequences, z, z], one_hot_targets, batch_size= BATCH_SIZE, epochs= EPOCH, validation_split=0.2)

### test ###
input_test = Input(shape=(1,))
x = embedding_layer(input_test)
x, h, c = lstm (x, initial_state=[initial_h,initial_c])
h2 = dense(x)
sampling_model = Model(inputs= [input_test, initial_h, initial_c], outputs= [h2, h, c])

idx2word = tokenizer.index_word
def sample_line():
    np_input = np.array([[word2ind['<sos>']]])
    h = np.zeros((1,M))
    c = np.zeros((1,M))

    eos=word2ind['<eos>']
    output_sequence = []

    for _ in range(max_sequence_len):
        o, h, c = sampling_model.predict([np_input, h,c])
        #shape of o???
        probs = o[0,0]
        if np.argmax(probs) == 0:
            pritn ("wtf")
        probs[0] = 0
        probs /= probs.sum()
        idx = np.random.choice(len(probs), p= probs)
        if idx == eos:
            break
        output_sequence.append(idx2word.get(idx, '<WTF %s>' % idx))

        np_input[0,0] = idx
    return ' '.join(output_sequence)

while True:
    for _ in range(6)    :
        print (sample_line())
    ans = input ("generate another ? y/n")
    if ans and str(ans[0]).lower.startswith('n'):
        break




