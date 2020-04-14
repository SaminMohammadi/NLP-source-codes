from __future__ import print_function, division
from builtins import range 

from keras.layers import Embedding, LSTM, Dense, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

BATCH_SIZE = 120
EPOCHS = 5
SEQUENCE_LENGTH = 50
DIMENSION = 100 
LATENT_DIM = 50
MAX_NUM_WORDS = 20000
NUM_SAMPLES = 20000

word2vec = {}
print ('loading in pre-trained embeddings...')
with open (f"./data/glove.6B.{DIMENSION}d.txt", encoding ='utf8') as f:
    for line in f:
        vector = line.split()
        word = vector[0]
        embedding = np.asarray(vector[1:], dtype='float32')
        word2vec[word] = embedding


print('loading in input and target data')
input_texts = []
target_texts = []
target_text_inputs = []

t = 0

for line in open('./data/spa.txt', encoding='utf8'):
    t += 1 
    if t > NUM_SAMPLES:
        break

    if '\t' not in line :
        continue 

    input_text, translation = line.split('\t')[0:2]

    target_text = translation + '<eos>'
    target_text_input = '<sos>' + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_text_inputs.append(target_text_input)
print ('num of sentence pairs:',len(input_texts))



## input and output are in different languages, needs to have two separate tokenizers
print('tokenization...')
tokenizer_source = Tokenizer(num_words= MAX_NUM_WORDS)
tokenizer_source.fit_on_texts(input_texts)
sequences = tokenizer_source.texts_to_sequences(input_texts)
word2ind_inputs = tokenizer_source.word_index
print ('number of tokens found in input:', len(word2ind_inputs))

tokenizer_target = Tokenizer(num_words= MAX_NUM_WORDS, filters= '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer_target.fit_on_texts(target_texts + target_text_inputs)
target_sequences = tokenizer_target.texts_to_sequences(target_texts)
target_sequences_input = tokenizer_target.texts_to_sequences(target_text_inputs)
word2ind_target = tokenizer_target.word_index 
print ('number of tokens found in target:', len(word2ind_target))

print('Padding...')
max_seq_len = max (len(s) for s in sequences)
sequences = pad_sequences(sequences, maxlen= max_seq_len)

max_seq_len_target = max(len(s) for s in target_sequences)
target_sequences = pad_sequences(target_sequences, maxlen= max_seq_len_target, padding='post')
target_sequences_input = pad_sequences(target_sequences_input, maxlen= max_seq_len_target, padding='post')

print ('Filling pre-trained embeddings...')
num_words = min(MAX_NUM_WORDS, len(word2ind_inputs)+1)
embedding_matrix = np.zeros((num_words, DIMENSION))
for word,i in word2ind_inputs.items():
    if i<num_words:
        embedding = word2vec.get(word)
        if embedding is not None:
            embedding_matrix[i] = embedding

num_words_target = min(MAX_NUM_WORDS, len(word2ind_target)+1)

decoder_targets_one_hot = np.zeros((
    len(input_texts),
    max_seq_len_target,
    num_words_target
), dtype= 'float32')

for i, d in enumerate(target_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] =1 

embedding_layer_encoder = Embedding(
    num_words,
    DIMENSION,
    input_length=max_seq_len,
    weights=[embedding_matrix],
    trainable=False
)

encoder_input = Input(shape=(max_seq_len,))
lstm = LSTM(LATENT_DIM, return_state = True)
x = embedding_layer_encoder(encoder_input)
output, h, c = lstm(x)
Encoder_status = [h, c]


decoder_input = Input(shape=(max_seq_len_target,))
embedding_layer_target = Embedding(
    num_words_target, 
    LATENT_DIM
)

x_decoder = embedding_layer_target(decoder_input)
lstm_decoder = LSTM(LATENT_DIM, return_state = True , return_sequences = True)
decoder_output , _, _ = lstm_decoder(x_decoder, initial_state= Encoder_status)
translation_output = Dense(num_words_target, activation= 'sigmoid')(decoder_output)

model = Model(inputs=[encoder_input, decoder_input], outputs= [translation_output] ) 
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
r= model.fit([sequences, target_sequences_input], decoder_targets_one_hot, batch_size= BATCH_SIZE, epochs= EPOCHS, validation_split=0.2)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='vali_loss')
plt.legend()
plt.show()


plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()









