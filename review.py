########################################
# CNN
########################################

from keras.datasets import mnist
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[2])
plt.show()

# why the 4th number is 1 ? grayscale
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Dense, Flatten 


model = Sequential()
model.add(Conv2D(filters=32, input_shape = (28,28,1), activation = 'relu', kernel_size=3)) # padding='same'/'valid' default is not the 'same'
model.add(MaxPooling2D(pool_size=(2,2), strides=1))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation=’softmax’))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)
model.predict(x_test[3:4])

'''
######################################
#  RNN
######################################
import json
file_path = "./data/patent_machine_learning_data.json"
data = []
with open(file_path) as f:
    data = json.load(f)
patent_title = []
patent_abstract = []
for item in data["patents"]:
    patent_title.append(item["patent_title"])
    patent_abstract.append(item["patent_abstract"])

from keras.preprocessing.text import Tokenizer
#OOV replaces the out-of-vocabs words by this token at texts_to_sequence
tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', oov_token=None) 
tokenizer.fit_on_texts(patent_abstract) # build the internal vocans 
sequences = tokenizer.texts_to_sequences(patent_abstract) # Converts tokens to integer numbers 

#check the sequences to texts

word_ind = tokenizer.index_word
#sequence_100_vocabs = ' '.join(index_word[w] for w in sequences[100][:100])

############################################
#   Builing the dataset

features = []
labels = []

training_length = 50
import numpy as np

for seq in sequences:
    for i in range(training_length, len(seq)):
        # Extract the features and label
        extract = seq[i - training_length:i + 1]
        # Set the features and label
        features.append(extract[:-1])
        labels.append(extract[-1])

features = np.array(features)


# one-hot encodeing of labels 
num_words = len(tokenizer.index_word)+1
label_array = np.zeros((len(features), num_words), dtype=np.int8)
for index, word_index in enumerate(labels):
    label_array[index][word_index] = 1


#labels[100]
#label_array[100][labels[100]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, label_array, test_size= 0.2)

import numpy as np
#######################################
#   Embedding features
glove_vectors = './data/glove.6B.100d.txt'
glove = np.loadtxt(glove_vectors, dtype='str', comments=None , encoding='utf8')
# Extract the vectors and words
vectors = glove[:, 1:].astype('float')
words = glove[:, 0]

# Create lookup of words to vectors
word_lookup = {word: vector for word, vector in zip(words, vectors)}
# New matrix to hold word embeddings
embedding_matrix = np.zeros((num_words, vectors.shape[1]))
for i, word in enumerate(word_ind.keys()):
    vector = word_lookup.get(word, None)
    # Record in matrix
    if vector is not None:
        embedding_matrix[i + 1, :] = vector
#######################################
#   Building Model

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model = Sequential()

model.add(
    Embedding(input_dim = num_words,
    input_length = training_length,
    output_dim = 100,
    weights = [embedding_matrix],
    trainable = False,
    mask_zero = True))

model.add(Masking(mask_value=0.0))
model.add(LSTM(64, return_sequences=False, dropout = 0.1, recurrent_dropout=0.1))
model.add(Dense(num_words, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#########################################
#   Call model

from keras.callbacks import EarlyStopping, ModelCheckpoint
# Create callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=5), ModelCheckpoint('./models/model.h5', save_best_only=True, save_weights_only=False)]
            
history = model.fit(X_train,  y_train, 
                    batch_size=2048, epochs=10,
                    callbacks=callbacks,
                    validation_data=(X_test, y_test))


from keras.models import load_model
# Load in model and evaluate on validation data
model = load_model('./models/model.h5')
model.evaluate(X_test, y_test)
'''

