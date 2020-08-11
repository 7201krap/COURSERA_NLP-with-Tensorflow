import json
import tensorflow as tf
import csv
import random
import numpy as np
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

tokenizer = Tokenizer()

data = open('irish-lyrics-eof.txt').read()

corpus = data.lower().split("\n")

# tokenize a list of sentences
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1 # plus one because of OOV

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Find the length of the longest sentence in the corpus
max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Split our sequences into our x's and y's. x = xs, y = labels
xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]

# One-hot encode the labels. This is actually is classification problem.
# Where given a sequence of words, I can classify from the corpus what the next word would likely be
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1), # Why we minus 1? the last element is the label
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

history = model.fit(xs, ys, epochs=100, verbose=1)

def plot_graphs(history_, string):
    plt.plot(history_.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

plot_graphs(history, 'accuracy')

seed_text = "Laurence went to dublin"
next_words = 100 # We will going to ask it for the next 100 words

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted  = model.predict_classes(token_list, verbose=1)
    output_word = ''
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text = seed_text + ' ' + output_word

print(seed_text)