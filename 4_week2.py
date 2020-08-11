# <Sarcasm classifier>
import os
import nltk.data
import csv
import json
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# ==================================================================================================================================================================
# Initialize global variables
# 최대 10000개의 vocabularies 들을 handle 할 수 있다.
vocab_size    = 10000
embedding_dim = 16
# maximum words contained in a sentence
max_length    = 100
# 끝에서 부터 내용들이 유실된다
trunc_type    = 'post'
# 0 을 뒤에서 부터 padding 해준다
padding_type  = 'post'
oov_tok       = "<OOV>"
# This data has 27000 records, we are going to use 20000 for training and the other for testing
training_size = 20000

# ==================================================================================================================================================================
# Open json file and pre-process it
with open('Sarcasm_Headlines_Dataset.json','r') as f:
    data = f.read()

data = "[" + data.replace("}", "},", data.count("}")-1) + "]"
data_store = json.loads(data)

# print(data_store)
# Data_store is a group of 'dictionary'

# ==================================================================================================================================================================
sentences = []
labels    = []

# Item is a 'dictionary type' variable
for item in data_store:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

# ==================================================================================================================================================================
# Split the corpus into training and validation sets
training_sentences = sentences[0:training_size]
testing_sentences  = sentences[training_size:]

training_labels = labels[0: training_size]
testing_labels  = labels[training_size:]

# pre-process 'sentences' and 'labels' to numpy array
training_sentences = np.array(training_sentences)
testing_sentences  = np.array(testing_sentences)

training_labels = np.array(training_labels)
testing_labels  = np.array(testing_labels)

# ==================================================================================================================================================================
# Now we have training and test sets of sentences and labels, it is time to 'sequence' them with 'padding'

# Generate an instance of Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
# Takes in the data and encodes it
tokenizer.fit_on_texts(training_sentences)
# Tokenizer provides a word index property which returns a dictionary containing k-v pairs
word_index = tokenizer.word_index

# Sequencing and Padding on training data
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded    = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Sequencing and Padding on testing data
testing_sequences  = tokenizer.texts_to_sequences(testing_sentences)
testing_padded     = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# ==================================================================================================================================================================
# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

num_epochs = 30
history = model.fit(training_padded,
                    training_labels,
                    epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels),
                    verbose=2)

# ==================================================================================================================================================================
def plot_graphs(_history, _string):
    # training
    plt.plot(_history.history[_string])
    # testing (validation)
    plt.plot(_history.history['val_'+_string])
    plt.xlabel("Epochs")
    plt.ylabel(_string)
    plt.legend([_string, 'val_'+_string])
    plt.show()

# 1. plot is about accuracy
plot_graphs(history, 'accuracy')

# 2. plot is about loss
plot_graphs(history, 'loss')

# ==================================================================================================================================================================
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim) = (10000, 16)

# helper function: k-v to v-k
'''
Now the data looks like this(left) but would like to change like this(right)

Hello : 1                                       1 : Hello
World : 2                                       2 : World
How   : 3                   ->                  3 : How
Are   : 4                                       4 : Are
You   : 5                                       5 : You
'''
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Write files
out_m = io.open('tsv_files/4week2_meta.tsv', 'w', encoding='utf-8') # word
out_v = io.open('tsv_files/4week2_vecs.tsv', 'w', encoding='utf-8') # vector value (number)

for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num] # word
    embeddings = weights[word_num]      # vector value
    out_m.write(word+'\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
out_v.close()
out_m.close()

sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))