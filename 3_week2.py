# <IMDB REVIEWS>
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
import io

from tensorflow.keras.preprocessing.text     import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
# We are going to use 'imdb'

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# training_sentences contains reviews
# training_labels    contains negative(0) or positive(1)
# s : sentence, l : label
for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())


print(len(testing_sentences))
print(len(training_sentences))
print(testing_labels)

# convert from straight array to numpy array
training_labels_final = np.array(training_labels)
testing_labels_final  = np.array(testing_labels)

# tokenize our sentences
vocab_size    = 10000
embedding_dim = 16
max_length    = 120
# cut off back of the review
trunc_type    = 'post'
oov_tok       = "<OOV>"

# num_words = vocab_size = 10000 it will take the 10000 most common words
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)

# all sentences contains 120 words each
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

# TESTING
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

# build a model
model = tf.keras.Sequential([
    # * Following line is very important (Embedding) *
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    # The result of the embedding will be a 2D array with the
    # 'length of the sentence' and the 'embedding dimension'
    # Therefore we need Flatten layer to make it to 1D array

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded,
          training_labels_final,
          epochs=num_epochs,
          validation_data=(testing_padded, testing_labels_final))

print("=====TESTING=====")
test_loss, test_acc = model.evaluate(testing_padded, testing_labels_final)
print("test_acc: ", test_acc)

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

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])

# write files
out_m = io.open('tsv_files/3week2_meta.tsv', 'w', encoding='utf-8') # word
out_v = io.open('tsv_files/3week2_vecs.tsv', 'w', encoding='utf-8') # vector value (number)

for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num] # word
    embeddings = weights[word_num]      # vector value
    out_m.write(word+'\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
out_v.close()
out_m.close()
