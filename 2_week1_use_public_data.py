import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text     import Tokenizer

# it allows to use sentences of different lengths and use padding or truncation to
# make all of the sentences the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

# with open("sarcasm.json","r") as f:
#     datastore = json.load(f)

with open('Sarcasm_Headlines_Dataset.json','r') as f:
    data = f.read()

data = "[" + data.replace("}", "},", data.count("}")-1) + "]"
datastore = json.loads(data)

sentences = []
labels    = []
urls      = []

for item in datastore:
    sentences.append(item['headline'])
    labels.   append(item['is_sarcastic'])
    urls.     append(item['article_link'])


tokenizer = Tokenizer(oov_token="<OOV>")
# takes in the data and encodes it
# tokenize a list of sentences
tokenizer.fit_on_texts(sentences)
# tokenizer provides a word index property which returns a dictionary containing k-v pairs
# encode a list of sentences to use those tokens
word_index = tokenizer.word_index

sequences  = tokenizer.texts_to_sequences(sentences)
padded     = pad_sequences(sequences, padding='post')

print("How many words do we have? \n -> ", len(word_index))
print("Example sentence: \n -> ", sentences[2])
print("Corresponding example sequence: \n -> ", sequences[2])
print("Corresponding padded  sequence: \n -> ", padded[2])
print("There are total: ", padded.shape[0], "sentences.")
print("And longest sentence contains", padded.shape[1], "words.")
