import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text     import Tokenizer
# it allows to use sentences of different lengths and use padding or truncation to
# make all of the sentences the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

# Tokenizer 의 인스턴스 생성. 100은 상당히 큼, 왜냐하면 우리는 단지 5개의 unique 한 단어들을 가지고 있음.
# num_words = 100 it will take the 100 most common words
# OOV = outer vocabulary
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")

tokenizer.fit_on_texts(sentences)
# tokenizer provides a word index property which returns a dictionary containing k-v pairs
word_index = tokenizer.word_index

# turn sentences into a set of sequences for me
# 위에 있는 문장들이 token 들로 변경 된다.
sequences = tokenizer.texts_to_sequences(sentences)

# padding
padded = pad_sequences(sequences, maxlen=5)
# padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)

# padding='post'    -> 0 을 padding 할때 뒤에서 부터 padding
# maxlen=5          -> 0 padding 이 될때는 가장 긴 문장을 기준으로 0 이 padding 된다.
#                      maxlen 은 sequence 의 길이를 결정한다.
# truncating='post' -> maxlen 을 하면 단어들이 유실될 것인데, 기본적으로는 앞에서 부터 단어들이 유실된다
#                      끝에 있는 내용 부터 유실하고 싶다면, truncating='post' 를 해준다.

'''
[[ 0  0  0  5  3  2  4]
 [ 0  0  0  5  3  2  7]
 [ 0  0  0  6  3  2  4]
 [ 8  6  9  2  4 10 11]]
'''

print("word_index dictionary: \n", word_index)
# {'i': 1, 'love': 2, 'my': 3, 'dog': 4, 'cat': 5}
# Note that 'I' is lower-cased here + strips punctuation

print("=====Tokenized sentences=====")

print("sequences: \n", sequences)

print("padded: \n", padded)

print("=====TESTING=====")

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_sequences = tokenizer.texts_to_sequences(test_data)

test_padded = pad_sequences(test_sequences, maxlen=10)

# word_index 에 존재하지 않는 단어들은 texts_to_sequences 할때 반영되지 않는다.
# 따라서 밑의 결과값은 -> [[4, 2, 1, 3], [1, 3, 1]]
print("test_sequences: \n", test_sequences)
print("test_padded: \n", test_padded)
'''
[[0 0 0 0 0 5 1 3 2 4]
 [0 0 0 0 0 2 4 1 2 1]]
'''
