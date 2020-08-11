import os
import nltk.data
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

directory = 'bbc/tech'
sentences = ''
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
#         print(filename)
        with open("bbc/tech/"+filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
                sentences = sentences + ' '.join(sent_detector.tokenize(line.strip()))



stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

all_sentences = []
for word in stopwords:
    token = " " + word + " "
    sentences = sentences.replace(token, " ")
    sentences = sentences.replace("  ", " ")
    sentences.replace('"', '')

all_sentences.append(sentences)

print(all_sentences)
