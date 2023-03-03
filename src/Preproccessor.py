# import nltk
# nltk.download('omw-1.4')
import string

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# import libraries for sequence padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.dataHandler import TrainingData


class TextPreprocessor:
    lemmatizer = ""
    def __init__(self):

        self.lemmatizer= WordNetLemmatizer()

    def preprocess_text(self,text):
        # Tokenize text
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(text)

        # Remove stop words
        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word.lower() not in stop_words]

        # Lemmatize words
        words = [self.lemmatizer.lemmatize(word) for word in words]
        completeSentences = " ".join(words)
        return completeSentences


    def feedPreprocessorAnArray(self,array):
        preprocessedArray = []
        for sent in array:
            preprocessedArray.append(self.preprocess_text(sent))

        return preprocessedArray

    def sequencePadding(self,textarray):

        tokenizer = Tokenizer(num_words=10000)
        # Fit the tokenizer on the input strings
        tokenizer.fit_on_texts(textarray)
        sequences = tokenizer.texts_to_sequences(textarray)

        max_len = max([len(seq) for seq in sequences])
        padded_sequences = pad_sequences(sequences, maxlen=29, padding='post')

        # Print the padded sequences
        return padded_sequences

#
