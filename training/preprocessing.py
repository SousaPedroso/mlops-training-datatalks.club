"""
Module to pre-process texts
"""
import re
import string

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def tokenize(dataset):
    # Initialize stopwords and stemmer
    stop_words = set(stopwords.words("portuguese"))
    stemmer = SnowballStemmer("portuguese")

    # Tokenize review text
    tokens = []

    text = dataset["review_text"]
    if text is not None:
        # Split review into words using regular expressions
        words = re.findall(r"\b\w+\b", text.lower())

        tokens = [
            stemmer.stem(word)
            for word in words
            if word not in stop_words and word not in string.punctuation and len(word) > 1
        ]

        tokens = " ".join([word for word in tokens if not word.isdigit()])

    if not tokens:
        tokens = " "

    dataset["tokens_list"] = tokens
    return dataset
