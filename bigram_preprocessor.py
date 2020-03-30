from preprocessor import Preprocessor
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
np.random.seed(2018)


class BigramPreprocessor(Preprocessor):

    def preprocess(self, text):
        #text = strip_newline(text)
        words = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS:
                words.append(token)
        #words = list(sent_to_words(text))
        #words = remove_stopwords(words)
        #words = list(words)
        bigram = gensim.models.Phrases(words, min_count = 15)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        #bigrams = [bigram_mod[review] for review in words]
        bigrams = bigram_mod[words]
        return bigrams

    def create_dictionary(self):
        dictionary = gensim.corpora.Dictionary(self.processed_docs)
        dictionary.filter_extremes(no_below=15, no_above=0.35, keep_n=10000)
        dictionary.compactify()
        return dictionary
 
