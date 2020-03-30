import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from abc import ABC, abstractmethod


class Preprocessor(ABC):

    def __init__(self, frame):
        self.frame = frame
        self.processed_docs = self.frame['text'].map(self.preprocess)
        self.dictionary = self.create_dictionary()
        self.bow_corpus = self.create_bow(self.dictionary)
    
    @abstractmethod
    def preprocess(self, text):
        """
        Abstract method to be implemented by subclasses
        """
        pass

    @abstractmethod
    def create_dictionary(self):
        """
        Abstract method to be implemented by subclasses
        """
        pass
        
    def get_dictionary(self):
        return self.dictionary

    def create_bow(self, dictionary):
        return [dictionary.doc2bow(doc) for doc in self.processed_docs]

    def get_bow(self):
        return self.bow_corpus
 
