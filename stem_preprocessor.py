from preprocessor import Preprocessor
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)


class StemPreprocessor(Preprocessor):
        
    def preprocess(self, text):
        """
        Preprocesses the text in order to create a dictionnary and a bag-of-words,
        removes stopwords and words under 3 characters,
        apply lemmatize and stem processings method
        :param text: Raw text from dataframe.
        :return: Preprocessed text ready to be used for LDA modeling.
        """
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(PorterStemmer().stem(WordNetLemmatizer().lemmatize(token, pos='v')))
        return result

    def create_dictionary(self):
        dictionary = gensim.corpora.Dictionary(self.processed_docs)
        dictionary.filter_extremes(no_below=25, no_above=0.4, keep_n=100000)
        return dictionary
