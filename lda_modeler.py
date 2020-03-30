import gensim


class LdaModeler:

    def __init__(self, dictionary, bow, topics):
        """
        Constructor for the LDA modeler. Creates the LDA model
        for the dictionary and bag-of-words corpus passed by parameters
        :param dictionary: Dictionary to be used to create the LDA model
        :param bow: Bag-of-words to be used to create the LDA model
        """
        self.lda_model = self.lda_modeling(dictionary, bow, topics)

    def get_model(self):
        """
        LDA model getter
        :return: LDA model
        """
        return self.lda_model

    def lda_modeling(self, dictionary, bow, topics):
        """
        Creates LDA model from dictionnary and bag-of-words
        :param dictionary: Dictionary to be used to create the LDA model
        :param bow: Bag-of-words to be used to create the LDA model
        :return: LDA model
        """
        lda_model = gensim.models.ldamulticore.LdaMulticore(
                           corpus=bow,
                           num_topics=topics,
                           id2word=dictionary,
                           chunksize=340,
                           workers=7,
                           passes=50,
                           eval_every = 1)
        return lda_model

    def get_document_topics(self, bow_corpus):
        """
        Topics getter of this LDA model for the given bag-of-words
        :param bow: Bag-of-words to be used to get the document topics
        :return: Document topics
        """
        return self.lda_model.get_document_topics(bow_corpus, minimum_probability=0.0)

    def topics_printer(self):
        """
        Prints the LDA model topics
        """
        print('Topics for this LDA model are :')
        for idx, topic in self.lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
        print('\n')

