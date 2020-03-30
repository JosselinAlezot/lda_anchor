import gensim


class Vectorizer:
    def __init__(self, frame, bow_corpus, lda_model, num_topics):
        self.topics_vectors = self.topics_vectors(frame, bow_corpus, lda_model, num_topics)

    def topics_vectors(self, frame, bow_corpus, lda_model, num_topics):
        topics_vectors = []
        for i in range(len(frame)):
            top_topics = lda_model.get_document_topics(bow_corpus[i])
            topic_vec = [top_topics[i][1] for i in range(num_topics)]
            topic_vec.extend([frame.iloc[i].posts_counts])
            topic_vec.extend([len(frame.iloc[i].text)])
            topics_vectors.append(topic_vec)
        return topics_vectors

    def get_vectors(self):
        return self.topics_vectors
