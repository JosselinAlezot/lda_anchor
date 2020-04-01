from user_framer import UserFramer
from post_framer import PostFramer
from group_framer import GroupFramer
from lda_modeler import LdaModeler
from stem_preprocessor import StemPreprocessor
from bigram_preprocessor import BigramPreprocessor
from vectorizer import Vectorizer
from classifier import Classifier


# Create dataframes from data
user_frames = UserFramer('./data_t1/', ['0', '1'])
user_frames.create_frames()
user_frames_concat = user_frames.get_concat_frames()


### Bigram strategy ###

# Sets number of topics for the LDA models
num_topics = 11

# Creates preprocessed docs and LDA model considering each user as a document
bigram_docs = BigramPreprocessor(user_frames_concat)
bigram_lda = LdaModeler(bigram_docs.get_dictionary(), bigram_docs.get_bow(), num_topics)
#user_lda.topics_printer()

# Creates vectors
bigram_vecs = Vectorizer(user_frames_concat, bigram_docs.get_bow(), bigram_lda, num_topics)

# Creates classifier
print("\nRésultats LDA entrainé sur les deux classes pour %d topics utilisant des bigrams\n" % num_topics)
bigram_classifer = Classifier(bigram_vecs.get_vectors(), user_frames_concat.label)


### Stemming strategy ###

# Sets number of topics for the LDA models
num_topics = 10

# Creates preprocessed docs and LDA model considering each user as a document
stem_docs = StemPreprocessor(user_frames_concat)
stem_lda = LdaModeler(stem_docs.get_dictionary(), stem_docs.get_bow(), num_topics)
#user_lda.topics_printer()

# Creates vectors
stem_vecs = Vectorizer(user_frames_concat, stem_docs.get_bow(), stem_lda, num_topics)

# Creates classifier
print("\nRésultats LDA entrainé sur les deux classes pour %d topics utilisant des stems\n" % num_topics)
stem_classifer = Classifier(stem_vecs.get_vectors(), user_frames_concat.label)


### Previously tested strategy ###

"""
# Creates LDA models and vectors for label 0
proc_docs_0 = Preprocessor(user_frames.get_frame('0'))
proc_docs_1 = Preprocessor(user_frames.get_frame('1'))

lda_0 = LdaModeler(proc_docs_0.get_dictionary(), proc_docs_0.get_bow(), num_topics)
#user_lda_0.topics_printer()
vecs0_lda0 = Vectorizer(user_frames.get_frame('0'), proc_docs_0.get_bow(), lda_0, num_topics)
vecs1_lda0 = Vectorizer(user_frames.get_frame('1'), proc_docs_1.create_alt_bow(proc_docs_0.get_dictionary()), lda_0, num_topics)

#print("Vecs0_lda0 : ")
#print(vecs0_lda0.get_vectors())
#print("Vecs1_lda0 : ")
#print(vecs1_lda0.get_vectors())

vectors_lda0 = vecs0_lda0.get_vectors() + vecs1_lda0.get_vectors()

for i in range(len(vectors_lda0)):
    vectors_lda0[i].extend([len(user_frames.get_concat_frames().iloc[i].text)])

print("\nRésultats LDA entrainé sur la classe 0 pour %d topics\n" % num_topics)
user_classifer_lda0 = Classifier(vectors_lda0, user_frames_concat.label)
"""
"""
# Creates LDA models and vectors for label 1

frames = UserFramer('./data_t1/', ['0', '1'])
frames.create_frames()
proc_docs_0 = Preprocessor(frames.get_frame('0'))
lda_0 = LdaModeler(proc_docs_0.get_dictionary(), proc_docs_0.get_bow())
#user_lda_0.topics_printer()

proc_docs_1 = Preprocessor(frames.get_frame('1'))
lda_1 = LdaModeler(proc_docs_1.get_dictionary(), proc_docs_1.get_bow())
#user_lda_1.topics_printer()
vecs0_lda1 = Vectorizer(frames.get_frame('0'), proc_docs_0.create_alt_bow(proc_docs_1.get_dictionary()), lda_1)
vecs1_lda1 = Vectorizer(frames.get_frame('1'), proc_docs_1.get_bow(), lda_1)
#print("Vecs0_lda1 : ")
#print(vecs0_lda1.get_vectors())
#print("Vecs1_lda1 : ")
#print(vecs1_lda1.get_vectors())

vectors_lda1 = vecs0_lda1.get_vectors() + vecs1_lda1.get_vectors()

for i in range(len(vectors_lda1)):
    vectors_lda1[i].extend([len(frames.get_concat_frames().iloc[i].text)])

print("Résultats LDA entrainé sur la classe 1")
user_classifer_lda1 = Classifier(vectors_lda1, user_frames_concat.label)
"""
#print("final vectors")
#print(vectors)


"""
# Creates 
vecs0 = []
if (len(vecs0_lda0.get_vectors()) - len(vecs0_lda1.get_vectors()) == 0):
    print("Entre condition 0")
    for i in range(len(vecs0_lda0.get_vectors())):
        vecs0.append(vecs0_lda0.get_vectors()[i] + vecs0_lda1.get_vectors()[i])

vecs1 = []
if (len(vecs1_lda0.get_vectors()) - len(vecs1_lda1.get_vectors()) == 0):
    print("Entre condition 1")
    for i in range(len(vecs1_lda0.get_vectors())):
        vecs1.append(vecs1_lda0.get_vectors()[i] + vecs1_lda1.get_vectors()[i])

vectors = vecs0 + vecs1
"""


