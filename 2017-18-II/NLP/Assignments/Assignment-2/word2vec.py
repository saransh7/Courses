import preprocessing as data
import BOW_tfidf_norm as bowscript
import numpy as np
import nltk
import string
import gensim
from gensim.models import Word2Vec
from sklearn import neural_network
vectorizer = data.vectorizer
corpus_train, docs_train = data.define_corpus(data.text_train)
corpus_test, docs_test = data.define_corpus(data.text_test)

word2vec_model = Word2Vec(corpus_train, min_count=1)

def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    #print len(doc)
    temp = []
    for word in doc:
      if word2vec_model.wv.vocab.has_key(word):
          temp.append(word2vec_model[word])
    return (np.mean(temp, axis=0)).tolist()

def wordtovec(corpus):
    word2vec_mean = []
    for doc in corpus: #look up each doc in model
       word2vec_mean.append(document_vector(word2vec_model, doc))
    return word2vec_mean


train = wordtovec(corpus_train)
test = wordtovec(corpus_test)
#print word2vec_test
print "word2vec--------------------"
bowscript.svm_classifier(train,data.posorneg_train,test)
bowscript.bayes_classifier(train,data.posorneg_train,test)
bowscript.logistic_classifier(train,data.posorneg_train,test)
bowscript.mlp_classifier(train,data.posorneg_train,test)
## tfidf weights
tfidf_train = bowscript.tfidf_train
tfidf_test = bowscript.tfidf_test

def w2v_tfidf(corpus,tfidf):
  i = 0
  j = 0
  tfidf_vectors = []
  for doc in corpus:
     word2vec_tfidf = []
     #print doc
     temp = vectorizer.transform(doc)
     temp = sum(temp)
     a = temp.toarray()[0]
     #print abowscript.svm_classifier(train,data.posorneg_train,test)
     b = tfidf[i]
     #print b
     word2vec_tfidf.append(np.multiply(a,b).tolist())
     word2vec_tfidf = word2vec_tfidf[0]
     word_vectors = []
     length = len(word2vec_tfidf)
     for j in range(length):
         if word2vec_tfidf[j] > 0.0:
             word_vectors.append(str((vectorizer.get_feature_names())[j]))
     #print word_vectors 
     word2vec_tfidf = list(filter(lambda x: (x > 0.0), word2vec_tfidf))
     temp3 = []
     for k in range(len(word2vec_tfidf)):
        if word2vec_model.wv.vocab.has_key(word_vectors[k]):
         temp3.append(word2vec_tfidf[k] * word2vec_model[word_vectors[k]])
        ##final wrd2vec_tfidf vectors
     tfidf_vectors.append(np.mean(np.array(temp3), axis = 0).tolist())
     #tfidf_vectors(np.mean(temp3)
     i = i + 1
  return tfidf_vectors
  
train = np.array(w2v_tfidf(corpus_train, tfidf_train))
#print train
tfidf = bowscript.tfidf_test
test = np.array(w2v_tfidf(corpus_test, tfidf_test))

print "word2vec_tfidf----------------------"
bowscript.svm_classifier(train,data.posorneg_train,test)
bowscript.bayes_classifier(train,data.posorneg_train,test)
bowscript.logistic_classifier(train,data.posorneg_train,test)
bowscript.mlp_classifier(train,data.posorneg_train,test)

#print data.posorneg_test