import preprocessing as data
import BOW_tfidf_norm as bowscript
import csv
import os
import numpy as np
import nltk
import string
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords


corpus_train, docs_train = data.define_corpus(data.text_train)
corpus_test, docs_test = data.define_corpus(data.text_test)
vocab = []

file = "/home/saransh/Desktop/nlp assignment 2/glove.6B.50d.txt"
def loadGloveModel(gloveFile):
    print ("Loading Glove Model") 
    with open(gloveFile) as f:
       content = f.readlines()
    
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        vocab.append(word)
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
     
model= loadGloveModel(file)

#print corpus2
def document_vector_glove(model, doc):
    temp = []
    for word in doc:
        #print word
        temp.append((model[word]).tolist())
    return (np.mean(np.array(temp), axis=0)).tolist()
    #print np.mean(np.array(temp)


def glove(corpus):
   corpus2 = []
   for corpora in corpus:
        tokens_filtered = []
        for token in corpora:
          if token in vocab:
             tokens_filtered.append(token)
        corpus2.append(tokens_filtered)
   
   glove_mean = []
   for doc in corpus2: #look up each doc in model
      glove_mean.append(document_vector_glove(model, doc))
   return glove_mean

train = glove(corpus_train)
test = glove(corpus_test)
print "glove"
bowscript.svm_classifier(train,data.posorneg_train,test)
bowscript.bayes_classifier(train,data.posorneg_train,test)
bowscript.logistic_classifier(train,data.posorneg_train,test)

#glove tfidf

transformer = TfidfTransformer(smooth_idf=False)

def glove_tfidf(corpus):
   i = 0
   corpus2 = []
   tfidf_vectors = []
   for corpora in corpus:
        tokens_filtered = []
        for token in corpora:
          if token in vocab:
             tokens_filtered.append(token)
        corpus2.append(tokens_filtered)
   
   docs2 = [" ".join(corpora) for corpora in corpus2] 
   bow2 = data.vectorizer.transform(docs2)
   count2 = bow2.toarray()
   tfidf2 = transformer.fit_transform(count2)

   for doc in corpus2:
      glove_tfidf = []
      #print doc
      temp = data.vectorizer.transform(doc)
      temp = sum(temp)
      a = temp.toarray()[0]
      #print a
      b = tfidf2.toarray()[i]
      #print b
      glove_tfidf.append(np.multiply(a,b).tolist())
      glove_tfidf = glove_tfidf[0]
      #print glove_tfidf
      word_vectors = []
      length = len(glove_tfidf)
      for j in range(length):
          if glove_tfidf[j] > 0.0:
              word_vectors.append(str((data.vectorizer.get_feature_names())[j]))
      #print word_vectors 
      glove_tfidf = list(filter(lambda x: (x > 0.0), glove_tfidf))
      temp3 = []
      #print (len(glove_tfidf))
      for k in range(len(glove_tfidf)):
         temp3.append(glove_tfidf[k] * model[word_vectors[k]])
         ##final wrd2vec_tfidf vectors
      tfidf_vectors.append(np.mean(np.array(temp3), axis = 0))
      #tfidf_vectors(np.mean(temp3))
      #print word2vec_tfidf
      i = i + 1
   return tfidf_vectors

train = glove_tfidf(corpus_train)
test = glove_tfidf(corpus_test)
print "glove_tfidf"
bowscript.svm_classifier(train,data.posorneg_train,test)
bowscript.bayes_classifier(train,data.posorneg_train,test)
bowscript.logistic_classifier(train,data.posorneg_train,test)