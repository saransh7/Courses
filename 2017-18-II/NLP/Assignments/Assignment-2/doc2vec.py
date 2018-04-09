import preprocessing as data
import BOW_tfidf_norm as bowscript
import csv
import os
import numpy as np
import nltk
import string
import gensim
from keras.models import Sequential
from keras.layers import Dense,  Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras import regularizers

corpus_train, docs_train = data.define_corpus(data.text_train)
corpus_test, docs_test = data.define_corpus(data.text_test)

#doc2vec
model = gensim.models.Doc2Vec(size=100,window = 20, min_count=0, alpha=0.025, min_alpha=0.025)

class LabeledLineSentence(object):

    def __init__(self, doc_list, labels_list):

        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,[self.labels_list[idx]])
index = []
temp = 0
for corpus in corpus_train:
    index.append(temp)
    temp = temp + 1

it_train = LabeledLineSentence(corpus_train, index)                

model.build_vocab(it_train)
#training of model
examples = len(corpus_train) 
for epoch in range(4):
 print 'iteration' + str(epoch+1)
 model.train(it_train, total_examples = examples, epochs=4)
 model.alpha -= 0.002
 model.min_alpha = model.alpha

test = []
train = []
i = 0
for temp in corpus_train: 
       train.append(model.docvecs[i])
       i = i +1

for temp in corpus_test: 
       test.append(model.infer_vector(temp))
print "doc2vec--------------------------------"
bowscript.svm_classifier(train,data.posorneg_train,test)
bowscript.bayes_classifier(train,data.posorneg_train,test)
bowscript.logistic_classifier(train,data.posorneg_train,test)
bowscript.mlp_classifier(train,data.posorneg_train,test)


model = Sequential()
model.add(LSTM(
        100,
        input_shape= (50, 300),
        dropout=0.2,
        recurrent_dropout=0.2
    ))

model.add(Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=regularizers.l2(0.5)
    ))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

X_test = np.array([temp.tolist() for temp in test])
X_train = np.array([temp.tolist() for temp in train])

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

#model.fit(train,data.posorneg_train)