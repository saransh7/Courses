# logic for feature vector: using three words before sentence terminator; taking three words before punctuation that are not sentence terminator and taking three words(test_0) after the starting to train the classifiers(text_0plus) 

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import svm
import numpy as np
file =  open("text.txt")
txt = file.read()
txt = (re.sub(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z][a-z]\.)(?<![A-Z][a-z]\.)(?!')(?<=\.|\?)", "<\\s>", txt))
txt = (re.sub(r"(?<=\.'|\!'|\?')\n", "<\\s>\\n", txt))
txt = (re.sub(r"(?<=(<\\s>))(?!\Z)(\s+)","\n<s>",txt))
txt = '<s>' + txt[:-4]
array = re.findall(r"(?<=<s>)((.|\n)+?)(?=<\\s>)",txt)
array1 = [i[0] for i in array]
array2 = []
# filtering out newline character
for temp in array1:
    temp = re.sub(r"\n", " ", temp) 
    temp = re.sub(r"\s\s", "\n", temp)
    array2.append(temp)
corpus = array2
#print corpus
vectorizer = CountVectorizer()
vectorizer.fit_transform(corpus)
#print vectorizer.vocabulary_


regex_feature = re.compile("\w+")
feature_vectors_1 = []
for temp in corpus:
   feature_vectors_1.append((filter(regex_feature.match ,word_tokenize(temp)))[-3:])
test_vector_1 = []
for temp in feature_vectors_1:
     a = (vectorizer.transform(temp)).toarray()
     sum_a =  [sum(i) for i in zip(*a)]
     #print sum_a
     test_vector_1.append(sum_a)
 
feature_vectors_0 = []
feature_vectors_0plus = []
for temp in corpus:
   feature_vectors_0plus.append((filter(regex_feature.match ,word_tokenize(temp)))[:3])
test_vector_0plus = []
for temp in feature_vectors_0plus:
     a = (vectorizer.transform(temp)).toarray()
     sum_a =  [sum(i) for i in zip(*a)]
     #print sum_a
     test_vector_0plus.append(sum_a)
#######
re1 = re.compile(r".{4,20}(?=\.|\?|\!|\.'|\?'|\!')", re.DOTALL)
for temp in corpus:
#      print temp
      array  = re1.findall( temp[:-3])
#      print array
      for temp_2 in array:
          feature_vectors_0.append((filter(regex_feature.match ,word_tokenize(temp_2)))[-3:])

test_vector_0 = []
for temp in feature_vectors_0:
     a = (vectorizer.transform(temp)).toarray()
     sum_a =  [sum(i) for i in zip(*a)]
     #print sum_a
     test_vector_0.append(sum_a)

##########training
test_vector_1_ = [x for x in test_vector_1 if x != []]
test_vector_0_ = [x for x in test_vector_0 if x != []]
test_vector_0plus_ = [x for x in test_vector_0plus if x != []]

len_1 = len(test_vector_1_)
len_0 = len(test_vector_0_)
len_0plus = len(test_vector_0plus_)

n_1 = (len_1 * 2)/3
n_0 = len_0 - (len_0)/10
n_0plus = n_0
array_1 = []

for i in range(0, n_1):
     array_1.append((np.asarray(test_vector_1_[i])))

for i in range(0, n_0):
     array_1.append((np.asarray(test_vector_0_[i])))

for i in range(0, n_0plus):
     array_1.append((np.asarray(test_vector_0plus_[i])))


Y_1 = []
Y_0 = []
for i in range(0,len_1):
    Y_1.append(1)

for i in range(0, len_0 * 2):
    Y_0.append(0)

y = np.asarray((Y_1[:n_1]) + (Y_0[:(n_0 + n_0plus)]))
X = np.asarray(array_1)

#print X
#print y
clf = svm.SVC(gamma = 0.001, C = 40 , class_weight = 'balanced')
clf.fit(X,y)
test_1 = np.asarray(test_vector_1_[(len_1 * 2)/3 : len_1])
test_0 = np.asarray(test_vector_0_[-((len_0)/10) : ])
result_1_1 = clf.predict(test_1)
result_1_0 = clf.predict(test_0)
print result_1_1 #true should be a vector of 1
print result_1_0 #true should be a vector of 0
efficiency1 = 100*(sum(result_1_1) + len(test_0) - sum(result_1_0))/(len(test_0) + len(test_1))

print "efficiency for svm is %d"% efficiency1
logreg = linear_model.LogisticRegression(C=10)
logreg.fit(X,y)
result_2_1 = logreg.predict(test_1)
result_2_0 = logreg.predict(test_0)
print result_2_1 #true should be a vector of 1
print result_2_0 #true should be a vector of 0
efficiency2 = 100*(sum(result_2_1) + len(test_0) - sum(result_2_0))/(len(test_0) + len(test_1))
print "efficiency for logistic is %d"% efficiency2
