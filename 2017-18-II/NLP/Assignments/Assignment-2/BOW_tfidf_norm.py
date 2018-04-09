import preprocessing as data
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
corpus_train, docs_train = data.define_corpus(data.text_train)
corpus_test, docs_test = data.define_corpus(data.text_test)
#print docs_train
bow_train = data.vectorizer.fit_transform(docs_train)
bow_test = data.vectorizer.transform(docs_test)
#bow_normalized
def bow_normalized(bow):
   bow_vector = bow.astype(float)
   bow_vector = bow_vector.toarray()
   bow_n = []
   for temp in bow_vector:
     div = sum(temp)
     bow_n.append([x / div for x in temp])
   return bow_n
bow_n_train =  bow_normalized(bow_train)
bow_n_test =  bow_normalized(bow_test)
#bow_tfidf

def tf_idf(bow):
  count = bow.toarray()
  transformer = TfidfTransformer(smooth_idf=False)
  tfidf = transformer.fit_transform(count)
  return tfidf.toarray()

tfidf_train =  tf_idf(bow_train)
tfidf_test =  tf_idf(bow_test)

#svm
from sklearn import svm

def svm_classifier(X,y,test_data):
 print "svm"
 model = svm.SVC()
 model.fit(X,y)
 y = model.predict(test_data)
 #print y
 print accuracy_score(y,data.posorneg_test)
 


#logistic
from sklearn import linear_model

def logistic_classifier(X,y,test_data):
    print "logistic"
    model = linear_model.LogisticRegression()
    model.fit(X,y)
    y = model.predict(test_data)
    #print y
    print accuracy_score(y,data.posorneg_test)
    


#naivebayes
from sklearn import naive_bayes
 
def bayes_classifier(X,y,test_data):
    print "bayesclassifier"
    model = naive_bayes.GaussianNB()
    model.fit(X,y)
    y = model.predict(test_data)
    #print y
    print accuracy_score(y,data.posorneg_test)

def mlp_classifier(X,y,test_data):
    print "mlpclassifier"
    model = MLPClassifier(hidden_layer_sizes=(100, 200), activation='relu', solver='adam')
    model.fit(X,y)
    y = model.predict(test_data)
    #print y
    print accuracy_score(y,data.posorneg_test)    

print "bow---------------------"
svm_classifier(bow_train.toarray(),data.posorneg_train,bow_test.toarray())
logistic_classifier(bow_train.toarray(),data.posorneg_train,bow_test.toarray())
bayes_classifier(bow_train.toarray(),data.posorneg_train,bow_test.toarray())

print "bow_n------------------------"
svm_classifier(bow_n_train,data.posorneg_train,bow_n_test)
bayes_classifier(bow_n_train,data.posorneg_train,bow_n_test)
logistic_classifier(bow_n_train,data.posorneg_train,bow_n_test)

print "tfidf-------------------------"
logistic_classifier(tfidf_train,data.posorneg_train,tfidf_test)
bayes_classifier(tfidf_train,data.posorneg_train,tfidf_test)
svm_classifier(tfidf_train,data.posorneg_train,tfidf_test)
#print tfidf_test