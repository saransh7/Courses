import csv
import os
import numpy as np
import nltk
import string
import re
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import glob
train_path = "/home/saransh/Desktop/nlp assignment 2/data/train/"
test_path = "/home/saransh/Desktop/nlp assignment 2/data/test/"

train_pos_file=glob.glob(train_path +  'pos/*.txt')
train_neg_file=glob.glob(train_path + "/neg/*.txt")
test_pos_file=glob.glob(test_path +  'pos/*.txt')
test_neg_file=glob.glob(train_path + "/neg/*.txt")

def preprocess(file):
    for i in range(len(file)):
        file[i]=re.sub("(\.|'|\"|\?|!|,|<br \/>|<br>)"," ",file[i])
    return file

def list_maker(file):
    new_list=[]
    for i in range(4):
        f=open(file[i],'rb')
        new_list.append(f.read())
        f.close()
    return new_list
#using list maker function for making the list of documents
train_pos=list_maker(train_pos_file)
train_neg=list_maker(train_neg_file)
test_pos=list_maker(test_pos_file)
test_neg=list_maker(test_neg_file)
#converting byte type into string and appending them
train_pos1=[]
train_neg1=[]
test_neg1 = []
test_pos1 = []
posorneg_test = []
posorneg_train = []

for i in range(len(train_pos)):
    train_pos1.append(train_pos[i].decode("utf-8"))
    test_pos1.append(test_pos[i].decode("utf-8"))
    posorneg_train.append(1) 

for i in range(len(train_neg)):
    train_neg1.append(train_neg[i].decode("utf-8"))
    test_neg1.append(test_neg[i].decode("utf-8"))
    posorneg_train.append(0)

posorneg_test = posorneg_train

train_pos1=preprocess(train_pos1)
train_neg1=preprocess(train_neg1)
test_pos1=preprocess(test_pos1)
test_neg1=preprocess(test_neg1)

text_train = np.concatenate((train_pos1, train_neg1), axis = 0)
index_train = [i for i in range(8)]
text_test = np.concatenate((test_pos1, test_neg1), axis = 0)
index_test = [i for i in range(8)]

vectorizer = CountVectorizer(stop_words= 'english')
tokenizer = vectorizer.build_tokenizer()

def define_corpus(text):
   corpus = []
   for sentence in text:
     #print sentence
     tokens = [ token.lower() for token in tokenizer(sentence)]
     #print tokens
     corpus.append(tokens)
     docs = [" ".join(corpora) for corpora in corpus]
   return (corpus, docs)


'''
def load_data(path):
    index = []
    text = []
    posorneg = []
    i = 0
    for filename in os.listdir(path + "pos"):
       if i < temp/2:
         data = open(path+"pos/"+filename, 'rb').read()
         index.append(i)
         text.append(data)
         posorneg.append(1)
         #token_dict.append(data.lower().translate(None, string.punctuation)) 
         i = i + 1
         #print data

    for filename in os.listdir(path + "neg"):
       if i < temp: 
        data = open(path+"neg/"+filename, 'rb').read()
        index.append(i)
        text.append(data)
        posorneg.append(0)
        #token_dict.append(data.lower().translate(None, string.punctuation)) 
        i = i + 1
    return (text, posorneg, index)

text_train, posorneg_train, index_train = load_data(train_path)
text_test, posorneg_test, index_test = load_data(test_path)
'''