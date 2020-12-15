# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:07:22 2020

@author: shubh
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from sklearn.feature_extraction.text import CountVectorizer
import sys
import warnings
warnings.filterwarnings("ignore")


if(len(sys.argv)!=3):
    print("Arguments wrong or missing. Please check again")
else:
    n_iter = int(sys.argv[1])
    r=int(sys.argv[2])



stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])


def read_data(folder):
    data_list = []
    file_data = os.listdir(folder)
    for data_file in file_data:
        f = open(folder + data_file, 'r',errors='ignore')
        data_list.append(f.read())
    f.close()
    return data_list
#read data from dataset$##########
    #train data#########
spam = read_data('train/spam/')
spams =[(email, 'spam') for email in spam]
ham = read_data('train/ham/')
hams = ([(email, 'ham') for email in ham])
data1 = pd.DataFrame(hams+spams)

########test data

spam = read_data('test/spam/')
spams =[(email, 'spam') for email in spam]
ham = read_data('test/ham/')
hams = ([(email, 'ham') for email in ham])
data2 = pd.DataFrame(hams+spams)
###########################
def decontracted(phrase):
# specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
# general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
#Pre process the data
preprocessed_reviews1 = []
for sentance in tqdm(data1[0].values):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    preprocessed_reviews1.append(sentance.strip())

preprocessed_reviews2 = []
for sentance in tqdm(data2[0].values):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    preprocessed_reviews2.append(sentance.strip())
    
preprocessed_reviews3 = []
for sentance in tqdm(preprocessed_reviews1):
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_reviews3.append(sentance.strip())
preprocessed_reviews4 = []
for sentance in tqdm(preprocessed_reviews2):
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_reviews4.append(sentance.strip())     



vectorizer = CountVectorizer()
vectorizer.fit(preprocessed_reviews1)
vector_train = vectorizer.transform(preprocessed_reviews1)
vector_test = vectorizer.transform(preprocessed_reviews2)
vector_wostop = vectorizer.transform(preprocessed_reviews4)


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))
Y_train = data1[1]
Y_train[Y_train == 'spam'] = 1
Y_train[Y_train == 'ham'] = 0
Y_train = pd.DataFrame(Y_train)
Y_train = Y_train.values

Y_test = data2[1]
Y_test[Y_test == 'spam'] = 1
Y_test[Y_test == 'ham'] = 0
Y_test = pd.DataFrame(Y_test)
Y_test = Y_test.values

W= np.zeros(9162)
W = np.random.normal(0,1,size = W.size)
W=W.reshape(9162,1)
b=0
t=1
x =vector_train.todense()
y= Y_train
for i in range(n_iter):
    dotp = (np.dot(x,W)+b)
    dotp[dotp > 0] = 1
    dotp[dotp <= 0] = -1
    deltaw = np.dot(x.T,(y - dotp))
    deltaw *= 1/x.shape[0]
    deltaw = r*deltaw
    W = W + deltaw
    bias1 = (y- dotp)
    bias1 = np.mean(bias1)
    bias1 *= 1/x.shape[0]
    bias1 = r*bias1   
    b = b + bias1
    
y_pred_train = (np.array(np.dot(vector_train.todense(),W)+b))
y_pred_test = (np.array(np.dot(vector_test.todense(),W)+b))
y_pred_wostop = (np.array(np.dot(vector_wostop.todense(),W)+b))


for i in range(len(Y_train)):
    y_pred_train[i] = sigmoid(np.sum(y_pred_train[i]))
for i in range(len(Y_test)):
    y_pred_test[i] = sigmoid(np.sum(y_pred_test[i]))
    y_pred_wostop[i] = sigmoid(np.sum(y_pred_wostop[i]))
    
y_pred_train[y_pred_train >0.5] = 1
y_pred_train[y_pred_train <=0.5 ] = 0
y_pred_test[y_pred_test >0.5] = 1
y_pred_test[y_pred_test <=0.5 ] = 0
y_pred_wostop[y_pred_wostop >0.5] = 1
y_pred_wostop[y_pred_wostop <=0.5 ] = 0

count=0
for i in range(len(y_pred_train)):
    if y_pred_train[i] == Y_train[i]:
        count += 1
    train_acc = ((count/len(y_pred_train))*100)
count=0
for i in range(len(y_pred_test)):
    if y_pred_test[i] == Y_test[i]:
        count += 1
    test_acc = ((count/len(y_pred_test))*100)
count=0
for i in range(len(y_pred_wostop)):
    if y_pred_wostop[i] == Y_test[i]:
        count += 1
    wostop_acc = ((count/len(y_pred_test))*100)
    
    
print("=============================================================")
print("Iterations: {} , Learning Rate : {} ".format(n_iter,r))    
print("Accuracy on Train Data :{}%  ".format(train_acc))
print("Accuracy on Test data : {}%".format(test_acc))
print("Accuracy on test data w/o stop words: {}%".format(wostop_acc))
print("=============================================================")


