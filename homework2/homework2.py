import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re


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

class model(object):
 
    def count_words(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts
    
    def fit(self, X, Y):
        self.email_count = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()
        n = len(X)
        self.email_count['spam'] = sum(1 for label in Y if label == 'spam')
        self.email_count['ham'] = sum(1 for label in Y if label == 'ham')
        self.log_class_priors['spam'] = np.log(self.email_count['spam'] / n) 
        self.log_class_priors['ham'] = np.log(self.email_count['ham'] / n)
        self.word_counts['spam'] = {}
        self.word_counts['ham'] = {}
        
        for x, y in zip(X, Y):
            c = 'spam' if y =='spam' else 'ham'
            counts = self.count_words(re.split("\W+", x))
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0
                    self.word_counts[c][word] += count       
                        
    def predict(self, X,Y):
        acc=0
        score=0
        total = len(Y)
        for x,y in zip(X,Y):
            counts = self.count_words(re.split("\W+", x))
            spam_score = 0
            ham_score = 0
            for word, _ in counts.items():
                if word not in self.vocab: continue
                log_w_by_spam = np.log( (self.word_counts['spam'].get(word, 0.0) + 1) / (self.email_count['spam'] + len(self.vocab)) )
                log_w_by_ham = np.log( (self.word_counts['ham'].get(word, 0.0) + 1) / (self.email_count['ham'] + len(self.vocab)) )
                spam_score += log_w_by_spam
                ham_score += log_w_by_ham
            spam_score = spam_score + self.log_class_priors['spam']
            ham_score = ham_score + self.log_class_priors['ham']
            if spam_score > ham_score and y=='spam':
                score = score + 1
            if spam_score<= ham_score and y=='ham':
                score = score + 1
        acc = (score / total)*100
        return acc
        



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


NB = model()
NB.fit(preprocessed_reviews1, data1[1])
print("Accuracy on training data with stop words is : ")
print(NB.predict(preprocessed_reviews1,data1[1]))
print("Accuracy on testing data with stop words is : ")
print(NB.predict(preprocessed_reviews2,data2[1]))
print("Training with stop words on train data and testing on test data without stop words  : Accuracy is : ")
print(NB.predict(preprocessed_reviews4,data2[1]))
print("Training without stop words on train data and testing without stop words on test data: Accuracy is :")
NB=model()
NB.fit(preprocessed_reviews3,data1[1])
print(NB.predict(preprocessed_reviews4,data2[1]))
