#!/usr/bin/env python3
#I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else. I understand that if I am involved in plagiarism or cheating I will have to sign an official form that I have cheated and that this form will be stored in my official university record. I also understand that I will receive a grade of 0 for the involved assignment for my first offense and that I will receive a grade of “F” for the course for any additional offense.
#Shubham Patwa
import pandas as pd
import numpy as np
from pprint import pprint
import sys


if(len(sys.argv)!=6):
    print("Arguments wrong or missing. Please check again")
else:
    train_data = pd.read_csv(sys.argv[1])
    cv_data = pd.read_csv(sys.argv[2])
    test_data = pd.read_csv(sys.argv[3])
    heu = (sys.argv[5])
    printf=sys.argv[4]


def VI(attribute):
    value = 1
    vals,counts= np.unique(attribute,return_counts=True)
    for i in range(len(vals)):
        value *= counts[i]/np.sum(counts)
    return value

def entropy(att):
    entropy = 0
    elements,counts = np.unique(att,return_counts = True)
    for i in range(len(elements)):
        entropy += np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts))])
    return entropy

def InfoGain(data,att):
    total_entropy = entropy(data['Class'])
    vals,counts= np.unique(data[att],return_counts=True)
    Weighted_Entropy = 0
    for i in range(len(vals)):
        Weighted_Entropy += np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[att]==vals[i]).dropna()['Class'])])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def VarImp(data,att):
    total_imp = VI(data['Class'])
    vals,counts= np.unique(data[att],return_counts=True)
    Weighted_Imp = 0
    for i in range(len(vals)):
        Weighted_Imp += np.sum([(counts[i]/np.sum(counts))*VI(data.where(data[att]==vals[i]).dropna()['Class'])])
    Information_Gain = total_imp - Weighted_Imp
    return Information_Gain



def buildtree(data,originaldata,features,parent_node_class = None):
    if len(np.unique(data.Class)) <= 1:
        return np.unique(data.Class)[0]
    elif len(data)==0:
        return np.unique(originaldata.Class)[np.argmax(np.unique(originaldata.Class,return_counts=True)[1])]
    elif len(features) ==0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data.Class)[np.argmax(np.unique(data.Class,return_counts=True)[1])]
        if (heu=="1"):
            item_values = [InfoGain(data,feature) for feature in features] 
        if (heu=="2"):
            item_values = [VarImp(data,feature) for feature in features] 
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature:{}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = buildtree(sub_data,data,features,parent_node_class)
            tree[best_feature][value] = subtree  
        return(tree)    




            
def predict(inst,tree):
    for nodes in tree.keys():  
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0
        try:
            if type(tree) is dict:
                prediction = predict(inst, tree)
            else:
                prediction = tree
                break;  
        except:
            prediction =1
    return prediction            

def test(data,tree): 
    sum = np.sum([(data.Class[i] ==predict(data.iloc[i],tree)) for i in range(len(data))])
    print('The prediction accuracy is: ',((sum)/len(data))*100,'%')
    


tree = buildtree(train_data,train_data,train_data.columns[:-1])
if(printf=="yes"):
    pprint(tree)
print("ON TRAIN DATA :")
test(train_data,tree)
print("ON VALIDATION DATA :")
test(cv_data,tree)
print("ON TEST DATA : ")
test(test_data,tree)
#if(heu=="2"):
#    tree = ID3Infogain(train_data,train_data,train_data.columns[:-1])
#    if(printf=="yes"):
#        pprint(tree)
#    print("ON TRAIN DATA :")
#    test(train_data,tree)
#    print("ON VALIDATION DATA ")
#    test(cv_data,tree)
#    print("ON TEST DATA : ")
#    test(test_data,tree)
    
#tree1 = ID3VI(train_data,train_data,train_data.columns[:-1])
#test1 = test(cv_data,tree)
#print("done")






