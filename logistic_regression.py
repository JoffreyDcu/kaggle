# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 20:42:11 2017

@author: Joffrey
"""
import csv
import numpy as np
from sklearn import linear_model

skip = True
train_file = open('data/train.csv')
csv_file = csv.reader(train_file)
    
Pclass,gender,age,sibSP,parch,fare,cabin,embarked = ([] for i in range(8))
for row in csv_file:
    if (skip == True):
        skip = False
        continue
    Pclass.append(row[0]) 
    gender.append(row[2]) 
    age.append(row[3]) 
    sibSP.append(row[4]) 
    parch.append(row[5]) 
    fare.append(row[7]) 
    cabin.append(row[8]) 
    embarked.append(row[9])
        
def scale(l):
    x = [float(k) for k in l]
    mean = sum(x)/float(len(x))
    mrange = max(x) - min(x)
    x = [(k-mean)/mrange for k in x]
    return x
    
def prepareG(gender):
    x = []
    for k in gender:
        if (k == "male"):
            x.append(.5)
        else:
            x.append(-.5)
    return x
    
def prepareA(age):
    x = [float(k) for k in age if (k!="")]
    mean = sum(x)/float(len(x))
    mrange = max(x) - min(x)
    y = []
    for i in range(0,len(age)):
        if (age[i] == ""):
            y.append(mean)
        else:
            y.append((float(age[i])-mean)/mrange)
    return y
    
def prepareC(cabin):
    x = []
    for k in cabin:
        if (k == ""):
            x.append(.5)
        else:
            x.append(-.5)
    return x
    
    
def prepareE(embarked):
    x = []
    for k in embarked:
        if (k=="S"):
            x.append(.5)
        elif (k=="Q"):
            x.append(0)
        else:
            x.append(-.5)
    return x
    
skip = True
train_file = open('data/train.csv')
csv_file = csv.reader(train_file)
    
Pclass,gender,age,sibSP,parch,fare,cabin,embarked,y_train = ([] for i in range(9))
for row in csv_file:
    if (skip == True):
        skip = False
        continue
    Pclass.append(row[2]) 
    gender.append(row[4]) 
    age.append(row[5]) 
    sibSP.append(row[6]) 
    parch.append(row[7]) 
    fare.append(row[9]) 
    cabin.append(row[10]) 
    embarked.append(row[11])
    y_train.append(row[1])
    
Pclass = scale(Pclass)
gender = prepareG(gender)
age = prepareA(age)
sibSP = scale(sibSP)
parch = scale(parch)
fare = prepareA(fare)
cabin = prepareC(cabin)
embarked = prepareE(embarked)


x_train = []
for x in range(0,891):
	x_train.append([Pclass[x],gender[x],age[x],sibSP[x],
		parch[x],fare[x],cabin[x],embarked[x]])

skip = True
train_file = open('data/test.csv')
csv_file = csv.reader(train_file)
    
Pclass,gender,age,sibSP,parch,fare,cabin,embarked = ([] for i in range(8))
for row in csv_file:
    if (skip == True):
        skip = False
        continue
    Pclass.append(row[1]) 
    gender.append(row[3]) 
    age.append(row[4]) 
    sibSP.append(row[5]) 
    parch.append(row[6]) 
    fare.append(row[8]) 
    cabin.append(row[9]) 
    embarked.append(row[10])
    
Pclass = scale(Pclass)
gender = prepareG(gender)
age = prepareA(age)
sibSP = scale(sibSP)
parch = scale(parch)
fare = prepareA(fare)
cabin = prepareC(cabin)
embarked = prepareE(embarked)

x_test = []
for x in range(0,418):
	x_test.append([Pclass[x],gender[x],age[x],sibSP[x],
		parch[x],fare[x],cabin[x],embarked[x]])
 
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)

clf = linear_model.LogisticRegression(C=1e5)
clf = clf.fit(x_train, y_train)
results = np.ones((len(x_test),2))
i = len(x_train) + 1

for x in range(0,len(x_test)):
	print (clf.predict(x_test[x].reshape(1, -1)))
	results[x,1] = (clf.predict(x_test[x].reshape(1, -1)))[0]
	results[x,0] = i
	i = i + 1

np.savetxt(open('data/submission.csv','ab'), results, delimiter=',', fmt = '%i') 