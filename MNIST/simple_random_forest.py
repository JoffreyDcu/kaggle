# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_csv = pd.read_csv('train.csv', header=0)
X = file_csv.iloc[:, 1:]
y = file_csv.iloc[:, 0]

image = X.iloc[5].as_matrix().reshape((28,28))
plt.imshow(image, 'gray_r')
plt.title('Image 5')

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing

binarizer = preprocessing.Binarizer()
X_binarized = binarizer.transform(X)
X = pd.DataFrame(X_binarized)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=8)

from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#
#classifiers = dict() 
#classifiers['Gaussian Naive Bayes'] = GaussianNB()
#classifiers['Decision Tree Classifier'] = DecisionTreeClassifier(random_state=8)
#classifiers['Random Forests'] = RandomForestClassifier(max_depth=2, random_state=0)
#classifiers['SVC'] = SVC()
#
#for clf_name, clf in classifiers.items():
#    clf.fit(X_train, y_train)
#    score = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy').mean()
#    print(clf_name, score)
    
clf = RandomForestClassifier(n_jobs=1, n_estimators=500, max_features='auto',random_state=8)
clf.fit(X_binarized, y)
test_data = pd.read_csv('test.csv')
test_binarized = binarizer.transform(test_data)
results = clf.predict(test_binarized[:])

df = pd.DataFrame(results)
df.index += 1
df.index.names = ['ImageId']
df.columns = ['Label']
df.to_csv('submission.csv', header=True)