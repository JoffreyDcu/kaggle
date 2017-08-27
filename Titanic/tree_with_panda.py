# -*- coding: utf-8 -*-

import pandas as pd
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

data = pd.read_csv('data/train.csv')

data['Age'].fillna(data['Age'].median(), inplace=True)
data.describe()

## Visualize Sex
survived_sex = data[data['Survived'] == 1]['Sex'].value_counts()
dead_sex = data[data['Survived'] == 0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex, dead_sex])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True)

## Visualize Age
plt.figure()
plt.hist([data[data['Survived'] == 1]['Age'], data[data['Survived'] == 0]['Age']], stacked=True, color = ['g','r'],
bins = 10, label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()

## Visualize Fare
plt.figure()
plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], stacked=True, color = ['g','r'],
bins = 10, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()

## Visualize Pclass
plt.figure()
plt.hist([data[data['Survived'] == 1]['Pclass'], data[data['Survived'] == 0]['Pclass']], stacked=True, color = ['g','r'],
bins = 10, label = ['Survived','Dead'])
plt.xlabel('Pclass')
plt.ylabel('Number of passengers')
plt.legend()

## Visualize Embarked
survived_embarked = data[data['Survived'] == 1]['Embarked'].value_counts()
dead_embarked = data[data['Survived'] == 0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embarked, dead_embarked])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True)


## Processing

## Merge datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
targets = train.Survived
train.drop('Survived', 1, inplace=True)
full = train.append(test)
full.reset_index(inplace=True)
full.drop('index', inplace=True, axis=1)

## Process Name, Age
def toRank(t):
    if t == 'Capt' or t == 'Col' or t == 'Maj':
        return 'Sdt'
    elif t == 'Jonkheer' or t == 'Don' or t == 'Sir' or t == 'Dona' or t == 'the Countess' or t == 'Lady':
        return 'Kng'
    elif t == 'Rev' or t == 'Dr':
        return 'Scf'
    else:
        return t
    
full['Title'] = full['Name'].map(lambda name : toRank(name.split(',')[1].split('.')[0].strip()))
full.drop('Name', axis=1, inplace=True)
grouped = full.groupby(['Sex','Pclass','Title']).median()

def fillAge(r):
    if np.isnan(grouped.loc[r['Sex'], r['Pclass'], r['Title']]['Age']):
        return grouped.loc[r['Sex'], r['Pclass']]['Age'].mean()
    else:
        return grouped.loc[r['Sex'], r['Pclass'], r['Title']]['Age']

full.Age = full.apply(lambda r : fillAge(r) if np.isnan(r['Age']) else r['Age'], axis=1)
title_dummies = pd.get_dummies(full['Title'], prefix='Title')
full = pd.concat([full, title_dummies], axis=1)
full.drop('Title', axis=1, inplace=True)

## Process Fare
full.Fare.fillna(full.Fare.mean(), inplace=True)

## Process Embarked
full.Embarked.fillna('S', inplace=True)
embarked_dummies = pd.get_dummies(full['Embarked'], prefix='Embarked')
full = pd.concat([full, embarked_dummies], axis=1)
full.drop('Embarked', axis=1, inplace=True)

## Process Cabin
full.Cabin.fillna('U', inplace=True)
full['Cabin'] = full['Cabin'].map(lambda c : c[0])
cabin_dummies = pd.get_dummies(full['Cabin'], prefix='Cabin')
full = pd.concat([full, cabin_dummies], axis=1)
full.drop('Cabin', axis=1, inplace=True)

## Process Sex
full['Sex'] = full['Sex'].map({'male':1,'female':0})

## Process Pclass
pclass_dummies = pd.get_dummies(full['Pclass'], prefix="Pclass")
full = pd.concat([full,pclass_dummies],axis=1)    
full.drop('Pclass',axis=1,inplace=True)

## Process Ticket
def prepareTicket(ticket):
    ticket = ticket.replace('.','').replace('/','').split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'
            
full['Ticket'] = full['Ticket'].map(lambda c : prepareTicket(c))
ticket_dummies = pd.get_dummies(full['Ticket'], prefix="Ticket")
full = pd.concat([full,ticket_dummies],axis=1)    
full.drop('Ticket',axis=1,inplace=True)

## Process Family
full['Family'] = full['SibSp'] + full['Parch'] + 1
full.drop('SibSp', axis=1, inplace=True)
full.drop('Parch', axis=1, inplace=True)

## Process PassengerId
full.drop('PassengerId', axis=1, inplace=True)


## Modelling

train = full.head(891)
test = full.iloc[891:]

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)

## Feature Selection
from sklearn.feature_selection import SelectFromModel

model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
test_reduced = model.transform(test)

## Tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

run_gs = False
if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [1, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)
    
## results
output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('data/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('data/submission.csv',index=False)