
# coding: utf-8

# ## Loading packages

# In[192]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer

pd.set_option('display.max_columns', 100)


# ## Loading data

# In[193]:

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# <a class="anchor" id="visual_inspection"></a>

# ## Data at first sight

# Here is an excerpt of the the data description for the competition:
# * Features that belong to **similar groupings are tagged** as such in the feature names (e.g.,  ind, reg, car, calc).
# * Feature names include the postfix **bin** to indicate binary features and **cat** to  indicate categorical features. 
# * Features **without these designations are either continuous or ordinal**. 
# * Values of **-1**  indicate that the feature was **missing** from the observation. 
# * The **target** columns signifies whether or not a claim was filed for that policy holder.
# 

# In[194]:

train.head()


# We indeed see the following
# * binary variables
# * categorical variables of which the category values are integers
# * other variables with integer or float values
# * variables with -1 representing missing values
# * the target variable and an ID variable

# Let's look at the number of rows and columns in the train data.

# In[195]:

train.shape


# We have 59 variables and 595.212 rows. Let's see if we have the same number of variables in the test data.<br>
# Let's see if there are duplicate rows in the training data.

# In[196]:

train.drop_duplicates()
train.shape


# No duplicate rows, so that's fine.

# In[197]:

test.shape


# We are missing one variable in the test set, but this is the target variable. So that's fine.<br>
# Let's now invesigate how many variables of each type we have.

# So later on we can create dummy variables for the 14 categorical variables. The *bin* variables are already binary and do not need dummification.

# In[198]:

train.info()


# Again, with the info() method we see that the data type is integer or float. No null values are present in the data set. That's normal because missing values are replaced by -1. We'll look into that later.

# <a class="anchor" id="metadata"></a>

# ## Metadata
# To facilitate the data management, we'll store meta-information about the variables in a DataFrame. This will be helpful when we want to select specific variables for analysis, visualization, modeling, ...
# 
# Concretely we will store:
# - **role**: input, ID, target
# - **level**: nominal, interval, ordinal, binary
# - **keep**: True or False
# - **dtype**: int, float, str

# In[199]:

data = []
for f in train.columns:
    # Defining the role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
         
    # Defining the level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train[f].dtype == 'float64':
        level = 'interval'
    elif train[f].dtype == 'int64':
        level = 'ordinal'
        
    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'id':
        keep = False
    
    # Defining the data type 
    dtype = train[f].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
    
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)


# In[200]:

meta


# Example to extract all nominal variables that are not dropped

# In[201]:

meta[(meta.level == 'nominal') & (meta.keep)].index


# Below the number of variables per role and level are displayed. 

# In[202]:

pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()


# <a class="anchor" id="descriptive_stats"></a>

# ## Descriptive statistics

# We can also apply the *describe* method on the dataframe. However, it doesn't make much sense to calculate the mean, std, ... on categorical variables and the id variable. We'll explore the categorical variables visually later.
# 
# Thanks to our meta file we can easily select the variables on which we want to compute the descriptive statistics. To keep things clear, we'll do this per data type.

# ### Interval variables

# In[203]:

v = meta[(meta.level == 'interval') & (meta.keep)].index
train[v].describe()


# #### reg variables
# - only ps_reg_03 has missing values
# - the range (min to max) differs between the variables. We could apply scaling (e.g. StandardScaler), but it depends on the classifier we will want to use.
# 
# #### car variables
# - ps_car_12 and ps_car_14 have missing values
# - again, the range differs and we could apply scaling.
# 
# #### calc variables
# - no missing values
# - this seems to be some kind of ratio as the maximum is 0.9
# - all three *_calc* variables have very similar distributions
# 
# 
# **Overall**, we can see that the range of the interval variables is rather small. Perhaps some transformation (e.g. log) is already applied in order to anonymize the data?
# 

# ### Ordinal variables

# In[204]:

v = meta[(meta.level == 'ordinal') & (meta.keep)].index
train[v].describe()


# - Only one missing variable: ps_car_11
# - We could apply scaling to deal with the different ranges

# ### Binary variables

# In[205]:

v = meta[(meta.level == 'binary') & (meta.keep)].index
train[v].describe()


# - A priori in the train data is 3.645%, which is **strongly imbalanced**. 
# - From the means we can conclude that for most variables the value is zero in most cases.


# <a class="anchor" id="data_quality"></a>

# ## Data Quality Checks

# ### Checking missing values
# Missings are represented as -1

# In[207]:

vars_with_missing = []

for f in train.columns:
    missings = train[train[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/train.shape[0]
        
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
        
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))


# - **ps_car_03_cat and ps_car_05_cat** have a large proportion of  records with missing values. We could remove these variables or keep them as a new category. For now, we remove them.
# - For the other categorical variables with missing values, we can leave the missing value -1 as such.
# - **ps_reg_03** (continuous) has missing values for 18% of all records. What should we do?
# - **ps_car_11** (ordinal) has only 1 record with missing values. Remove it.
# - **ps_car_14** (continuous) has missing values for 7% of all records. What should we do?
# 
# We obtain roughly the same proportion of missing values in the test dataset.

# In[208]:

# Dropping the variables with too many missing values
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)
meta.loc[(vars_to_drop),'keep'] = False  # Updating the meta

# Dropping the record with missing ps_car_11
train = train[train.ps_car_11 != -1]

# Imputing with the mean or mode
mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()


# In[209]:

# Dropping the variables with too many missing values
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
test.drop(vars_to_drop, inplace=True, axis=1)

# Imputing with the mean or mode
test['ps_reg_03'] = mean_imp.fit_transform(test[['ps_reg_03']]).ravel()
test['ps_car_14'] = mean_imp.fit_transform(test[['ps_car_14']]).ravel()


# In[210]:

train.describe()


# ## Exploratory Data Visualization

# ### Categorical variables

# As we can see from the variables **with missing values**,  it is a good idea to keep the missing values as a separate category value, instead of replacing them by the mode for instance. The customers with a missing value appear to have a much higher (in some cases much lower) probability to ask for an insurance claim.



# For the ordinal variables we do not see many correlations. We could, on the other hand, look at how the distributions are when grouping by the target value.

# <a class="anchor" id="feat_engineering"></a>

# ## Feature engineering

# ### Creating dummy variables

# In[219]:

v = meta[(meta.level == 'nominal') & (meta.keep)].index
print('Before dummification we have {} variables in train'.format(train.shape[1]))
train = pd.get_dummies(train, columns=v, drop_first=True)
print('After dummification we have {} variables in train'.format(train.shape[1]))


# In[220]:

print('Before dummification we have {} variables in test'.format(test.shape[1]))
test = pd.get_dummies(test, columns=v, drop_first=True)
print('After dummification we have {} variables in test'.format(test.shape[1]))


# So, creating dummy variables adds 52 variables to the training set.

# ### Creating interaction variables

# In[221]:

v = meta[(meta.level == 'interval') & (meta.keep)].index
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
interactions = pd.DataFrame(data=poly.fit_transform(train[v]), columns=poly.get_feature_names(v))
# Merge the interaction variables to the train data
print('Before creating interactions we have {} variables in train'.format(train.shape[1]))
train = pd.concat([train, pd.DataFrame(data=interactions, columns=poly.get_feature_names(v))], axis=1)
print('After creating interactions we have {} variables in train'.format(train.shape[1]))


# In[222]:

interactions = pd.DataFrame(data=poly.fit_transform(test[v]), columns=poly.get_feature_names(v))
# Merge the interaction variables to the train data
print('Before creating interactions we have {} variables in test'.format(test.shape[1]))
test = pd.concat([test, pd.DataFrame(data=interactions, columns=poly.get_feature_names(v))], axis=1)
print('After creating interactions we have {} variables in test'.format(test.shape[1]))


# This adds extra interaction variables to the train data. Thanks to the *get_feature_names* method we can assign column names to these 
# new variables.

# <a class="anchor" id="feat_selection"></a>

# ## Feature selection

# ### Selecting features with a PCA
# 

# Cleaning nan

train.dropna(inplace=True)

def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

gini_scorer = make_scorer(normalized_gini, greater_is_better = True)
# In[228]:

X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', PCA()),
 #   ('classify', LinearSVC(class_weight={1: 20}))
 #   ('classify', SVC(kernel = 'rbf', probability=True, class_weight={1: 10}))
    ('classify', AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, criterion='gini'), algorithm="SAMME", n_estimators=50))
 #   ('classify', AdaBoostClassifier(LinearSVC(), algorithm="SAMME", n_estimators=50))
 #   ('classify', DecisionTreeClassifier(criterion = 'gini'))
])

N_FEATURES_OPTIONS = [20, 30,50]
TREE_DEPTH =[5,10,20,30]
#C_OPTIONS = [1,0.1,0.01,0.001]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
#       'classify__C': C_OPTIONS
#        'classify__max_depth': TREE_DEPTH
    },
]
reducer_labels = ['PCA']

grid = GridSearchCV(pipe, cv=3, n_jobs=-1, param_grid=param_grid, verbose=10)
grid.fit(X_train, y_train)

#mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
#mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
#bestN = mean_scores.argmax(axis=0)
#mean_scores = mean_scores.max(axis=0)


# In[ ]:

#Testing linearSVC I use CalibratedClassifier to get the probas. SVC(kernel=linear) is very slow. Don't use it

pipe2 = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', PCA(n_components=20)),
    ('classify', CalibratedClassifierCV(AdaBoostClassifier(LinearSVC(C=1),algorithm='SAMME'), method='sigmoid', cv=3))])
pipe2.fit(train.drop(['id', 'target'], axis=1)[:50000], train['target'][:50000])
output = pipe2.predict_proba(test.drop(['id'], axis=1))[:,1]
df_output = pd.DataFrame()
df_output['id'] = test['id']
df_output['target'] = output
df_output[['id','target']].to_csv('submission.csv',index=False, sep=",")

df_output['target'].head(200)


# In[ ]:

plt.figure()
plt.plot(N_FEATURES_OPTIONS,mean_scores[0])
plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.ylabel('Digit classification accuracy')
plt.ylim((0, 1))
plt.show()


# In[ ]:

output = grid.best_estimator_.predict_proba(test.drop(['id'], axis=1))
df_output = pd.DataFrame()
df_output['id'] = test['id']
df_output['target'] = output[:,1]
df_output[['id','target']].to_csv('submission.csv',index=False, sep=",")


# In[ ]:

df_output['target'].head(200)

