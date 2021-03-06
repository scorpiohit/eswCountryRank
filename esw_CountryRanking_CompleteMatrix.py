# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:42:38 2018

@author: msharma
"""

import matplotlib.pyplot as plt
import sys
import math
import pickle
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectKBest,chi2
import pandas as pd
import numpy as np
import copy  


from collections import OrderedDict, defaultdict
#sys.path.append("../tools/")

#==============================================================================
# Opening excel workbook in python and reading the sheets by their index values
#==============================================================================

from excel_reading import Opening_excel


Sheet,CountryData = Opening_excel('CountryRanking_Data.xlsx')
ExternalData_2017 = copy.deepcopy(CountryData['ExternalData_2017'])
InternalData_2017 = copy.deepcopy(CountryData['InternalData_2017'])
CompleteMatrix = copy.deepcopy(CountryData['CompleteMatrix'])
BrandFlagExternal = copy.deepcopy(CountryData['BrandFlagExternal'])
FeaturesMapping = copy.deepcopy(CountryData['FeaturesMapping'])


#==============================================================================
#==============================================================================
# # Creating a dictionary 'Data' which has both external and internal countries data
# # Then I'll make sets for the features and countries name from the data 
#==============================================================================
#==============================================================================

Data = {}
for d in CountryData.values():
    for k, v in d.iteritems():
        Data.setdefault(k, {}).update(v)

a = 'CompleteMatrix'

#Counting features in dataset  
OverallFeatures = set(x for l in CountryData[a].values() for x in l)     #set(x for i in xls.sheet_names for l in CountryData[i].values() for x in l)
ExternalFeatures = list(set(x for l in ExternalData_2017.values() for x in l))
InternalFeatures = list(set(x for l in InternalData_2017.values() for x in l))
#==============================================================================
# print '\n Features for each record:    ', sorted(OverallFeatures)
# print '\n Count of features for every record:    ', len(OverallFeatures)
#==============================================================================



#Counting countries in dataset
CountryNames = set()
for i,j in CountryData[a].items():
    if type(i) is tuple:
        CountryNames.add(i[1])
        
#==============================================================================
# print "\n Total Countries : ", len(CountryNames)   
#==============================================================================

SignificantFeature = []
for i,j in FeaturesMapping.items():
    for k,l in j.items():
        if l==0:
            SignificantFeature.append(i)

def RemoveItem(d,key):
    r = dict(d)
    for i,j in r.items():
        if i==key:
            del r[i]
    return r

for i,j in CountryData[a].items():
    for m in range(len(SignificantFeature)):
        for k,l in j.items():
            if k==SignificantFeature[m]:
                del j[k]

#print CountryData[a]
        
    

##==============================================================================
#==============================================================================
# #### Task 1: Select what features you'll use.
# ####         features_list is a list of strings, each of which is a feature name.
# ####         load the dictionary containing the dataset
#==============================================================================
##==============================================================================


from feature_format import featureFormat
from feature_format import targetFeatureSplit
from feature_format import featureFormat_nan

 


features_list_base = list(sorted(set(x for l in CountryData[a].values() for x in l),reverse=True))
retailerslist = copy.deepcopy(features_list_base[9:22])                                               #Because of total 13 retailers, when sheets other than BrandFlagExternal is used need to change the number
Only_features = copy.deepcopy(features_list_base[:9])+copy.deepcopy(features_list_base[22:])



#==============================================================================
### Task2 : Removing features which has almost 50% NULL values.
###         Min-Max Scaling of features for normalization 
###         Extract features and labels from dataset for local testing
#==============================================================================

### Store to my_dataset for easy export below.
my_dataset = copy.deepcopy(CountryData[a])

def count_valid_values(data_dict):
    """ counts the number of non-NaN values for each feature """
    counts = dict.fromkeys(data_dict.itervalues().next().keys(), 0)
    for record in data_dict:
        person = data_dict[record]
        for field in person:
            x = np.array(person[field],dtype=np.float)
#            print "{0}: {1}".format(field,x)
            if not np.isnan(x):
                counts[field] += 1
    return counts

valid_features = count_valid_values(my_dataset)
print '\n Count of valid(non-NAN) records for each feature:    ', valid_features

for i,j in valid_features.items():
    if (i not in InternalFeatures and j<1690):
        for k,l in my_dataset.items():
            del l[i]
#print "\n The dataset is    ",my_dataset

Valid_dict = dict((k, v) for k, v in valid_features.items() if (v >=1690 or k in InternalFeatures))
valid_ = sorted(Valid_dict.keys(),reverse=True)
valid_ = valid_[:9]+valid_[22:]
print '\n Valid features are   :    ',valid_




#==============================================================================
#==============================================================================
# # ### Task3 : Plotting of features
#==============================================================================
#==============================================================================
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

data1 = featureFormat(my_dataset, valid_)

##Scaling the features
scaler = preprocessing.MinMaxScaler()
data1 = scaler.fit_transform(data1)

list_base = []
for i in range(len(valid_)):
    j = []
    for point in data1:
        j.append(point[i])
    list_base.append(j)


df = pd.DataFrame(list_base,index=valid_)
FinalDf = df.transpose()
#print FinalDf







data2 = featureFormat_nan(my_dataset,valid_)
list_base_2 = []

for i in range(len(valid_)):
    j = []
    for point in data2:
        j.append(point[i])
    list_base_2.append(j)


df_2 = pd.DataFrame(list_base_2,index=valid_)
FinalDf_2 = df_2.transpose()
FinalDf_2 = FinalDf_2.sub(FinalDf_2.min()).div((FinalDf_2.max() - FinalDf_2.min()))
#print FinalDf_2

df2_mean = FinalDf_2.mean(axis = 1)
df2_mean = df2_mean.fillna(0)
#print df2_mean
#
X = FinalDf.as_matrix()
##print X
Y = df2_mean.as_matrix()
##print Y

Y_dict = df2_mean.to_dict()
Y_list = df2_mean.tolist()


#rf = RandomForestRegressor()
#rf.fit(X, Y)
#
#print "Features sorted by their score:"
#print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), valid_), 
#             reverse=True)


t0 = time()             
             
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 0.7, 0.3))
     scores.append((round(np.mean(score), 3), valid_[i]))
print sorted(scores, reverse=True)

print "\n Decision tree algorithm time =    ", round(time()-t0, 5), "s"

#
#
#
#
##
##def Best_Features(data_dict, features_list,k_value):
##    """ runs scikit-learn's SelectKBest feature selection
##        returns dict where keys=features, values=scores
##    """
##    FeatureData = featureFormat(data_dict, features_list)
##    labels_1, features = targetFeatureSplit(FeatureData)
##    
##    k_best = SelectKBest(k=k_value)
##    k_best.fit(features, labels_1)
##    scores = np.nan_to_num(k_best.scores_)
##    unsorted_pairs = zip(features_list[1:], scores)
##    sorted_pairs = sorted(unsorted_pairs, key=lambda x: x[1],reverse = True)[:k_value]
##    k_best_features = dict(sorted_pairs[:k_value])
###    print "\n {0} best features:    {1}".format(k_value, sorted_pairs)
##    return k_best_features,sorted_pairs,labels_1,features
##
##
##
##labels = []
##for i in range(0,len(retailerslist)):
##    Only_features_new = copy.deepcopy(Only_features)
##    Only_features_new.insert(0,retailerslist[i])
##    k_best,sorted_pairs,labels_1,features = Best_Features(CountryData[a], Only_features_new, 10)
##    labels.append(labels_1)
##
###print "\n Features are       ", labels   
##print "\n The best features for brand {0}:    {1}".format(Only_features_new[0], sorted_pairs)
#
##==============================================================================
## print 'k best features are: ', k_best
##==============================================================================
#
#
##features_list = []
##features_list += k_best.keys()
#
#
#
#
#
#
#
#
##scatter_matrix(FinalDf, alpha=1, figsize=(18,18), diagonal='kde')
##
##
###print(FinalDf.describe())
###print(FinalDf.corr())
###print(FinalDf.cov())
##
##
##
##
##
###==============================================================================
### ### Task 4:  Try a varity of classifiers
### ###           Please name your classifier clf for easy export below.
### ###           Note that if you want to do PCA or other multi-stage operations,
### ###           you'll need to use Pipelines. For more info:
### ###           http://scikit-learn.org/stable/modules/pipeline.html
###==============================================================================
##
### Try a variety of classifiers.
##### use KFold for split and validate algorithm
##
##
##from sklearn import model_selection
##kf= model_selection.KFold(n_splits=5,shuffle=True,random_state=100)
##for train_indices, test_indices in kf.split(features):
##    #make training and testing sets
##    features_train= [features[ii] for ii in train_indices]
##    features_test= [features[ii] for ii in test_indices]
##    labels_train=[labels[ii] for ii in train_indices]
##    labels_test=[labels[ii] for ii in test_indices]
##
##    
##t0 = time()
##
##### K-means Clustering
##from sklearn.cluster import KMeans
##k_clf = KMeans(n_clusters=266, tol=0.001)
##
##### Adaboost Classifier
##from sklearn.ensemble import AdaBoostClassifier
##a_clf = AdaBoostClassifier(algorithm='SAMME.R')
##
##### Support Vector Machine Classifier
##from sklearn.svm import SVC
##s_clf = SVC(kernel='rbf', C=1000)
##
##### Random Forest
##from sklearn.ensemble import RandomForestClassifier
##rf_clf = RandomForestClassifier()
##
##### Stochastic Gradient Descent - Logistic Regression
##from sklearn.linear_model import SGDClassifier
##g_clf = SGDClassifier(loss='log')
##
##### Decision Tree Classifier
##from sklearn.tree import DecisionTreeClassifier
##d_clf = DecisionTreeClassifier()
##
##### Naive Bayes GaussianNB
##from sklearn.naive_bayes import GaussianNB
##gnb_clf = GaussianNB()
##
##### Logistic Regression Classifier
##from sklearn.linear_model import LogisticRegression
##l_clf = LogisticRegression(C=10**20, solver='liblinear', tol=10**-20)
##
##
##k_clf.fit(features_train,labels_train)
##pred = k_clf.predict(features_test)
##score = k_clf.score(features_test,labels_test)
##
##print '\n Accuracy score before tuning =    ', score
###print 'Precision Score = ', precision_score(labels_test,pred,average = 'binary')
##print "\n Decision tree algorithm time =    ", round(time()-t0, 5), "s"
##
##
##
##
##
###==============================================================================
### ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### ###           using our testing script. Check the tester.py script in the final project
### ###           folder for details on the evaluation method, especially the test_classifier
### ###           function. Because of the small size of the dataset, the script uses
### ###           stratified shuffle split cross validation. For more info: 
### ###           http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
###==============================================================================
##
### Uncomment validation 1 and validation 2 one at a time to get the individual results
##
##
#### Validation 1: train_test_split ####
##features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.1, random_state=50)
##print labels_train
##
##### Validation 2: StratifiedShuffleSplit ####
###from sklearn.cross_validation import StratifiedShuffleSplit
###cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
###for train_idx, test_idx in cv: 
###        features_train = []
###        features_test  = []
###        labels_train   = []
###        labels_test    = []
###        for ii in train_idx:
###            features_train.append( features[ii] )
###            labels_train.append( labels[ii] )
###        for jj in test_idx:
###            features_test.append( features[jj] )
###            labels_test.append( labels[jj] )
##
##
#### Two classifier have been tuned for better performance. Try to uncomment one classifier at a time to get individual result
##
###### Decision Tree Classifier: ####
###clf = DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=None,
###            max_features=None, max_leaf_nodes=18, min_samples_leaf=1,
###            min_samples_split=11, min_weight_fraction_leaf=0.0,
###            presort=False, random_state=None, splitter='best')
###clf = clf.fit(features_train,labels_train)
###pred= clf.predict(features_test)
###
###
######Logistic Regression Classifier: #####
###clf = LogisticRegression(C=10**27, class_weight='balanced', dual=False,
###          fit_intercept=True, intercept_scaling=1, max_iter=150,
###          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
###          solver='liblinear', tol=10**-30, verbose=0, warm_start=False)
###clf = clf.fit(features_train,labels_train)
###pred = clf.predict(features_test)
##
##
##### KMeans Clustering Classifier: #####
##clf = KMeans(n_clusters=10, init='k-means++', n_init=20, max_iter=400, tol=0.0001, precompute_distances='auto', 
##             verbose=0, random_state=100, copy_x=True, n_jobs=1, algorithm='auto')
##clf = clf.fit(features_train,labels_train)
##pred = clf.predict(features_test)
##
##
##accuracy=accuracy_score(labels_test, pred)
##
##print "\n Validating algorithm:"
##print " Accuracy score after tuning =    ", accuracy
##
### Precision score is the ratio of true positives to both true positives and false positives.
### Using precision_score function to calculate precision score. 
##print ' Precision Score =    ', precision_score(labels_test,pred,pos_label = 0,average ='weighted')
##
### Recall score is the ratio of true positives to true positives and false negatives
### Using recall_score function to calculate recall score. 
##print ' Recall Score =    ', recall_score(labels_test,pred,pos_label = 0,average ='weighted')
##
##
##### Task 6: Dump your classifier, dataset, and features_list so anyone can
##### check your results. You do not need to change anything below, but make sure
##### that the version of poi_id.py that you submit can be run on its own and
##### generates the necessary .pkl files for validating your results.
##
##pickle.dump(clf, open("my_classifier.pkl", "w") )
##pickle.dump(data_dict, open("my_dataset.pkl", "w") )
##pickle.dump(ExternalFeatures, open("my_feature_list.pkl", "w") )
##
##
