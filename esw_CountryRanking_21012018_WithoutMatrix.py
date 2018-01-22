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
from sklearn.feature_selection import SelectKBest
import pandas as pd
import numpy as np


from collections import OrderedDict, defaultdict
sys.path.append("../tools/")

#==============================================================================
# Opening excel workbook in python and reading the sheets by their index values
#==============================================================================

from excel_reading import Opening_excel


Sheet,CountryData = Opening_excel('CountryRanking_Data.xlsx')
ExternalData_2017 = CountryData['ExternalData_2017']
InternalData_2017 = CountryData['InternalData_2017']


#==============================================================================
# Creating a dictionary 'Data' which has both external and internal countries data
# Next step is to add Brands key to the dataset which is present in the tuple
# Then I'll make sets for the features and countries name separately for external and internal countries data 
#==============================================================================

Data = {}
for d in CountryData.values():
    for k, v in d.iteritems():
        Data.setdefault(k, {}).update(v)

for i,j in Data.items():
    if type(i) is tuple:
        j['Brand'] = i[0]

   
OverallFeatures = set(x for l in Data.values() for x in l)     #set(x for i in xls.sheet_names for l in CountryData[i].values() for x in l)
ExternalFeatures = list(set(x for l in ExternalData_2017.values() for x in l))
InternalFeatures = list(set(x for l in InternalData_2017.values() for x in l))
#==============================================================================
# #print '\n Features for each record:    ', sorted(OverallFeatures)
# #print '\n Count of features for every record:    ', len(OverallFeatures)
#==============================================================================




InCountry = set()
ExCountry = set()


for i,j in Data.items():
    if type(i) is tuple:
        InCountry.add(i[1])
    else:
        ExCountry.add(i)  
#    j.update({'New Rank':None})
#    j.update({'New OverallScore':None})


OverallCountries = InCountry.union(ExCountry)
#==============================================================================
# #print "Total Internal Countries : ", sorted(InCountry)   
# #print "Total External Countries : ", sorted(ExCountry)  
# #print "Total Countries : ", sorted(OverallCountries) 
#==============================================================================




        
temp1 = []
temp2 = []
dictList1 = []
dictList2 = []

 
for key, value in Data.iteritems():
    temp1 = [key,value]
    if type(key) is tuple:
        dictList1.append(temp1)
    else:
        dictList2.append(temp1)




PreProcessedDataInternal = {row[0]:row[1] for row in dictList1}
PreProcessedDataExternal = {row[0]:row[1] for row in dictList2}


#==============================================================================
# If you have a list of lists and want EVERYTHING to be duplicated, you need to perform a DEEP copy
#==============================================================================
import copy       
dictList3 = copy.deepcopy(dictList1)



for i in dictList2:
    if i[0] not in InCountry:
        temp2 = [i[0],i[1]]
        dictList3.append(temp2)


for i in dictList3:
    if type(i[0]) is not tuple:
        for x in range(len(InternalFeatures)):
            i[1].update({InternalFeatures[x]:None})
    elif i[0][1] not in ExCountry:
        for x in range(len(ExternalFeatures)):
            i[1].update({ExternalFeatures[x]:None})
    for j in dictList2:
        if i[0][1]==j[0]:
            i[1].update(j[1])



Brand = {"Brand":None}
for i in dictList3:
    if type(i[0]) is not tuple:
        i[1].update(Brand)


PreProcessedData = {row[0]:row[1] for row in dictList3}
#print PreProcessedData




##==============================================================================
#### Task 1: Select what features you'll use.
####         features_list is a list of strings, each of which is a feature name.
####         load the dictionary containing the dataset
##==============================================================================


from feature_format import featureFormat
from feature_format import targetFeatureSplit


OverallFeatures.remove('Brand')
features_list_base = list(OverallFeatures)
#print features_list_base



def Best_Features(data_dict, features_list, k_value):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    FeatureData = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(FeatureData)

    k_best = SelectKBest(k=k_value)
    k_best.fit(features, labels)
    scores = np.nan_to_num(k_best.scores_)
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = sorted(unsorted_pairs, key=lambda x: x[1],reverse = True)[:k_value]
    k_best_features = dict(sorted_pairs[:k_value])
    print "\n {0} best features:    {1}".format(k_value, sorted_pairs)
    return k_best_features



#k_best = Best_Features(PreProcessedDataInternal, InternalFeatures, 3)
k_best = Best_Features(PreProcessedDataExternal, ExternalFeatures, 10)
#k_best = Best_Features(PreProcessedData, features_list_base, 7)
#==============================================================================
# #print 'k best features are: ', k_best
#==============================================================================

features_list = []
features_list += k_best.keys()

#==============================================================================
# ### Task2 : Plotting of features
#==============================================================================
from pandas.plotting import scatter_matrix


data1 = featureFormat(PreProcessedDataExternal, ExternalFeatures)
list_base = []

for i in range(len(ExternalFeatures)):
    j = []
    for point in data1:
        j.append(point[i])
    list_base.append(j)


df = pd.DataFrame(list_base,index=ExternalFeatures)
FinalDf = df.transpose()
#scatter_matrix(FinalDf, alpha=1, figsize=(18,18), diagonal='kde')





#==============================================================================
### Task3 : Min-Max Scaling of features for normalization 
###         Extract features and labels from dataset for local testing
#==============================================================================

### Store to my_dataset for easy export below.
my_dataset = PreProcessedDataExternal

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


### Extract features and labels from dataset for local testing


 def targetFeatureSplit( data ):
     """ 
         given a numpy array like the one returned from featureFormat, separate out the first feature and put it into its own list 
         (this should be the quantity you want to predict) return targets and features as separate lists
         (sklearn can generally handle both lists and numpy arrays as input formats when training/predicting)
     """
     target = []
     features = []
     for item in data:
         target.append( item[0] )
         features.append( item[1:] )
         
     return target, features



data_dict = featureFormat(my_dataset, features_list, sort_keys = True)
#print "\n data_dict:", data_dict
labels, features = targetFeatureSplit(data_dict)
#print "\n The Labels are     :      ", labels
#print "\n The Labels are     :      ", features


# scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)






#==============================================================================
# ### Task 4:  Try a varity of classifiers
# ###           Please name your classifier clf for easy export below.
# ###           Note that if you want to do PCA or other multi-stage operations,
# ###           you'll need to use Pipelines. For more info:
# ###           http://scikit-learn.org/stable/modules/pipeline.html
#==============================================================================

# Try a variety of classifiers.
### use KFold for split and validate algorithm


from sklearn import model_selection
kf= model_selection.KFold(n_splits=10)
for train_indices, test_indices in kf.split(features):
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

    
t0 = time()

### K-means Clustering
from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.001)

### Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
a_clf = AdaBoostClassifier(algorithm='SAMME.R')

### Support Vector Machine Classifier
from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=1000)

### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()

### Stochastic Gradient Descent - Logistic Regression
from sklearn.linear_model import SGDClassifier
g_clf = SGDClassifier(loss='log')

### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
d_clf = DecisionTreeClassifier()

### Naive Bayes GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB()

### Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
l_clf = LogisticRegression(C=10**20, solver='liblinear', tol=10**-20)


k_clf.fit(features_train,labels_train)
pred = k_clf.predict(features_test)
score = k_clf.score(features_test,labels_test)

print '\n Accuracy score before tuning =    ', score
#print 'Precision Score = ', precision_score(labels_test,pred,average = 'binary')
print "\n Decision tree algorithm time =    ", round(time()-t0, 5), "s"





#==============================================================================
# ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
# ###           using our testing script. Check the tester.py script in the final project
# ###           folder for details on the evaluation method, especially the test_classifier
# ###           function. Because of the small size of the dataset, the script uses
# ###           stratified shuffle split cross validation. For more info: 
# ###           http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
#==============================================================================
#
### Uncomment validation 1 and validation 2 one at a time to get the individual results
#
#
#### Validation 1: train_test_split ####
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=50)
#
#
#### Validation 2: StratifiedShuffleSplit ####
#from sklearn.cross_validation import StratifiedShuffleSplit
#cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
#for train_idx, test_idx in cv: 
#        features_train = []
#        features_test  = []
#        labels_train   = []
#        labels_test    = []
#        for ii in train_idx:
#            features_train.append( features[ii] )
#            labels_train.append( labels[ii] )
#        for jj in test_idx:
#            features_test.append( features[jj] )
#            labels_test.append( labels[jj] )
#
#
### Two classifier have been tuned for better performance. Try to uncomment one classifier at a time to get individual result
#
#### Decision Tree Classifier: ####
#clf = DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=None,
#            max_features=None, max_leaf_nodes=18, min_samples_leaf=1,
#            min_samples_split=11, min_weight_fraction_leaf=0.0,
#            presort=False, random_state=None, splitter='best')
#clf = clf.fit(features_train,labels_train)
#pred= clf.predict(features_test)
#
#
####Logistic Regression Classifier: #####
#clf = LogisticRegression(C=10**27, class_weight='balanced', dual=False,
#          fit_intercept=True, intercept_scaling=1, max_iter=150,
#          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#          solver='liblinear', tol=10**-30, verbose=0, warm_start=False)
#clf = clf.fit(features_train,labels_train)
#pred = clf.predict(features_test)
#
#
#### KMeans Clustering Classifier: #####
#clf = KMeans(n_clusters=10, n_init=20, max_iter=400, tol=0.0001, precompute_distances='auto', 
#             verbose=1, random_state=100, copy_x=True, n_jobs=1, algorithm='auto')
#clf = clf.fit(features_train,labels_train)
#pred = clf.predict(features_test)
#
#
#accuracy=accuracy_score(labels_test, pred)
#
#print "\n Validating algorithm:"
#print " Accuracy score after tuning =    ", accuracy
#
## Precision score is the ratio of true positives to both true positives and false positives.
## Using precision_score function to calculate precision score. 
#print ' Precision Score =    ', precision_score(labels_test,pred,pos_label = 0,average ='binary')
#
## Recall score is the ratio of true positives to true positives and false negatives
## Using recall_score function to calculate recall score. 
#print ' Recall Score =    ', recall_score(labels_test,pred,pos_label = 0,average ='binary')
#
#
#### Task 6: Dump your classifier, dataset, and features_list so anyone can
#### check your results. You do not need to change anything below, but make sure
#### that the version of poi_id.py that you submit can be run on its own and
#### generates the necessary .pkl files for validating your results.
#
#pickle.dump(clf, open("my_classifier.pkl", "w") )
#pickle.dump(data_dict, open("my_dataset.pkl", "w") )
#pickle.dump(ExternalFeatures, open("my_feature_list.pkl", "w") )
#
#
