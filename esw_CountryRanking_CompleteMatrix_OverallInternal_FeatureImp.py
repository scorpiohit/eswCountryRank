# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:29:39 2018

@author: Mohit Sharma
"""

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
import xlsxwriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectKBest,chi2
import pandas as pd
import numpy as np
import copy  


t0 = time()  

from collections import OrderedDict, defaultdict
#sys.path.append("../tools/")

#==============================================================================
# =============================================================================
# # Task1  : Opening excel workbook in python and reading the sheets by their index values
# #           Creating a dictionary 'Data' which has both external and internal countries data
# #           Then I'll make sets for the features and countries name from the data 
# =============================================================================
#==============================================================================

from excel_reading import Opening_excel


Sheet,CountryData = Opening_excel('CountryRanking_Data.xlsx')
Matrix_Internal_Retailer = copy.deepcopy(CountryData['Matrix_Internal_Retailer'])
CompleteMatrix = copy.deepcopy(CountryData['CompleteMatrix'])
Matrix_Overall = copy.deepcopy(CountryData['Matrix_Overall'])
Matrix_External_BrandFlag = copy.deepcopy(CountryData['Matrix_External_BrandFlag'])
FeaturesMapping = copy.deepcopy(CountryData['FeaturesMapping'])



Data = {}
for d in CountryData.values():
    for k, v in d.iteritems():
        Data.setdefault(k, {}).update(v)

a = 'Matrix_Overall'

#Counting features in dataset  
OverallFeatures = sorted( FeaturesMapping.keys())   #set(x for i in xls.sheet_names for l in CountryData[i].values() for x in l)
ExternalFeatures = OverallFeatures[:-9]
InternalFeatures = OverallFeatures[-9:]
# =============================================================================
# print '\n Features for each record:    ', sorted(OverallFeatures)
# print '\n Count of features for every record:    ', len(OverallFeatures)
# =============================================================================




#Counting countries in dataset
CountryNames = CountryData[a].keys()
#for i,j in CountryData[a].items():
#    if type(i) is tuple:
#        CountryNames.add(i[1])
        
#==============================================================================
# print "\n Total Countries : ", len(CountryNames)   
#==============================================================================

InSignificantFeature = []
for i,j in FeaturesMapping.items():
    for k,l in j.items():
        if l==0:
            InSignificantFeature.append(i)

for i,j in CountryData[a].items():
    for m in range(len(InSignificantFeature)):
        for k,l in j.items():
            if k==InSignificantFeature[m]:
                del j[k]


   
    

##==============================================================================
#==============================================================================
# #### Task2 : Select what features you'll use.
# ####         features_list is a list of strings, each of which is a feature name.
# ####         load the dictionary containing the dataset
#==============================================================================
##==============================================================================


from feature_format import featureFormat
from feature_format import targetFeatureSplit
from feature_format import featureFormat_nan

 


features_list_base = list(sorted(set(x for l in CountryData[a].values() for x in l),reverse=True))
retailerslist = copy.deepcopy(features_list_base[9:22])                                               #Because of total 13 retailers, when sheets other than BrandFlagExternal is used need to change the number
Retailers = sorted([s.replace('Flag ','') for s in retailerslist])
Only_features = copy.deepcopy(features_list_base[:9])+copy.deepcopy(features_list_base[22:])
#print '\n',features_list_base




def dataset_maker(data_dict):
      p_dict = {}
      for i,j in data_dict.items():
            for k,l in j.items():
                  x = np.array(l,dtype=np.float)
                  if k =='I5' and np.isnan(x)==False:
                        p_dict.update({str(i):j})    
      return p_dict

CountryData[a] = dataset_maker(CountryData[a])



#==============================================================================
# =============================================================================
# ### Task3 : Removing features which has almost 50% NULL values.
# ###         Min-Max Scaling of features for normalization 
# ###         Extract features and labels from dataset for local testing
# =============================================================================
#==============================================================================


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


def removing_Nan_features(data_dict):
      
      vf = count_valid_values(data_dict)
      for i,j in vf.items():
            if (i not in InternalFeatures and j<130):
                  for k,l in data_dict.items():
                        del l[i]
      return data_dict




### Store to my_dataset for easy export below.
#my_dataset,valid_features,Valid_dict,valid_ = {},{},{},{}

my_dataset = copy.deepcopy(CountryData[a])
valid_features = count_valid_values(my_dataset)
#print '\n Count of valid(non-NAN) records for feature :  ',valid_features
my_dataset = removing_Nan_features(my_dataset)                       
#Valid_dict,valid_ = {},{}
#for i in Retailers:
Valid_dict = dict((k, v) for k, v in valid_features.items() if (v >=130 or k in InternalFeatures))
valid_ = sorted(Valid_dict.keys(),reverse=True)
valid_ = valid_[4:9]                         ##valid_[i][:9]+valid_[i][22:]
print '\n Valid features are   :  {0}   '.format(valid_)





#==============================================================================
#==============================================================================
# # ### Task3 : Plotting of features
#==============================================================================
#==============================================================================
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing


##Scaling the features
scaler = preprocessing.MinMaxScaler()




data1 = featureFormat(my_dataset, valid_)
data2 = featureFormat_nan(my_dataset,valid_)
data1 = scaler.fit_transform(data1)

list_base = []
for k in range(len(valid_)):
      j = []
      for point in data1:
            j.append(point[k])
      list_base.append(j)

df = pd.DataFrame(list_base,index=valid_)
FinalDf = df.transpose()

list_base_2 = []
for k in range(len(valid_)):
      j1 = []
      for point in data2:
            j1.append(point[k])
      list_base_2.append(j1)

df_2 = pd.DataFrame(list_base_2,index=valid_)
FinalDf_2 = df_2.transpose()
FinalDf_2 = FinalDf_2.sub(FinalDf_2.min()).div((FinalDf_2.max() - FinalDf_2.min()))     

df2_mean = FinalDf_2.mean(axis = 1)
df2_mean = df2_mean.fillna(0)

X = FinalDf.as_matrix()
Y = df2_mean.as_matrix()
# =============================================================================
Y_dict = df2_mean.to_dict()
Y_list = df2_mean.tolist()
# =============================================================================




print "\n Data Preparation time =    ", round(time()-t0, 5), "seconds"


t1 = time()
  



 

rf = RandomForestRegressor(n_estimators=100, max_depth=4)
#for i in Retailers:
scores = {}
rf_ = rf.fit(X,Y)
score = zip(map(lambda x: round(x, 4), rf_.feature_importances_), valid_)
for k in range(len(score)):
      scores.update({valid_[k]:score[k][0]})


print "Features sorted by their score: ",scores  

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # rf = RandomForestRegressor(n_estimators=30, max_depth=4)
# # # # #for i in Retailers:
# # # # scores = {}
# # # # #x_ = X[i]
# # # # #y_ = Y[i]
# # # # #v_ = valid_[i]
# # # # for k in range(X.shape[1]):
# # # #       score = cross_val_score(rf, X[:, k:k+1], Y, scoring="r2",
# # # #                                     cv=ShuffleSplit(len(X), 0.7, 0.3))
# # # #       scores.update({valid_[k]:round(np.mean(score), 3)})
# # # # #      Score.update({i:scores})
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
      
          

print "\n Decision tree algorithm time =    ", round(time()-t1, 5), "seconds"




#df_output = pd.DataFrame.from_dict(data=scores,orient = 'index')
#df_output.to_csv('RetailerDatasets_OverallInternal.csv',index=True)

df_output = pd.DataFrame.from_dict(data=scores,orient = 'index')
df_output.to_csv('RetailerDatasets_OverallInternal_02Feb.csv',index=True)


