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

a = 'CompleteMatrix'

#Counting features in dataset  
OverallFeatures = sorted( FeaturesMapping.keys())   #set(x for i in xls.sheet_names for l in CountryData[i].values() for x in l)
ExternalFeatures = OverallFeatures[:-9]
InternalFeatures = OverallFeatures[-9:]
# =============================================================================
# print '\n Features for each record:    ', sorted(OverallFeatures)
# print '\n Count of features for every record:    ', len(OverallFeatures)
# =============================================================================




#Counting countries in dataset
CountryNames = set()
for i,j in CountryData[a].items():
    if type(i) is tuple:
        CountryNames.add(i[1])
        
#==============================================================================
# print "\n Total Countries : ", len(CountryNames)   
#==============================================================================



        
    

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
#print '\n',Retailers




def dataset_maker(data_dict, value):
      p_dict = {}
#      parent_dict = {}
      for i,j in data_dict.groupby(value):
            p_dict.update({str(i) : j.reset_index(drop=True)})
#      for k,l in p_dict.groupby(Flag):
#            parent_dict.update({str(k) : j.reset_index(drop=True)})      
      return p_dict

parent_dataset_1 = dataset_maker(Sheet['CompleteMatrix'], 'Brand')

                
                

di = {}
for i,j in parent_dataset_1.items():
      j.set_index(['CountryCode'],  inplace=True)
      di[i] = j.to_dict(orient='index')
      
#print '\n the parent dataset is ',di['Kuiu']
#print '\n the parent dataset is ',di_11


InSignificantFeature = []
for i,j in FeaturesMapping.items():
    for k,l in j.items():
        if l==0:
            InSignificantFeature.append(i)

#print '\n InSignificantFeature ',InSignificantFeature

for i,j in di.items():                     #CountryData[a].items():
      for k,l in j.items():
            for a in range(len(InSignificantFeature)):
                  for m,n in l.items():
                        if m==InSignificantFeature[a]:
                              del l[m]


#print '\n the parent dataset is ',di['Calvin Klein']


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
my_dataset,valid_features,Valid_dict,valid_ = {},{},{},{}
for i in Retailers:
      my_dataset[i] = copy.deepcopy(di[i])
      valid_features[i] = count_valid_values(my_dataset[i])
#      print '\n Count of valid(non-NAN) records for {0} feature: {1}   '.format(i,valid_features[i])
      my_dataset[i] = removing_Nan_features(my_dataset[i])                       
#Valid_dict,valid_ = {},{}
#for i in Retailers:
      Valid_dict[i] = dict((k, v) for k, v in valid_features[i].items() if (v >=130 or k in InternalFeatures))
      valid_[i] = sorted(Valid_dict[i].keys(),reverse=True)
      valid_[i] = valid_[i][22:]                          ##valid_[i][:9]+valid_[i][22:]                  ##Use the commented code for all internal and exteranl features
#      print '\n Valid {0} features are   :  {1}   '.format(i,valid_[i])





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


data1,data2,list_base,list_base_2,df,df_2,FinalDf,FinalDf_2,df2_mean,X,Y,Y_dict,Y_list = {},{},{},{},{},{},{},{},{},{},{},{},{}
for i in Retailers:
      data1[i] = featureFormat(my_dataset[i], valid_[i])
      data2[i] = featureFormat_nan(my_dataset[i],valid_[i])
      data1[i] = scaler.fit_transform(data1[i])
#list_base = {}
#for i in Retailers:
      list_ = []
      for k in range(len(valid_[i])):
            j = []
            for point in data1[i]:
                  j.append(point[k])
            list_.append(j)
      list_base.update({i:list_})
#df = {}
#FinalDf = {}
#for i in Retailers:
      df[i] = pd.DataFrame(list_base[i],index=valid_[i])
      FinalDf[i] = df[i].transpose()
#list_base_2 = {}
#for i in Retailers:
      list2_ = []
      for k in range(len(valid_[i])):
            j1 = []
            for point in data2[i]:
                  j1.append(point[k])
            list2_.append(j1)
      list_base_2.update({i:list2_})
#df_2 = {}
#FinalDf_2 = {}
#for i in Retailers:
      df_2[i] = pd.DataFrame(list_base_2[i],index=valid_[i])
      FinalDf_2[i] = df_2[i].transpose()
      FinalDf_2[i] = FinalDf_2[i].sub(FinalDf_2[i].min()).div((FinalDf_2[i].max() - FinalDf_2[i].min()))     
#df2_mean = {}
#for i in Retailers:
      df2_mean[i] = FinalDf_2[i].mean(axis = 1)
      df2_mean[i] = df2_mean[i].fillna(0)
#X,Y,Y_dict,Y_list = {},{},{},{}
#for i in Retailers:
      X[i] = FinalDf[i].as_matrix()
      Y[i] = df2_mean[i].as_matrix()
# =============================================================================
      Y_dict[i] = df2_mean[i].to_dict()
      Y_list[i] = df2_mean[i].tolist()
# =============================================================================



print "\n Data Preparation time =    ", round(time()-t0, 5), "seconds"


t1 = time()
  
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # #rf = RandomForestRegressor()
# # # # #rf.fit(X, Y)
# # # # #
# # # # #print "Features sorted by their score:"
# # # # #print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), valid_), 
# # # # #             reverse=True)
# =============================================================================
# =============================================================================
# =============================================================================
# 
# =============================================================================

Score = {}
rf = RandomForestRegressor(n_estimators=30, max_depth=4)
for i in Retailers:
      scores = {}
      x_ = X[i]
      y_ = Y[i]
      v_ = valid_[i]
      for k in range(x_.shape[1]):
            score = cross_val_score(rf, x_[:, k:k+1], y_, scoring="r2",
                                          cv=ShuffleSplit(len(x_), 0.7, 0.3))
            scores.update({v_[k]:round(np.mean(score), 3)})
      Score.update({i:scores})
      
             
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # rf = RandomForestRegressor(n_estimators=30, max_depth=4)
# # # # scores = []
# # # # for i in range(X.shape[1]):
# # # #      score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
# # # #                               cv=ShuffleSplit(len(X), 0.7, 0.3))
# # # #      scores.append((round(np.mean(score), 3), valid_[i]))
# # # # print sorted(scores, reverse=True)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

print "\n Decision tree algorithm time =    ", round(time()-t1, 5), "seconds"





df_output = pd.DataFrame.from_dict(data=Score,orient = 'index')
df_output.to_csv('RetailerDatasets_OnlyExternal.csv',index=True)

