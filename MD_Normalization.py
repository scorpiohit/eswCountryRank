# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:25:45 2018

@author: Mohit Sharma
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from feature_format import featureFormat
from feature_format import targetFeatureSplit
from feature_format import featureFormat_nan
import copy  


xls = pd.ExcelFile('FeatureScores.xlsx')
FeatureScores = pd.read_excel(xls,0)
FeatureScores.set_index(['Brand'],  inplace=True)
FeatureScores = FeatureScores.to_dict(orient='index')

OverallFeatures = sorted(list(set(x for l in FeatureScores.values() for x in l)))
ExternalFeatures = OverallFeatures[:-9]
InternalFeatures = OverallFeatures[-9:]

#print '\n Features for each record:    ', InternalFeatures



MasterData = pd.read_excel(xls,2)
MasterData.set_index(['Brand','CountryCode'],  inplace=True)
MasterData = MasterData.to_dict(orient='index')
MasterData = pd.DataFrame(MasterData)
MasterData = MasterData.transpose()
MasterData = MasterData.to_dict()

#print '\n Normalized FeatureScores : ', MasterData.keys()





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





valid_features = count_valid_values(FeatureScores)
valid_ = sorted(valid_features.keys(),reverse=True)
valid_md = list(sorted(set(x for l in MasterData.values() for x in l),reverse=True))
#valid_md = valid_md[0:9]+valid_md[23:]
valid_i = valid_[:9]
valid_e = valid_[9:]   
#print '\n Features for each record:    ', len(valid_md)




from sklearn.preprocessing import Imputer
imp_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp_median = Imputer(missing_values='NaN', strategy='median', axis=0)
 



data1 = featureFormat(FeatureScores, valid_)
data2 = featureFormat(MasterData, valid_md)


X_normalized1 = normalize(data1, norm='l1')
X_normalized1 = X_normalized1.tolist()
X_normalized1.insert(0,valid_)


MD_mean = imp_mean.fit_transform(data2)
X_normalized_mean = normalize(MD_mean, norm='l1')
X_normalized_mean = X_normalized_mean.tolist()
X_normalized_mean.insert(0,valid_md)

MD_median = imp_median.fit_transform(data2)
X_normalized_median = normalize(MD_median, norm='l1')
X_normalized_median = X_normalized_median.tolist()
X_normalized_median.insert(0,valid_md)







#print '\n Normalized FeatureScores : ', FeatureScores.keys()
print '\n Normalized FeatureScores : ', MasterData.keys()


#import csv
###
###with open('FS_norm.csv', 'w') as csvfile:
###    writer = csv.writer(csvfile)
###    writer.writerows(X_normalized1)
###    
#with open('MD_norm_mean.csv', 'wb') as csvfile:
#      writer = csv.writer(csvfile)
#      writer.writerows(X_normalized_mean)
#    
#with open('MD_norm_median.csv', 'wb') as csvfile:
#      writer = csv.writer(csvfile)
#      writer.writerows(X_normalized_median)
    
    
            
      

