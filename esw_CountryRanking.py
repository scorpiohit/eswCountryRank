# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:42:38 2018

@author: msharma
"""

import matplotlib.pyplot as plt
import sys
import pickle
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectKBest

from collections import OrderedDict, defaultdict
sys.path.append("../tools/")

#==============================================================================
# Opening excel workbook in python and reading the sheets by their index values
#==============================================================================

import pandas as pd

Sheet = {}
CountryData = {}
xls = pd.ExcelFile('CountryRanking_Data.xlsx')


for i in xls.sheet_names:
    Sheet[i]= pd.read_excel(xls, i)
    if i == 'InternalData_2017':
        Sheet[i].set_index(['Brand','Country'],  inplace=True)
    else:
        Sheet[i].set_index(['Country'],  inplace=True)
    CountryData[i] = Sheet[i].to_dict(orient='index') 

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
        j['Brands'] = i[0]


#Features = set(x for i in xls.sheet_names for l in CountryData[i].values() for x in l)   
Features = set(x for l in Data.values() for x in l)   
ExternalFeatures = list(set(x for l in ExternalData_2017.values() for x in l))
InternalFeatures = list(set(x for l in InternalData_2017.values() for x in l))
print '\n Features for each record:    ', sorted(Features)
print '\n Count of features for every record:    ', len(Features)




InCountry = set()
ExCountry = set()

for i,j in Data.items():
    if type(i) is tuple:
        InCountry.add(i[1])
    else:
        ExCountry.add(i)  
    j.update({'New Rank':None})
    j.update({'New OverallScore':None})


OverallCountries = InCountry.union(ExCountry)
print "Total Internal Countries : ", sorted(InCountry)   
print "Total External Countries : ", sorted(ExCountry)  
print "Total Countries : ", sorted(OverallCountries) 




        
temp1 = []
temp2 = []
dictList1 = []
dictList2 = []
dictList = []
 
for key, value in Data.iteritems():
    temp1 = [key,value]
    if type(key) is tuple:
        dictList1.append(temp1)
    else:
        dictList2.append(temp1)
        

for i in dictList2:
    if i[0] not in InCountry:
        temp2 = [i[0],i[1]]
        dictList1.append(temp2)

for i in dictList1:
    if type(i[0]) is not tuple:
        for x in range(len(InternalFeatures)):
            i[1].update({InternalFeatures[x]:None})
    elif i[0][1] not in ExCountry:
        for x in range(len(ExternalFeatures)):
            i[1].update({ExternalFeatures[x]:None})
    for j in dictList2:
        if i[0][1]==j[0]:
            i[1].update(j[1])

Brands = {"Brands":None}
for i in dictList1:
    if type(i[0]) is not tuple:
        i[1].update(Brands)


PreProcessedData = {row[0]:row[1] for row in dictList1}
#print PreProcessedData




##==============================================================================
#### Task 1: Select what features you'll use.
####         features_list is a list of strings, each of which is a feature name.
####         load the dictionary containing the dataset
##==============================================================================


from feature_format import featureFormat
from feature_format import targetFeatureSplit



features_list_base = ['MarketConsumptionCapacity', 
                      'MarketIntensity', 
                      'EconomicFreedom', 
                      'CountryRisk', 
                      'CommercialInfrastructure', 
                      'MarketSize', 
                      'MarketReceptivity', 
                      'MarketGrowthRate']
#                      'PreOrders',
#                      'Orders',
#                      'Order Value']
#                      'AOV']



def Best_Features(data_dict, features_list, k_value):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k_value)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = sorted(unsorted_pairs, key=lambda x: x[1],reverse = True)[:k_value]
    k_best_features = dict(sorted_pairs[:k_value])
    print "\n {0} best features:    {1}".format(k_value, sorted_pairs)
    return k_best_features


k_best = Best_Features(PreProcessedData, features_list_base, 5)
print 'k best features are: ',k_best
##
###features_list = ['poi']
###features_list += k_best.keys()
###new_features = ["total_to_salary", "expenses_to_salary"]
###features_list += new_features
###
###
####Mohit's testing
###print ExternalData_2017
###
###for key in ExternalData_2017.keys():
###    tmp_list = []
###    for feature in features_list_base:
###        try:
###            print ExternalData_2017[key][feature]
###        except ValueError:
###            print "Non-numeric data found"
####        except KeyError:
####            print "error: key ", feature, " not present"
###        value = ExternalData_2017[key][feature]
###        if value is None:
###            value = 0
###        tmp_list.append( float(value) )
###
###
###k_best = Best_Features(ExternalData_2017, features_list_base, 8)
###
###data = featureFormat(ExternalData_2017, features_list_base)
####print "\n The data_dict is ", ExternalData_2017
####for p in data: 
####    print "\n The data is ", p