# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:02:47 2018

@author: msharma
"""

import numpy as np
import math
#==============================================================================
#      convert dictionary to numpy array of features
#        remove_None = True will convert "None" string to 0.0
#        remove_all_zeroes = True will omit any data points for which
#            all the features you seek are 0.0
#        remove_any_zeroes = True will omit any data points for which
#            any of the features you seek are 0.0
#        sort_keys = True sorts keys by alphabetical order. Setting the value as
#            a string opens the corresponding pickle file with a preset key
#            order (this is used for Python 3 compatibility, and sort_keys
#            should be left as False for the course mini-projects).
#    
#==============================================================================

def featureFormat( dictionary, features,remove_NaN=True, remove_None=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):



    return_list = []



    if sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()
        
        

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key ", feature, " not present"
            except TypeError:
                print "Argument is not string or number"
            except ValueError:
                print "Input values are of proper size"
                return
            value = dictionary[key][feature]
            if (value is None or math.isnan(float(value))) and (remove_None or remove_NaN) :   
                value = 0
            tmp_list.append( float(value) )
            
            
        append = True
        test_list = tmp_list

        # Logic for deciding whether or not to add the data point.
        append = True

        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 :   #and item != "None" and item != "NaN"
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list :    #or "None" in test_list or "NaN" in test_list
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)




#==============================================================================
# def targetFeatureSplit( data ):
#     """ 
#         given a numpy array like the one returned from
#         featureFormat, separate out the first feature
#         and put it into its own list (this should be the 
#         quantity you want to predict)
# 
#         return targets and features as separate lists
# 
#         (sklearn can generally handle both lists and numpy arrays as 
#         input formats when training/predicting)
#     """
# 
#     target = []
#     features = []
#     for item in data:
#         target.append( item[0] )
#         features.append( item[1:] )
# 
#     return target, features
#==============================================================================
