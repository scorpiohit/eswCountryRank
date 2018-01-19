# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:00:27 2018

@author: msharma
"""
import pandas as pd
import numpy as np



def Opening_excel(FileName):
    Sheet = {}
    CountryData = {}
    xls = pd.ExcelFile(FileName)
    
    for i in xls.sheet_names:
        Sheet[i]= pd.read_excel(xls, i)
        if i == 'InternalData_2017':
            Sheet[i].set_index(['Brand','Country Code'],  inplace=True)
        else:
            Sheet[i].set_index(['Country Code'],  inplace=True)
        CountryData[i] = Sheet[i].to_dict(orient='index') 
    
    return Sheet,CountryData