# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:11:08 2019

@author: Megamind
"""

#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#taking care of missing data - using average to fill missing data (can use median or mode)
from sklearn.preprocessing import Imputer
#axis 0 = mean of collumn, axis 1 = mean of rows
imputer = Imputer(missing_values=np.nan,strategy = "mean", axis = 0)
#fit conducts mean  for calculating missing data, upperbound not included
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])