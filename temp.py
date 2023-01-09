# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Import Dependecies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

#Data Collection and Analysis
big_mart_data = pd.read_csv('Train.csv')

big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)
mode_of_outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
missing_values = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[missing_values, 'Outlet_Size'] = big_mart_data.loc[missing_values, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size)
big_mart_data.isnull().sum()
print(big_mart_data['Outlet_Size'])