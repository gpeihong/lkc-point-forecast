#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#04 Simple Forecast Combination
## This code takes the forecasts from 03IndividualModels and creates forecast combinations
## namely Mean, Median, Expanding Weighted Mean, and Rolling Weighted Mean


# In[ ]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import itertools
import statsmodels.api as sm
import os


# In[ ]:


## same as in 03Autoregression, this creates epiweeks from the dates
from epiweeks import Week, Year
from datetime import date
def create_epiweek(date):
    return Week.fromdate(date)
def create_epiweekplot(epiweek):
    epiweek = str(epiweek)
    return F'Y{epiweek[:4]}W{epiweek[4:]}'
def create_epiweek_fromstr(str):
    return Week.fromstring(str)


# In[ ]:


## Functions for generating weights

def min_max_norm_vector(x: pd.Series):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    scaler = MinMaxScaler()
    w0 = scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
    w0 = pd.Series(w0, index = x.index)
    return w0

def proportion(x):
    """ Proportion of sum
    """
    return x / np.sum(x)
    
def normalize_and_proportion(x):
    """ Min max normalization followed by proportion
    """
    return proportion(min_max_norm_vector(x))


# In[ ]:


## Weighted Mean using Bates and granger (P4) eXpanding MSE

## Generate xmse
def generate_xmse(dataset, target_var):
    y_pred = dataset.copy()
    y_xmse = y_pred[[target_var]].copy().drop(target_var, axis=1)
    window_end = y_xmse.index[0]
    df_end = y_xmse.index[-1]
    model_list = list(y_pred.loc[:, y_pred.columns != target_var].columns.values)
    while window_end != df_end + 1:
        y_pred_xmse = y_pred.loc[:window_end]
        for model in model_list:
            y_xmse.at[window_end,model+'_xmse'] = mean_squared_error(y_pred_xmse[[target_var]], y_pred_xmse[[model]])    
        window_end += 1
    return y_xmse.dropna().apply(func = lambda x: normalize_and_proportion(-x), axis = 1)


# In[ ]:


## Weighted Mean using Bates and granger (P3) current time

## Generate mse
def generate_mse(dataset, target_var):
    y_pred = dataset.copy()
    y_mse = y_pred[[target_var]].copy().drop(target_var, axis=1)
    window_end = y_mse.index[0]
    df_end = y_mse.index[-1]
    model_list = list(y_pred.loc[:, y_pred.columns != target_var].columns.values)
    while window_end != df_end + 1:
        y_pred_mse = y_pred.loc[window_end]
        for model in model_list:
            y_mse.at[window_end,model+'_mse'] = mean_squared_error(y_pred_mse[[target_var]], y_pred_mse[[model]])    
        window_end += 1
    
    return y_mse.dropna().apply(func = lambda x: normalize_and_proportion(-x), axis = 1)


# In[ ]:


## Generate combination (P1: Mean, P2: Median, P3: XMSE, P4: BG) 
def generate_pred_combi(dataset, target_var, y_xmse, y_mse):
    y_pred = dataset.copy()
    y_val = y_pred[[target_var]].copy()
    y_pred_combi = y_pred[[target_var]].copy().drop(target_var, axis=1)
    y_pred_combi['mean'] = y_pred.loc[:, y_pred.columns != target_var].mean(numeric_only = True, axis = 1)
    y_pred_combi['median'] = y_pred.loc[:, y_pred.columns != target_var].median(numeric_only = True, axis = 1)
    for epiweek in y_xmse.index:
        y_pred_combi.at[epiweek, 'mean_xmse'] = np.average(y_pred.loc[epiweek, y_pred.columns != target_var], weights = y_xmse.loc[epiweek])
    for epiweek in y_mse.index:
        y_pred_combi.at[epiweek, 'mean_BG'] = np.average(y_pred.loc[epiweek, y_pred.columns != target_var], weights = y_mse.loc[epiweek])
    
    return pd.concat([y_val, y_pred_combi], axis = 'columns')


# In[ ]:


def generate_forecast_combi(dataset, target_var, filename, xmse_weights_directory, BG_weights_directory):
    xmse_weights_directory_path = os.path.join(target_var, xmse_weights_directory)
    BG_weights_directory_path = os.path.join(target_var, BG_weights_directory)
    if not os.path.exists(xmse_weights_directory_path):
        os.makedirs(xmse_weights_directory_path)
    if not os.path.exists(BG_weights_directory_path):
        os.makedirs(BG_weights_directory_path)
        
    xmse_file = os.path.join(xmse_weights_directory_path, filename)
    BG_file = os.path.join(BG_weights_directory_path, filename)
    y_xmse = generate_xmse(dataset, target_var)
    y_mse = generate_mse(dataset, target_var)
    y_xmse.to_csv(xmse_file)
    y_mse.to_csv(BG_file)
    return generate_pred_combi(dataset, target_var, y_xmse, y_mse)


# In[ ]:


## This function finds the forecast prediction files from 03Autogression,
## and then creates prediction forecast combination outputs

def forecast_combination(target_var, pred_directory, xmse_weights_directory, BG_weights_directory):
    directory = os.path.join(target_var, pred_directory)
    for filename in os.listdir(directory):
        pred_file = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(pred_file):
            print(pred_file)
            
            y_pred = pd.read_csv(pred_file, parse_dates = [0], dayfirst = True, encoding='utf-8')
            y_pred['epiweek'] = y_pred['epiweek'].apply(create_epiweek_fromstr)
            y_pred = y_pred.set_index('epiweek')

            y_pred_combi = generate_forecast_combi(y_pred, target_var, filename, xmse_weights_directory, BG_weights_directory)
            pred_combi_path = os.path.join(target_var, 'pred_combi')
            if not os.path.exists(pred_combi_path):
                os.makedirs(pred_combi_path)
            y_pred_combi.to_csv(os.path.join(pred_combi_path, filename))

            print('done')


# In[ ]:


def full_forecast_combination(target_variables_file, pred_directory, xmse_weights_directory, BG_weights_directory):
    target_variables = []
    with open(target_variables_file, 'r') as file:
        for line in file:
            # Remove linebreak which is the last character of the string
            target_variable = line[:-1]
            # Add item to the list
            target_variables.append(target_variable)
    print(target_variables)

    for target_var in target_variables:
        forecast_combination(target_var, pred_directory, xmse_weights_directory, BG_weights_directory)
    
full_forecast_combination('target_variables.txt', 'pred', 'xmse_weights', 'BG_weights')


# In[ ]:




