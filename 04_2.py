#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 04_2 CSR and Random projections


# In[ ]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
weatherclimateED = pd.read_csv('weatherclimateED.csv', parse_dates = [0], dayfirst = True)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import itertools
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')
from IPython.display import clear_output
import os
from joblib import Parallel, delayed
from itertools import combinations
import random


# In[ ]:


## Epiweeks Module converts dates to CDC Epiweek format
## Further documentation on https://pypi.org/project/epiweeks/
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


## This section creates a full complete dataset that includes all the variables of interest that will be used
## iloc function selects the relevant variables of interest based on column number
## Problematic weather columns (i.e. don't select!): 6, 16, 17, 19, 20
## Disease columns excluded due to limited dataset: 21:24, 25

weatherclimateED['epiweek'] = weatherclimateED['Date'].apply(create_epiweek)
weatherclimateED = weatherclimateED.set_index('epiweek')
weatherclimateED = weatherclimateED.iloc[:, np.r_[30:32, 33:39, 40, 42 , 45:47, 49:51,  52:54, 1:6, 8:15]]
weatherclimateED.info()


# In[ ]:


def keep_by_index(lst, indices):
    return [item for idx, item in enumerate(lst) if idx in indices]


# In[ ]:


## This function takes the full dataset and creates an initial dataset with the specified range
## also returns the name of the target variable for creation of the initial dataset
## note disease_var here is an integer based off the column number
## we call the number of exogenous predictors, P. Pick p = 1, 2, ..., 5
def create_initial_dataset(dataset, disease_var: int, P, p):
    target_disease_order = disease_var # disease_var is a number
    explore_df = dataset.copy()
    range_start = Week(2009,1)
    range_end = Week (2018,52)
    explore_df = explore_df.loc[range_start:range_end]
    target_var = explore_df.columns.values.tolist()[disease_var]
     
    ## For P8
    combs = list(combinations(range(16, 16 + P), p))
    eds_list = list(range(0, target_disease_order)) + list(range(target_disease_order+1, 16))
    combs_eds = list(combinations(eds_list, p))
    
    new_explore_dfs = []

    for comb in combs:
        for comb_eds in combs_eds:
            new_explore_df_parts = [explore_df[[explore_df.columns[target_disease_order]]]]
            for i in range(p):
                column_selected = explore_df.iloc[:, [comb[i], comb_eds[i]]]
                new_explore_df_parts.append(column_selected)
            new_explore_df = pd.concat(new_explore_df_parts, axis=1)
            new_explore_dfs.append(new_explore_df)
    
    if p > 1:
        keep_index_list = np.random.choice(len(new_explore_dfs), size=1000, replace=False)
        new_explore_dfs = keep_by_index(new_explore_dfs, keep_index_list)    
    
    return explore_df, new_explore_dfs, target_var


# In[ ]:


# Create lagged dataset
def create_lagged_dataset(dataset, lag, target_var):
    lagged_dataset = dataset.copy()
    columns_list = list(lagged_dataset.columns)
    data_join = {}
    for column in columns_list:
        if column == target_var:
            data_join[column] = lagged_dataset[column]
        for n in range(1,lag+1):
            data_join[F'{column}_L{n}'] = lagged_dataset[column].shift(n)
    lagged_dataset = pd.concat(data_join.values(), axis=1, ignore_index = True)
    lagged_dataset.columns = data_join.keys()
    return lagged_dataset.dropna()


# In[ ]:


## Step is the number of weeks ahead that we are forecasting, e.g. step=2 is 2 weeks ahead.
## Note step=1 results in no change to dataset, i.e. use generated lagged variables to forecast current. 
def create_stepped_dataset(dataset, step, target_var):
    stepped_dataset = dataset.copy()
    y = stepped_dataset[[target_var]].shift(-step+1)
    if step != 1:
        X = stepped_dataset.iloc[:-step+1, :]
    else:
        X = stepped_dataset
    return X.drop(target_var, axis = 1), y.dropna()
## So now target variable (y variable for exploration) is shifted back by 2 weeks. i.e., taking the y-value from 2 weeks later
## and setting it to the current index. So linear regression of y+2 with the current X values. X will have
## a smaller dataset with the last 2 time points removed because of the shift. 


# In[ ]:


def create_window(X, window_perc):
    return X.index[0], X.index[int(len(X)*window_perc)]
def create_output_dataset(y, window_end):
    return y.copy().loc[window_end+1:]


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from group_lasso import GroupLasso
import xgboost as xgb
import lightgbm as lgb
# from nixtlats import TimeGPT
from sklearn.decomposition import PCA
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro
from rpy2.robjects import globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
from rpy2.robjects.conversion import localconverter


def coefs(model, coefs_path, filename):
    coefs_path

## This function runs the first order regression for the target disease, for one specified lag and step

def regression_comb(X_dataset, y_dataset, window_start, window_end, y_pred, test_length, target_var, lag):
    count = 0
    df_end = X_dataset.index[-1]
    while window_end != df_end:
        X = X_dataset.copy()
        y = y_dataset.copy()
        # Note: .loc is end-inclusive    
        X_train = X.loc[window_start:window_end]
        #print(X_train.info())
        ## values.ravel() converts y_train to numpy array for compatibility with models (update: already deleted this)
        y_train = y.loc[window_start:window_end]
        #print(len(y_train))
        ## double square brackets so X_test is extracted as a pandas df instead of series
        X_test = X.loc[[window_end+1]]
        #print(X_test)
        y_test = y.loc[window_end+1]
        #print(y_test)
    
        ## Scaling
        scaler = StandardScaler()
        ## .fit_transform stores the scaling parameters (fit), and transforms the training set
        X_train = scaler.fit_transform(X_train)
        ## .transform takes the previously stored scaling parameters to transform the test set
        ## Therefore, test set is transformed based on the training set parameters
        X_test = scaler.transform(X_test)
        # For all models using all variables, I use dataframe and not array
        X_train = pd.DataFrame(X_train)
        X_train.columns = X.columns
        X_test = pd.DataFrame(X_test)
        X_test.columns = X.columns


        ## P8 CSL p = 1, 2, 3
        csl = LinearRegression()
        csl.fit(X_train, y_train)
        # Make predictions and store
        y_pred.at[window_end+1, 'CSL'] = csl.predict(X_test)

        

        ##
        #keep track of model progress, every number of weeks
        tracking_interval = 5
#         if window_end.weektuple()[1] % tracking_interval == 0:
#             print(F'{target_var} done with {window_end+1}; {count} out of {test_length}')
            
        ## Implement expanding window
        #window_start = window_start+1 (only for rolling window)
        window_end += 1
        count += 1

#     print(F'The last epiweek for {target_var} to be predicted is: {window_end}')
#     print(F'The total number of predicted epiweeks for {target_var} is: {count}')
    

def regression_comb_1(X_dataset, y_dataset, window_start, window_end, y_pred, test_length, target_var, lag):
    count = 0
    df_end = X_dataset.index[-1]
    while window_end != df_end:
        X = X_dataset.copy()
        y = y_dataset.copy()
        # Note: .loc is end-inclusive    
        X_train = X.loc[window_start:window_end]
        #print(X_train.info())
        ## values.ravel() converts y_train to numpy array for compatibility with models (update: already deleted this)
        y_train = y.loc[window_start:window_end]
        #print(len(y_train))
        ## double square brackets so X_test is extracted as a pandas df instead of series
        X_test = X.loc[[window_end+1]]
        #print(X_test)
        y_test = y.loc[window_end+1]
        #print(y_test)
    
        ## Scaling
        scaler = StandardScaler()
        ## .fit_transform stores the scaling parameters (fit), and transforms the training set
        X_train = scaler.fit_transform(X_train)
        ## .transform takes the previously stored scaling parameters to transform the test set
        ## Therefore, test set is transformed based on the training set parameters
        X_test = scaler.transform(X_test)
        # For all models using all variables, I use dataframe and not array
        X_train = pd.DataFrame(X_train)
        X_train.columns = X.columns
        X_test = pd.DataFrame(X_test)
        X_test.columns = X.columns


        ## P9 RP p = 1, 2, 3
        rp = LinearRegression()
        rp.fit(X_train, y_train)
        # Make predictions and store
        y_pred.at[window_end+1, 'RP'] = rp.predict(X_test)

        

        ##
        #keep track of model progress, every number of weeks
        tracking_interval = 5
#         if window_end.weektuple()[1] % tracking_interval == 0:
#             print(F'{target_var} done with {window_end+1}; {count} out of {test_length}')
            
        ## Implement expanding window
        #window_start = window_start+1 (only for rolling window)
        window_end += 1
        count += 1

#     print(F'The last epiweek for {target_var} to be predicted is: {window_end}')
#     print(F'The total number of predicted epiweeks for {target_var} is: {count}')


# In[ ]:


## This function sets up the first order regression for the target disease, for one specified lag and step

def run_regression_comb(dataset, new_datasets, lag, step, target_var, window_perc, p):
    print(F'Running regression combination for {target_var} lag {lag} step {step}')
    
    ## For P8
    preds = []
    for new_dataset in new_datasets:
        lagged_dataset = create_lagged_dataset(new_dataset, lag, target_var)

        X, y = create_stepped_dataset(lagged_dataset, step, target_var)

        window_start, window_end = create_window(X, window_perc)

#         print(F'The first epiweek to be predicted for {target_var} lag {lag} step {step} is: {window_end+1}')

        y_pred = create_output_dataset(y, window_end)

        train_length = len(X.loc[window_start:window_end])
#         print(F'The initial training dataset length for {target_var} lag {lag} step {step} is: {train_length}')


        test_length = len(X.loc[window_end+1:])
#         print(F'The initial testing dataset length for {target_var} lag {lag} step {step} is: {test_length}')


        regression_comb(X, y, window_start, window_end, y_pred, test_length, target_var, lag)
        preds.append(y_pred)
    average_pred = pd.concat(preds, axis=0).groupby(level=0).mean() 
    
    
    ## For P9
    lagged_dataset = create_lagged_dataset(dataset, lag, target_var)

    X, y = create_stepped_dataset(lagged_dataset, step, target_var)

    window_start, window_end = create_window(X, window_perc)
    
    train_length = len(X.loc[window_start:window_end])

    test_length = len(X.loc[window_end+1:])
    
    target_start = 0
    for count, column in enumerate(X.columns):
        if column[0:-3] == target_var:
            target_start = count
            break
    target_end = target_start + 8
    
    X_env = X[X.columns[128:].to_list()]
    X_only_target = X[X.columns[target_start:target_end].to_list()]
    eds_columns = X.columns[0:target_start].to_list() + X.columns[target_end:128].to_list()
    X_eds = X[eds_columns]
    nrow_matrix_env = np.array(X_env).shape[1]
    nrow_matrix_eds = np.array(X_eds).shape[1]
    ncol_matrix = p
    num_matrices = 100
    # np.random.seed(123)
    matrices_env = [np.random.normal(0, 1, (nrow_matrix_env, ncol_matrix)) for _ in range(num_matrices)]
    matrices_eds = [np.random.normal(0, 1, (nrow_matrix_eds, ncol_matrix)) for _ in range(num_matrices)]

    # sampling 1000 from 100 * 100
    all_combinations = [(x, y) for x in range(num_matrices) for y in range(num_matrices)]
    sampled_tuples = random.sample(all_combinations, 1000)

    new_X_sets = []
    for tuple in sampled_tuples:
        X_proj_env = pd.DataFrame(np.dot(np.array(X_env), matrices_env[tuple[0]]))
        X_proj_env.columns = X_proj_env.columns.astype(str)
        X_proj_env.index = X.index

        X_proj_eds = pd.DataFrame(np.dot(np.array(X_eds), matrices_eds[tuple[1]]))
        X_proj_eds.columns = X_proj_eds.columns.astype(str)
        X_proj_eds.index = X.index

        new_X_sets.append(pd.concat([X_only_target, X_proj_env, X_proj_eds], axis=1))
    
    preds_1 = []
    for new_X in new_X_sets:
        y_pred_1 = create_output_dataset(y, window_end)
        regression_comb_1(new_X, y, window_start, window_end, y_pred_1, test_length, target_var, lag)
        preds_1.append(y_pred_1)
    average_pred_1 = pd.concat(preds_1, axis=0).groupby(level=0).mean()
    average_pred_1.drop(columns=target_var, inplace=True)
    
    final_pred = pd.concat([average_pred, average_pred_1], axis=1)
    
    pred_path = os.path.join(target_var, f'pred_csr_rp_{p}')

    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    pred_path = os.path.join(pred_path, F'L{lag}_S{step}.csv')


    final_pred.to_csv(pred_path)
    

    print(F'Completed for {target_var} lag {lag} step {step} p {p}')
    clear_output(wait=False)


# In[ ]:


## This function runs the regression for one disease, for all lags and steps, hence the for loop

def run_disease_regression_comb(dataset, disease_var, lag_start, lag_end, step_start, step_end, P):
    
    ## Note how the integer disease_var is input into this function, and then
    ## the string target_var is returned for the remaining functions
    
    for p in range(1,4):
        explore_df, new_explore_dfs, target_var = create_initial_dataset(dataset, disease_var, P, p)

        with open("target_variables.txt") as target_variables_file:
            if target_var not in target_variables_file.read():
                with open("target_variables.txt", 'a') as target_variables_file:
                    target_variables_file.write(F'{target_var}\n')

        ## run the first order regression for all lags and steps for this target variable
        print(F'Running regression for {target_var}')
        for lag in range(lag_start, lag_end):
            for step in range(step_start, step_end):
                run_regression_comb(explore_df, new_explore_dfs, lag = lag, step = step, target_var = target_var, window_perc = 0.7, p=p)


# In[ ]:


## Main function call using Parallel
## x in range (0,16) represents the 16 diseases that are the target variables. However, for this function we input them as integers
## the create_initial_dataset function will convert the integer format to string format
## Using parallel, each disease can be run on one computer core
np.random.seed(123)
Parallel(n_jobs=-2, verbose=51)(delayed(run_disease_regression_comb)(weatherclimateED, x, 8, 9, 1, 13, 12) for x in range(0,16))
#run_full_regression(weatherclimateED, range(0,16), 8, 9, 1, 9)


# In[ ]:




