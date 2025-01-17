#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#05 Second Order Regression
## This code takes the prediction forecast output from 03Autoregression and runs a second order regression
## In the first order (03IndividualModels) case: disease and weather variables were the X values for predicting Y - target variable
## In the second order case (this one): prediction forecasts from first order, i.e., Naive Forecast, Linear Regression, LASSO, Ridge, etc
## are the new X values for predicting Y the target variable. 

## During this second order regression, we run the same forecast methods, e.g. Linear, RandomForest, etc, a second time
## In the same way that in first order regression the forecast methods help us choose which disease/weather variables best predict the target variable,
## In second order regression, the forecast methods help us choose which forecast methods best predict the target variable. 

## The code's structure here is almost identical to 03.ipynb, but it takes in the inputs from 03.ipynb prediction outputs.
## Additionally, naive forecast method is omitted from second order. 


# In[ ]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import itertools
import statsmodels.api as sm
from IPython.display import clear_output
import os
from joblib import Parallel, delayed
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro
from rpy2.robjects import globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
from rpy2.robjects.conversion import localconverter


# In[ ]:


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


def create_X_and_y(dataset, target_var):
    X_and_y = dataset.copy()
    return X_and_y.drop(target_var, axis = 1), X_and_y[[target_var]]


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

def regression_without_naive(X_dataset, y_dataset, window_start, window_end, y_pred, y_params, aenet_weights, test_length):
    count = 0
    df_end = X_dataset.index[-1]
    while window_end != df_end:
        X = X_dataset.copy()
        y = y_dataset.copy()
        # Note: .loc is end-inclusive    
        X_train = X.loc[window_start:window_end]
        #print(X_train.info())
        ## values.ravel() converts y_train to numpy array for compatibility with models
        y_train = y.loc[window_start:window_end]
        #print(len(y_train))
        ## double square brackets so X_test is extracted as a pandas df instead of series
        X_test = X.loc[[window_end+1]]
        X_test_beofre_norm = X_test    # This is for P5 (3) 
        #print(X_test)
        y_test = y.loc[window_end+1]
        
        ## To implement P5 (3), reconstruct the dataset.
        X_new_list = []
        for i, mod in enumerate(X.columns):
            if i == 0:
                y_new = pd.DataFrame(y.values - X[[X.columns[0]]].values, columns=y.columns)
                y_new.index = y.index
            else:
                column_new = pd.DataFrame(X[[X.columns[i]]].values - X[[X.columns[0]]].values, columns=[mod])
                column_new.index = X.index
                X_new_list.append(column_new)

        X_new = pd.concat(X_new_list, axis=1)
        
        X_new_train = X_new.loc[window_start:window_end]
        #print(X_train.info())
        ## values.ravel() converts y_train to numpy array for compatibility with models
        y_new_train = y_new.loc[window_start:window_end].values.ravel()
        #print(len(y_train))
        ## double square brackets so X_test is extracted as a pandas df instead of series
        X_new_test = X_new.loc[[window_end+1]]
        #print(X_test)
        y_new_test = y_new.loc[window_end+1]
    
        ## Scaling
        scaler = StandardScaler()
        ## .fit_transform stores the scaling parameters (fit), and transforms the training set
        X_train = scaler.fit_transform(X_train)
        ## .transform takes the previously stored scaling parameters to transform the test set
        ## Therefore, test set is transformed based on the training set parameters
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train)
        X_train.columns = X.columns
        X_test = pd.DataFrame(X_test)
        X_test.columns = X.columns

        ## Scaling for X_new (对于X_new, 我保留了标准化后的np.array格式)
        scaler = StandardScaler()
        ## .fit_transform stores the scaling parameters (fit), and transforms the training set
        X_new_train = scaler.fit_transform(X_new_train)
        ## .transform takes the previously stored scaling parameters to transform the test set
        ## Therefore, test set is transformed based on the training set parameters
        X_new_test = scaler.transform(X_new_test)
        
        ## evaluate variance
    
        ## Naive Forecast N/A for second order regression
        # add the [0] to extract as float, and not as series
        #y_pred.at[window_end+1, 'naive'] = naive.loc[window_end+1][0]
        
        ## P5 (1): Linear Regression Model
        linreg_model_1 = LinearRegression()
        # Fit the model to the training data
        linreg_model_1.fit(X_train, y_train)
        # Make predictions and store
        y_pred.at[window_end+1, 'linreg_c_1'] = linreg_model_1.predict(X_test)
        
        ## P5 (2): Linear Regression Model with no constraints on coef and no intercept
        linreg_model_2 = LinearRegression(fit_intercept=False)
        # Fit the model to the training data
        linreg_model_2.fit(X_train, y_train)
        # Make predictions and store
        y_pred.at[window_end+1, 'linreg_c_2'] = linreg_model_2.predict(X_test)
        
        ## P5 (3): Linear Regression Model with all coefs sum to 1
        linreg_model_3 = LinearRegression()
        # Fit the model to the training data
        linreg_model_3.fit(X_new_train, y_new_train)
        # Make predictions and store
        y_pred.at[window_end+1, 'linreg_c_3'] = linreg_model_3.predict(X_new_test) + X_test_beofre_norm.iloc[0, 0]


#         ## Implement cross-validation split
#         tscv = TimeSeriesSplit(n_splits = 5)

#         ## ElasticNet Model
#         elasticnet_cv = ElasticNetCV(cv = tscv, max_iter = 100000)
#         elasticnet_cv.fit(X_train, y_train)
    
#         # Create the ElasticNet model with the optimal l1 and alpha values
#         elasticnet_model = ElasticNet(alpha = elasticnet_cv.alpha_, l1_ratio = elasticnet_cv.l1_ratio_)
#         elasticnet_model.fit(X_train, y_train)
#         y_pred.at[window_end+1, 'elasticnet_2'] = elasticnet_model.predict(X_test)
#         y_params.at[window_end+1, 'elasticnet_2_alpha'] = elasticnet_cv.alpha_
#         y_params.at[window_end+1, 'elasticnet_2_l1ratio'] = elasticnet_cv.l1_ratio_
        
        
        ## P6: Adaptive ElasticNet Model
        anent_model = LinearRegression() # first fit a normal lm
        anent_model.fit(X_train, y_train)
        
        pandas2ri.activate()

        glmnet = importr('glmnet')
        base = importr('base')
        base.set_seed(123)

        X_r_train = pandas2ri.py2rpy(X_train)
        y_r_train = pandas2ri.py2rpy(y_train)

        # Convert to matrices in R
        ro.r.assign('X_r_train', X_r_train)
        ro.r.assign('y_r_train', y_r_train)
        ro.r('X_r_train <- as.matrix(X_r_train)')
        ro.r('y_r_train <- as.matrix(y_r_train)')

        # Fit a linear model (without intercept, adjust according to your needs)
        ro.r('model <- lm(y_r_train ~ X_r_train)')  # Adjusted to omit the intercept

        # Extract coefficients
        ro.r('coefficients <- as.numeric(coef(model))[-1]')  # Adjust if intercept is included

        # Perform adaptive LASSO with cross-validation
        ro.r('''
        aenet_cv <- cv.glmnet(x = X_r_train, y = y_r_train,
                                type.measure = "mse",
                                nfold = 10,
                                alpha = 0.5,
                                penalty.factor = 1 / (abs(coefficients) + 1e-5),
                                keep = TRUE)
        ''')

        # Extract best coefficients
        ro.r('best_aenet_coef <- coef(aenet_cv, s = aenet_cv$lambda.min)')
        ro.r('best_aenet_coef <- as.numeric(best_aenet_coef)')
        ro.r('penalty_factor <- as.numeric(1 / (abs(coefficients) + 1e-5))')
        ro.r('aenet_lambda <- as.numeric(aenet_cv$lambda.min)')

        # Retrieve the best coefficients back into Python
        with localconverter(ro.default_converter + pandas2ri.converter):
            best_aenet_coef = ro.conversion.rpy2py(ro.globalenv['best_aenet_coef'])
            penalty_factor = ro.conversion.rpy2py(ro.globalenv['penalty_factor'])
            aenet_lambda = ro.conversion.rpy2py(ro.globalenv['aenet_lambda'])
        
        
        anent_model.coef_ = best_aenet_coef[1:].reshape(1, X_train.shape[1])
        anent_model.intercept_ = np.array([best_aenet_coef[0]])
        y_pred.at[window_end+1, 'aenet_2'] = anent_model.predict(X_test)
        
        # store the weights (penalty factors) and lambda
        for i, mod in enumerate(X_train.columns):
            y_params.at[window_end+1, f'pf_{mod}'] = penalty_factor[i]
            aenet_weights.at[window_end+1, f'{mod}_scaled_coef'] = (anent_model.coef_[0][i] / np.abs(anent_model.coef_[0]).sum()) + 1e-5 # rescale the coefs of aenet
        y_params.at[window_end+1, 'aenet_2_lambda'] = aenet_lambda
        
        
        ## P7: Random Forest
        randomforest_model = RandomForestRegressor(n_estimators = 1000, max_features = 'sqrt', random_state = 18)
        randomforest_model.fit(X_train, y_train)
        y_pred.at[window_end+1, 'randomforest_2'] = randomforest_model.predict(X_test)
    
    
        
        ##
        #keep track of model progress, every number of weeks
        tracking_interval = 5
        if window_end.weektuple()[1] % tracking_interval == 0:
            print(F'done with {window_end+1}; {count} out of {test_length}')
        
        ## Implement expanding window
        #window_start = window_start+1 (only for rolling window)
        window_end += 1
        count += 1

    print(F'The last epiweek to be predicted is: {window_end}')
    print(F'The total number of predicted epiweeks is: {count}')


# In[ ]:


def second_order_regression(dataset, target_var, window_perc):
    #print('Running for lag '+str(lag)+' step '+str(step))

    #no naive for second order regression
    #naive = create_naive(dataset, target_var)
    #print(naive.info())
    
    #lagged_dataset = create_lagged_dataset(dataset, lag)
    
    X, y = create_X_and_y(dataset, target_var)
    print(X.info())
    print(y.info())
    
    window_start, window_end = create_window(X, window_perc)
    print(F'The first epiweek to be predicted is: {window_end+1}')
    
    y_pred = create_output_dataset(y, window_end)
    y_params = create_output_dataset(y, window_end)
    aenet_weights = create_output_dataset(y, window_end)
    
    train_length = len(X.loc[window_start:window_end])
    print(F'The initial training dataset length is: {train_length}')
    test_length = len(X.loc[window_end+1:])
    print(F'The initial testing dataset length is: {test_length}')

    regression_without_naive(X, y, window_start, window_end, y_pred, y_params, aenet_weights, test_length)
    #print('Completed for lag '+str(lag)+' step '+str(step))
    clear_output(wait=False)
    return y_pred, y_params, aenet_weights.drop(columns=[target_var])


# In[ ]:


def run_second_order_regression(target_var, pred_directory):
    directory = os.path.join(target_var, pred_directory)
    for filename in os.listdir(directory):
        pred_file = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(pred_file):
            print(pred_file)
            
            y_pred = pd.read_csv(pred_file, parse_dates = [0], dayfirst = True)
            y_pred['epiweek'] = y_pred['epiweek'].apply(create_epiweek_fromstr)
            y_pred = y_pred.set_index('epiweek')
            y_pred_2, y_params_2, aenet_scaled_coef = second_order_regression(y_pred, target_var, 0.7)


            pred_2_path = os.path.join(target_var, 'pred_2')
            if not os.path.exists(pred_2_path):
                os.makedirs(pred_2_path)
            y_pred_2.to_csv(os.path.join(pred_2_path, filename))

            params_2_path = os.path.join(target_var, 'params_2')
            if not os.path.exists(params_2_path):
                os.makedirs(params_2_path)
            y_params_2.to_csv(os.path.join(params_2_path, filename))
            
            aenet_weights_path = os.path.join(target_var, 'aenet_weights')
            if not os.path.exists(aenet_weights_path):
                os.makedirs(aenet_weights_path)
            aenet_scaled_coef.to_csv(os.path.join(aenet_weights_path, filename))

            print('done')


# In[ ]:


def full_second_order_regression(target_variables_file, pred_directory):
    target_variables = []
    with open(target_variables_file, 'r') as file:
        for line in file:
            # Remove linebreak which is the last character of the string
            target_variable = line[:-1]
            # Add item to the list
            target_variables.append(target_variable)
    print(target_variables)

    Parallel(n_jobs=-2, verbose=51)(delayed(run_second_order_regression)(target_var, pred_directory) for target_var in target_variables)
    
full_second_order_regression('target_variables.txt', 'pred')


# In[ ]:




