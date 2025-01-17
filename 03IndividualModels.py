#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 03 Individual forecasting models


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
weatherclimateED['Date'] = pd.to_datetime(weatherclimateED['Date'])
weatherclimateED['epiweek'] = weatherclimateED['Date'].apply(create_epiweek)
weatherclimateED = weatherclimateED.set_index('epiweek')
weatherclimateED = weatherclimateED.iloc[:, np.r_[30:32, 33:39, 40, 42 , 45:47, 49:51,  52:54, 1:6, 8:15]]
weatherclimateED.info()


# In[ ]:


## This function takes the full dataset and creates an initial dataset with the specified range
## also returns the name of the target variable for creation of the initial dataset
## note disease_var here is an integer based off the column number
def create_initial_dataset(dataset, disease_var: int):
    explore_df = dataset.copy()
    range_start = Week(2009,1)
    range_end = Week (2018,52)
    explore_df = explore_df.loc[range_start:range_end]
    target_var = explore_df.columns.values.tolist()[disease_var]

    if not os.path.exists(target_var):
        os.makedirs(target_var)
    path = os.path.join(target_var, F'initial_dataset.csv')
    
    explore_df.to_csv(path)
    #explore_df1 is pure AR and explore_df2 is with environmetal vairables
    explore_df_1 = explore_df[[target_var]] 
    explore_df_2 = pd.merge(explore_df[[target_var]], explore_df[explore_df.columns[16:28].to_list()], on='epiweek')
#     explore_df_pure = explore_df.drop(columns=target_var)
    return explore_df, explore_df_1, explore_df_2, target_var


# In[ ]:


def create_naive(dataset, step, target_var):
    naive = dataset.copy()
    naive = naive[[target_var]].shift(step)
    return naive.dropna()


# In[ ]:


def create_history_mean(dataset, lag, step, target_var):
    origin_history_mean = dataset.copy()
    history_mean = origin_history_mean[[target_var]].shift(step)
    for i in range(step + 1, step + lag):
        history_mean += origin_history_mean.shift(i)
    return history_mean.dropna() / lag


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

def errors(model, model_name, X_train, y_train, errors_path, filename, pred_train_path):
    if model_name == 'naive':
        y_train_errors = pd.DataFrame(X_train - y_train)
        y_pred_train = pd.DataFrame(X_train)
    else:
        y_train_errors = pd.DataFrame(model.predict(X_train) - y_train)
        y_pred_train = pd.DataFrame(model.predict(X_train))
    #print(y_train_errors)
    #y_train_errors will contain list of errors in training set from window_start to window_end
    #right before window_end+1 in the filename
    errors_path = os.path.join(errors_path, model_name)
    if not os.path.exists(errors_path):
        os.makedirs(errors_path)
    y_train_errors_path = os.path.join(errors_path, F'{filename}.csv')
    y_train_errors.to_csv(y_train_errors_path)

    pred_train_path = os.path.join(pred_train_path, model_name)
    if not os.path.exists(pred_train_path):
        os.makedirs(pred_train_path)
    y_pred_train_path = os.path.join(pred_train_path, F'{filename}.csv')
    y_pred_train.to_csv(y_pred_train_path)

def coefs(model, coefs_path, filename):
    coefs_path

## This function runs the first order regression for the target disease, for one specified lag and step

def regression_with_naive(X_dataset, y_dataset, X_dataset_1, y_dataset_1, X_dataset_2, y_dataset_2, window_start, window_end, y_pred, y_params, errors_path, test_length, naive, history_mean, target_var, y_ridge, y_lasso, pred_train_path, lag):
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

        
        ## data processing for pure AR 
        X_1 = X_dataset_1.copy()
        y_1 = y_dataset_1.copy()
        # Note: .loc is end-inclusive    
        X_train_1 = X_1.loc[window_start:window_end]
        y_train_1 = y_1.loc[window_start:window_end]
        ## double square brackets so X_test is extracted as a pandas df instead of series
        X_test_1 = X_1.loc[[window_end+1]]
        y_test_1 = y_1.loc[window_end+1]
    
        ## Scaling
        scaler = StandardScaler()
        ## .fit_transform stores the scaling parameters (fit), and transforms the training set
        X_train_1 = scaler.fit_transform(X_train_1)
        ## .transform takes the previously stored scaling parameters to transform the test set
        ## Therefore, test set is transformed based on the training set parameters
        X_test_1 = scaler.transform(X_test_1)
        # For pure factor model, I use dataframe and not array
        X_train_1 = pd.DataFrame(X_train_1)
        X_train_1.columns = X_1.columns
        X_test_1 = pd.DataFrame(X_test_1)
        X_test_1.columns = X_1.columns
        
        ## data processing for AR with environmental variables
        X_2 = X_dataset_2.copy()
        y_2 = y_dataset_2.copy()
        # Note: .loc is end-inclusive    
        X_train_2 = X_2.loc[window_start:window_end]
        y_train_2 = y_2.loc[window_start:window_end]
        ## double square brackets so X_test is extracted as a pandas df instead of series
        X_test_2 = X_2.loc[[window_end+1]]
        y_test_2 = y_2.loc[window_end+1]
    
        ## Scaling
        scaler = StandardScaler()
        ## .fit_transform stores the scaling parameters (fit), and transforms the training set
        X_train_2 = scaler.fit_transform(X_train_2)
        ## .transform takes the previously stored scaling parameters to transform the test set
        ## Therefore, test set is transformed based on the training set parameters
        X_test_2 = scaler.transform(X_test_2)
    
        ## evaluate variance
        
        ## Implement cross-validation split
        tscv = TimeSeriesSplit(n_splits = 5)
        
    
        ## 1. Naive Forecast
        # add the [0] to extract as float, and not as series
        y_pred.at[window_end+1, 'naive'] = naive.loc[window_end+1][0]

#         errors(naive, 'naive', naive.loc[window_start:window_end].values.ravel(), y_train, errors_path, window_end+1, pred_train_path)
        
        ## 2. Historical mean (rolling window = lag)
        y_pred.at[window_end+1, 'historymean'] = history_mean.loc[window_end+1][0]

        ## 3. Pure AR
        ar_pure = LinearRegression()
        ar_pure.fit(X_train_1, y_train_1)
        # Make predictions and store
        y_pred.at[window_end+1, 'ar_pure'] = ar_pure.predict(X_test_1)
        
        ## 4. AR only with environmental variables
        ar_env = LinearRegression()
        ar_env.fit(X_train_2, y_train_2)
        # Make predictions and store
        y_pred.at[window_end+1, 'ar_env'] = ar_env.predict(X_test_2)
        
        ## 5. AR with all variables
        ar_all = LinearRegression()
        ar_all.fit(X_train, y_train)
        # Make predictions and store
        y_pred.at[window_end+1, 'ar_all'] = ar_all.predict(X_test)

#         errors(linreg_model, 'linreg', X_train, y_train, errors_path, window_end+1, pred_train_path)
        
    
        ## 6. Ridge model
        ridge_cv = RidgeCV(cv = tscv)
        ridge_cv.fit(X_train, y_train)
    
        ridge_model = Ridge(alpha = ridge_cv.alpha_)
        ridge_model.fit(X_train, y_train)
        
        y_pred.at[window_end+1, 'ridge'] = ridge_model.predict(X_test)
        y_params.at[window_end+1, 'ridge_alpha'] = ridge_cv.alpha_


#         alpha = ridge_cv.alpha_
#         ridge_edf = 0
#         for d in ridge_model.coef_:
#             ridge_edf += d**2/(d**2+alpha)
        
#         y_ridge.at[window_end+1, 'ridge_edf'] = ridge_edf
        
#         errors(ridge_model, 'ridge', X_train, y_train, errors_path, window_end+1, pred_train_path)

        
        ## 7. Lasso Model
        lasso_cv = LassoCV(cv = tscv, random_state = 18, max_iter = 100000)
        lasso_cv.fit(X_train, y_train)
        
        # Create the Lasso model with the optimal alpha value
        lasso_model = Lasso(alpha = lasso_cv.alpha_)
        lasso_model.fit(X_train, y_train)
        y_pred.at[window_end+1, 'lasso'] = lasso_model.predict(X_test)
        y_params.at[window_end+1, 'lasso_alpha'] = lasso_cv.alpha_
        y_params.at[window_end+1, 'lasso_n_iter'] = lasso_cv.n_iter_
        
#        y_lasso.at[window_end+1, 'lasso_edf'] = np.count_nonzero(lasso_model.coef_)

#         errors(lasso_model, 'lasso', X_train, y_train, errors_path, window_end+1, pred_train_path)
        
        ## 8. Adaptive Lasso regression
        linear_reg = LinearRegression()
        linear_reg.fit(X_train, y_train)
        initial_coef = linear_reg.coef_
        # Calculate weights for the adaptive Lasso
        weights = 1 / (np.abs(initial_coef) + 1e-5)
        X_train_weighted = X_train / weights
        
        lasso_adaptive = Lasso(alpha = lasso_cv.alpha_)
        lasso_adaptive.fit(X_train_weighted, y_train)
        
        lasso_adaptive.coef_ = lasso_adaptive.coef_ / weights
        y_pred.at[window_end+1, 'alasso'] = lasso_adaptive.predict(X_test)
        
        
        ## 9. Group Lasso regression
        group_sizes = [8 for i in range(int(X_train.shape[1]/lag))]
        groups = np.concatenate(
            [size * [i] for i, size in enumerate(group_sizes)]
        ).reshape(-1, 1)
        
        lambda_values = 10.0 ** np.arange(-1, 1, 0.3)  # Adjust according to your needs
        alpha_values = np.arange(0, 1, 0.4)

        # Placeholder for storing error for each lambda value
        errors = np.zeros((len(lambda_values), len(alpha_values)))

        # Loop over each lambda value
        for i, lambda_val in enumerate(lambda_values):
            for j, alpha_val in enumerate(alpha_values):
                temp_errors = []
                for train_index_cv, test_index_cv in tscv.split(X_train):
                    X_train_cv, X_test_cv = X_train.iloc[train_index_cv], X_train.iloc[test_index_cv]
                    y_train_cv, y_test_cv = y_train.iloc[train_index_cv], y_train.iloc[test_index_cv]

                    # Initialize and fit the GroupLasso model
                    model = GroupLasso(groups=groups, group_reg=lambda_val, l1_reg=alpha_val, random_state=18, scale_reg="inverse_group_size", fit_intercept=True, n_iter=100000, supress_warning=True)
                    model.fit(X_train_cv, y_train_cv)

                    # Predict and calculate MSE for this fold
                    y_pred_cv = model.predict(X_test_cv)
                    mse = mean_squared_error(y_test_cv, y_pred_cv)
                    temp_errors.append(mse)

                # Average MSE across all folds for this lambda
                errors[i, j] = np.mean(temp_errors)

        # Find the lambda value and alpha value with the lowest error
        min_error_idx = np.unravel_index(errors.argmin(), errors.shape)
        best_lambda = lambda_values[min_error_idx[0]]
        best_alpha = alpha_values[min_error_idx[1]]
        
        # Create the sgl model with the optimal lambda and alpha value
        sgl_model = GroupLasso(groups=groups, group_reg=best_lambda, l1_reg=best_alpha, random_state=18, scale_reg="inverse_group_size", fit_intercept=True, n_iter=100000, supress_warning=True)
        sgl_model.fit(X_train, y_train)
        y_pred.at[window_end+1, 'sgl'] = sgl_model.predict(X_test)
        y_params.at[window_end+1, 'sgl_lambda'] = best_lambda
        y_params.at[window_end+1, 'sgl_alpha'] = best_alpha
        
        
        ## 10. ElasticNet Model
        elasticnet_cv = ElasticNetCV(cv = tscv, max_iter = 100000)
        elasticnet_cv.fit(X_train, y_train)
    
        # Create the ElasticNet model with the optimal l1 and alpha values
        elasticnet_model = ElasticNet(alpha = elasticnet_cv.alpha_, l1_ratio = elasticnet_cv.l1_ratio_)
        elasticnet_model.fit(X_train, y_train)
        y_pred.at[window_end+1, 'elasticnet'] = elasticnet_model.predict(X_test)
        
        y_params.at[window_end+1, 'elasticnet_alpha'] = elasticnet_cv.alpha_
        y_params.at[window_end+1, 'elasticnet_l1ratio'] = elasticnet_cv.l1_ratio_

#         errors(elasticnet_model, 'elasticnet', X_train, y_train, errors_path, window_end+1, pred_train_path)
        
        ## 11. Adaptive ElasticNet Model
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

        # Retrieve the best coefficients back into Python
        with localconverter(ro.default_converter + pandas2ri.converter):
            best_aenet_coef = ro.conversion.rpy2py(ro.globalenv['best_aenet_coef'])
        
        anent_model.coef_ = best_aenet_coef[1:].reshape(1, X_train.shape[1])
        anent_model.intercept_ = np.array([best_aenet_coef[0]])
        y_pred.at[window_end+1, 'aenet'] = anent_model.predict(X_test)
        
        ## 12. Pure factor model
        print(X_train)
        remove_names = []
        for name in X_train.columns:
            if name[0:-3] == target_var:
                remove_names.append(name)

        X_train_pure = X_train.drop(columns=remove_names)
        X_test_pure = X_test.drop(columns=remove_names)
        pca = PCA()
        pca.fit(X_train_pure)
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
          #to explain more than 85% of the variance
        num_components = np.where(cumulative_variance_ratio >= 0.85)[0][0] + 1 
        pca_new = PCA(n_components=num_components)
        X_train_pca = pca_new.fit_transform(X_train_pure)
        X_train_pca = pd.DataFrame(X_train_pca)
        X_train_pca.columns = X_train_pca.columns.astype(str)
        X_train_pca = pd.merge(X_train_1, X_train_pca, left_index=True, right_index=True)

        X_test_pca = pca_new.transform(X_test_pure)
        X_test_pca = pd.DataFrame(X_test_pca)
        X_test_pca.columns = X_test_pca.columns.astype(str)
        X_test_pca = pd.merge(X_test_1, X_test_pca, left_index=True, right_index=True)
        
        pure_factor_model = LinearRegression()
        pure_factor_model.fit(X_train_pca, y_train)
        y_pred.at[window_end+1, 'purefactor'] = pure_factor_model.predict(X_test_pca)
        
        ## 13. Random Forest
        randomforest_model = RandomForestRegressor(n_estimators = 1000, max_features = 'sqrt', random_state = 18)
        randomforest_model.fit(X_train, y_train)
        y_pred.at[window_end+1, 'randomforest'] = randomforest_model.predict(X_test)
#         errors(randomforest_model, 'randomforest', X_train, y_train, errors_path, window_end+1, pred_train_path)
    
    
        ## 14. KNN
        knn_model = KNeighborsRegressor() #  default parameters
        knn_model.fit(X_train, y_train)
        y_pred.at[window_end+1, 'knn'] = knn_model.predict(X_test)
#         errors(knn_model, 'knn', X_train, y_train, errors_path, window_end+1, pred_train_path)


        ## 15. XGBoost
        xgboost_model = xgb.XGBRegressor(n_estimators=1000, random_state=18)
        xgboost_model.fit(X_train, y_train)
        y_pred.at[window_end+1, 'xgboost'] = xgboost_model.predict(X_test)
#         errors(gradientboost_model, 'gradientboost', X_train, y_train, errors_path, window_end+1, pred_train_path)


        ## 16. LightGBM
        lightgbm_model = lgb.LGBMRegressor(objective='regression', n_estimators= 1000, random_state=18, verbosity=-1)
        lightgbm_model.fit(X_train, y_train)
        y_pred.at[window_end+1, 'lightgbm'] = lightgbm_model.predict(X_test)
#         errors(gradientboost_model, 'gradientboost', X_train, y_train, errors_path, window_end+1, pred_train_path)
        
    
        ##
        #keep track of model progress, every number of weeks
        tracking_interval = 5
        if window_end.weektuple()[1] % tracking_interval == 0:
            print(F'{target_var} done with {window_end+1}; {count} out of {test_length}')
            
        ## Implement expanding window
        #window_start = window_start+1 (only for rolling window)
        window_end += 1
        count += 1

    print(F'The last epiweek for {target_var} to be predicted is: {window_end}')
    print(F'The total number of predicted epiweeks for {target_var} is: {count}')


# In[ ]:


## This function sets up the first order regression for the target disease, for one specified lag and step

def run_first_order_regression(dataset, dataset_1, dataset_2, lag, step, target_var, window_perc):
    print(F'Running first order regression for {target_var} lag {lag} step {step}')
    
    naive = create_naive(dataset, step, target_var)
    history_mean = create_history_mean(dataset, lag, step, target_var)
    
    lagged_dataset = create_lagged_dataset(dataset, lag, target_var)
    lagged_dataset_1 = create_lagged_dataset(dataset_1, lag, target_var)
    lagged_dataset_2 = create_lagged_dataset(dataset_2, lag, target_var)
    
    X, y = create_stepped_dataset(lagged_dataset, step, target_var)
    X_1, y_1 = create_stepped_dataset(lagged_dataset_1, step, target_var)
    X_2, y_2 = create_stepped_dataset(lagged_dataset_2, step, target_var)
    
    window_start, window_end = create_window(X, window_perc)

    print(F'The first epiweek to be predicted for {target_var} lag {lag} step {step} is: {window_end+1}')

    ## this is to get naive forecast in the initial trainset for MASE calculation
    #### naive forecast and other models have different mechanisms, so the data length is different
    if naive.index[0] <= window_start:
        y_pred_train_naive = dataset.copy()[[target_var]].loc[window_start:window_end]
        y_pred_train_naive.loc[:, 'naive_for_mase'] = np.array(naive.loc[window_start:window_end])
    else:
        y_pred_train_naive = dataset.copy()[[target_var]].loc[naive.index[0]:window_end]
        y_pred_train_naive.loc[:, 'naive_for_mase'] = np.array(naive.loc[:window_end])
    pred_train_naive_path = os.path.join(target_var, 'pred_train_naive')
    if not os.path.exists(pred_train_naive_path):
        os.makedirs(pred_train_naive_path)
    pred_train_naive_path = os.path.join(pred_train_naive_path, F'L{lag}_S{step}.csv')
    y_pred_train_naive.to_csv(pred_train_naive_path)
    
    
    y_pred = create_output_dataset(y, window_end)
    y_params = create_output_dataset(y, window_end)
    y_ridge = create_output_dataset(y, window_end)
    y_lasso = create_output_dataset(y, window_end)

    train_length = len(X.loc[window_start:window_end])
    print(F'The initial training dataset length for {target_var} lag {lag} step {step} is: {train_length}')


    test_length = len(X.loc[window_end+1:])
    print(F'The initial testing dataset length for {target_var} lag {lag} step {step} is: {test_length}')

    errors_path = os.path.join(target_var, 'errors', F'L{lag}_S{step}')
    pred_train_path = os.path.join(target_var, 'pred_train', F'L{lag}_S{step}')
    
    if not os.path.exists(errors_path):
        os.makedirs(errors_path)
    if not os.path.exists(pred_train_path):
        os.makedirs(pred_train_path)
        
    regression_with_naive(X, y, X_1, y_1, X_2, y_2, window_start, window_end, y_pred, y_params, errors_path, test_length, naive, history_mean, target_var, y_ridge, y_lasso, pred_train_path, lag)

    pred_path = os.path.join(target_var, 'pred')
    params_path = os.path.join(target_var, 'params')
    
    ridge_path = os.path.join(target_var, 'ridge_param')
    lasso_path = os.path.join(target_var, 'lasso_param')

    '''
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    if not os.path.exists(params_path):
        os.makedirs(params_path)
    '''
    if not os.path.exists(ridge_path):
        os.makedirs(ridge_path)
    if not os.path.exists(lasso_path):
        os.makedirs(lasso_path)

    pred_path = os.path.join(pred_path, F'L{lag}_S{step}.csv')
    params_path = os.path.join(params_path, F'L{lag}_S{step}.csv')
    ridge_path = os.path.join(ridge_path, F'L{lag}_S{step}.csv')
    lasso_path = os.path.join(lasso_path, F'L{lag}_S{step}.csv')    


    
    y_pred.to_csv(pred_path)
    y_params.to_csv(params_path)

    y_ridge.to_csv(ridge_path)
    y_lasso.to_csv(lasso_path)
    

    print(F'Completed for {target_var} lag {lag} step {step}')
    clear_output(wait=False)
    return y_ridge, y_lasso


# In[ ]:


## This function runs the regression for one disease, for all lags and steps, hence the for loop

def run_disease_regression(dataset, disease_var, lag_start, lag_end, step_start, step_end):
    
    ## Note how the integer disease_var is input into this function, and then
    ## the string target_var is returned for the remaining functions
    explore_df, explore_df_1, explore_df_2, target_var = create_initial_dataset(dataset, disease_var)

    with open("target_variables.txt") as target_variables_file:
        if target_var not in target_variables_file.read():
            with open("target_variables.txt", 'a') as target_variables_file:
                target_variables_file.write(F'{target_var}\n')
    
    ## run the first order regression for all lags and steps for this target variable
    print(F'Running regression for {target_var}')
    for lag in range(lag_start, lag_end):
        for step in range(step_start, step_end):
            run_first_order_regression(explore_df, explore_df_1, explore_df_2, lag = lag, step = step, target_var = target_var, window_perc = 0.7)


# In[ ]:


## x in range (0,16) represents the 16 diseases that are the target variables. However, for this function we input them as integers
## the create_initial_dataset function will convert the integer format to string format
## Using parallel, each disease can be run on one computer core
Parallel(n_jobs=-2, verbose=51)(delayed(run_disease_regression)(weatherclimateED, x, 8, 9, 1, 13) for x in range(0,16))
#run_full_regression(weatherclimateED, range(0,16), 8, 9, 1, 9)


# In[ ]:




