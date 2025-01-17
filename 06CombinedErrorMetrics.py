#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 06 Combined Error Metrics
## This code first combines all the predictions (from first order, forecast combination, and second order), then
## calculates the error metrics in order to compare which forecast method is best


# In[2]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import itertools
import statsmodels.api as sm
import os
from joblib import Parallel, delayed


# In[3]:


from epiweeks import Week, Year
from datetime import date
def create_epiweek(date):
    return Week.fromdate(date)
def create_epiweekplot(epiweek):
    epiweek = str(epiweek)
    return F'Y{epiweek[:4]}W{epiweek[4:]}'
def create_epiweek_fromstr(str):
    return Week.fromstring(str)


# In[4]:


## Symmetrical Mean Absolute Percentage Error
## Suitable for datasets where the actual values are small or close to 0

def smape(A, F):
    return 1/len(A) * np.sum(np.abs(F - A) / ((np.abs(A) + np.abs(F) + np.finfo(float).eps)/2))


# In[5]:


def generate_error_metrics(dataset, train_naive, target_var):
    pred = dataset.copy()
    model_list = list(pred.columns.values)
    y = pred[[target_var]]
    model_list.remove(target_var)
    
    # for MASE
    pred_train_naive = train_naive.copy()
    y_train_naive_val = pred_train_naive[[target_var]]
    naive_train_naive_val = pred_train_naive[['naive_for_mase']]
    
    ##
    error_df = pd.DataFrame()
    #print(model_list)

    for model in model_list:
        model_val = pred[[model]].dropna()
        window_start = model_val.index[0]
        window_end = model_val.index[-1]
        y_val = y.loc[window_start:window_end].copy()

        error_df.at[model, 'MSE'] = mean_squared_error(y_val, model_val)
        error_df.at[model, 'MAPE'] = mean_absolute_percentage_error(y_val, model_val)
        error_df.at[model, 'MAE'] = mean_absolute_error(y_val, model_val)
        error_df.at[model, 'SMAPE'] = smape(np.array(y_val), np.array(model_val))


        
        ## MASE: scale MAE to MAE of Naive Forecast (naive forecast window matched to model window)
        naive_val = pred[['naive']].loc[window_start:window_end].copy()
        error_df.at[model, 'MASE'] = mean_absolute_error(y_val, model_val)/mean_absolute_error(y_train_naive_val, naive_train_naive_val)

        ## Diebold-Mariano against Naive
        if model == 'naive':
            dm_stat, pvalue = 0, 0
        else:
            dm_stat, pvalue = dm_test(y_val, naive_val, model_val)
            if pvalue < 0.05:
                pvalue = 'R'
            else:
                pvalue = 'A'

        error_df.at[model, 'DM'], error_df.at[model, 'pval'] = dm_stat, pvalue

    return error_df


# In[6]:


from itertools import islice
from typing import Sequence, Callable, List, Tuple
from math import lgamma, fabs, isnan, nan, exp, log, log1p, sqrt


class InvalidParameterException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class ZeroVarianceException(ArithmeticError):
    def __init__(self, message: str):
        super().__init__(message)


def autocovariance(X: Sequence[float], k: int, mean: float) -> float:
    """
    Returns the k-lagged autocovariance for the input iterable.
    """
    return sum((a - mean) * (b - mean) for a, b in zip(islice(X, k, None), X)) / len(X)


def log_beta(a: float, b: float) -> float:
    """
    Returns the natural logarithm of the beta function computed on
    arguments `a` and `b`.
    """
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def evaluate_continuous_fraction(
    fa: Callable[[int, float], float],
    fb: Callable[[int, float], float],
    x: float,
    *,
    epsilon: float = 1e-10,
    maxiter: int = 10000,
    small: float = 1e-50
) -> float:
    """
    Evaluate a continuous fraction.
    """
    h_prev = fa(0, x)
    if fabs(h_prev < small):
        h_prev = small

    n: int = 1
    d_prev: float = 0.0
    c_prev: float = h_prev
    hn: float = h_prev

    while n < maxiter:
        a = fa(n, x)
        b = fb(n, x)

        dn = a + b * d_prev
        if fabs(dn) < small:
            dn = small

        cn = a + b / c_prev
        if fabs(cn) < small:
            cn = small

        dn = 1 / dn
        delta_n = cn * dn
        hn = h_prev * delta_n

        if fabs(delta_n - 1.0) < epsilon:
            break

        d_prev = dn
        c_prev = cn
        h_prev = hn

        n += 1

    return hn


def regularized_incomplete_beta(
    x: float, a: float, b: float, *, epsilon: float = 1e-10, maxiter: int = 10000
) -> float:
    if isnan(x) or isnan(a) or isnan(b) or x < 0 or x > 1 or a <= 0 or b <= 0:
        return nan

    if x > (a + 1) / (2 + b + a) and 1 - x <= (b + 1) / (2 + b + a):
        return 1 - regularized_incomplete_beta(
            1 - x, b, a, epsilon=epsilon, maxiter=maxiter
        )

    def fa(n: int, x: float) -> float:
        return 1.0

    def fb(n: int, x: float) -> float:
        if n % 2 == 0:
            m = n / 2.0
            return (m * (b - m) * x) / ((a + (2 * m) - 1) * (a + (2 * m)))

        m = (n - 1.0) / 2.0
        return -((a + m) * (a + b + m) * x) / ((a + (2 * m)) * (a + (2 * m) + 1.0))

    return exp(
        a * log(x) + b * log1p(-x) - log(a) - log_beta(a, b)
    ) / evaluate_continuous_fraction(fa, fb, x, epsilon=epsilon, maxiter=maxiter)


def dm_test(
    V: Sequence[float],
    P1: Sequence[float],
    P2: Sequence[float],
    *,
    loss: Callable[[float, float], float] = lambda u, v: (u - v) ** 2,
    h: int = 1,
    one_sided: bool = False,
    harvey_correction: bool = True
) -> Tuple[float, float]:
    r"""
    Performs the Diebold-Mariano test. The null hypothesis is that the two forecasts (`P1`, `P2`) have the same accuracy.

    Parameters
    ----------
    V: Sequence[float]
        The actual timeseries.

    P1: Sequence[float]
        First prediction series.

    P2: Sequence[float]
        Second prediction series.

    loss: Callable[[float, float], float]
        Loss function. At each time step of the series, each prediction is charged a loss, 
        computed as per this function. The Diebold-Mariano test is agnostic with respect to 
        the loss function, and this implementation supports arbitrarily specified (for example asymmetric) 
        functions. The two arguments are, *in this order*, the actual value and the predicted value. 
        Default is squared error (i.e. `lambda u, v: (u - v) ** 2`)

    h: int
        The forecast horizon. Default is 1.

    one_sided: bool
        If set to true, returns the p-value for a one-sided test instead of a two-sided test. Default is false.

    harvey_correcetion: bool
        If set to true, uses a modified test statistics as per Harvey, Leybourne and Newbold (1997).

    Returns
    -------
    A tuple of two values. The first is the test statistic, the second is the p-value.
    """
    if not (len(V) == len(P1) == len(P2)):
        raise InvalidParameterException(
            "Actual timeseries and prediction series must have the same length."
        )

    if h <= 0:
        raise InvalidParameterException(
            "Invalid parameter for horizon length. Must be a positive integer."
        )

    V = V.values.tolist()
    P1 = P1.values.tolist()
    P2 = P2.values.tolist()

    n = len(P1)
    mean = 0.0
    loss1 = 0.0
    loss2 = 0.0
    D: List[float] = []

    '''
    l1 += loss(v, p1)
    l2 += loss(v, p2)
    mean += l1 - l2
    '''
    for i in range(0,n):
        l1 = loss(V[i][0], P1[i][0])
        l2 = loss(V[i][0], P2[i][0])
        D.append(l1 - l2)
        mean += l1 - l2
        loss1 += l1
        loss2 += l2

    mean /= n
    
    '''
    for v, p1, p2 in zip(V, P1, P2):
        l1 = loss(v, p1)
        l2 = loss(v, p2)
        D.append(l1 - l2)
        mean += l1 - l2
        loss1 += l1
        loss2 += l2

    mean /= n
    '''
    
    V_d = 0.0
    for i in range(h):
        V_d += autocovariance(D, i, mean)
        if i == 0:
            V_d /= 2

    V_d = 2 * V_d / n

    if V_d == 0:
        raise ZeroVarianceException(
            "Variance of the DM statistic is zero. Maybe the prediction series are identical?"
        )

    if harvey_correction:
        harvey_adj = sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dmstat = harvey_adj / sqrt(V_d) * mean
    else:
        dmstat = mean / sqrt(V_d)

    pvalue = regularized_incomplete_beta(
        (n - 1) / ((n - 1) + dmstat ** 2), 0.5 * (n - 1), 0.5
    )

    if one_sided:
        # please change dmstat < 0 or dmstat > 0; also change the num_of_NE.csv in 07.ipynb 
        if dmstat > 0:
            pvalue = pvalue 
        else:
            pvalue = 1 

    return dmstat, pvalue


# In[7]:


def prepare_diebold_mariano(dataset, target_var, model_1, model_2):
    pred = dataset.copy()
    if len(pred[[model_1]].dropna()) < len(pred[[model_2]].dropna()):
        model_1_val = pred[[model_1]].dropna()
        window_start = model_1_val.index[0]
        window_end = model_1_val.index[-1]
        model_2_val = pred[[model_2]].loc[window_start:window_end].copy()
        y_val = pred[[target_var]].loc[window_start:window_end].copy()


        return model_1_val, model_2_val, y_val
    else:
        model_2_val = pred[[model_2]].dropna()
        window_start = model_2_val.index[0]
        window_end = model_2_val.index[-1]
        model_1_val = pred[[model_1]].loc[window_start:window_end].copy()
        y_val = pred[[target_var]].loc[window_start:window_end].copy()

        return model_1_val, model_2_val, y_val

def evaluate_pvalue(pvalue):
    if pvalue < 0.05:
    #non-equivalent, i.e. we reject the null hypothesis that both models have equal predictive capability
    #non-equivalence in RED
        pvalue = -1
    else:
    #pvalue > 0.05
    #equivalent, i.e. we accept the null hypothesis that both models have equal predictive capability
    #not enough evidence to show that one model predictive better than the other
    #equivalence in BLACK
        pvalue = 1
    return pvalue

def generate_diebold_mariano(dataset, target_var):
    pred = dataset.copy()
    model_list = list(pred.columns.values)
    y = pred[[target_var]]
    model_list.remove(target_var)

    diebold_mariano_dmstat_df = pd.DataFrame(index=model_list, columns=model_list)
    diebold_mariano_pvalue_df = pd.DataFrame(index=model_list, columns=model_list)
    
    for model_1 in model_list:
        for model_2 in model_list:
            if model_1 == model_2:
                dm_stat, pvalue = 0, 0
            else:
                if pd.isna(diebold_mariano_pvalue_df.loc[model_2, model_1]):
                    model_1_val, model_2_val, y_val = prepare_diebold_mariano(pred, target_var, model_1, model_2)
                    dm_stat, pvalue = dm_test(y_val, model_1_val, model_2_val, one_sided=True)
                    pvalue = evaluate_pvalue(pvalue)
                else:
                    dm_stat, pvalue = 0, 0
            diebold_mariano_dmstat_df.at[model_1, model_2], diebold_mariano_pvalue_df.at[model_1, model_2] = dm_stat, pvalue
    return diebold_mariano_dmstat_df, diebold_mariano_pvalue_df
    


# In[8]:


def combined_error_metrics(target_var, pred_directory, pred_combi_directory, pred_2_directory, csr_rp_1_directory, csr_rp_2_directory, csr_rp_3_directory, pred_train_naive_directory):
    pred_dir = os.path.join(target_var, pred_directory)
    pred_combi_dir = os.path.join(target_var, pred_combi_directory)
    pred_2_dir = os.path.join(target_var, pred_2_directory)
    csr_rp_1_dir = os.path.join(target_var, csr_rp_1_directory)
    csr_rp_2_dir = os.path.join(target_var, csr_rp_2_directory)
    csr_rp_3_dir = os.path.join(target_var, csr_rp_3_directory)
    pred_train_naive_dir = os.path.join(target_var, pred_train_naive_directory)
#     print(pred_dir)
    for filename in os.listdir(pred_dir):
        print(filename)
        pred_file = os.path.join(pred_dir, filename)
        pred_combi_file = os.path.join(pred_combi_dir, filename)
        pred_2_file = os.path.join(pred_2_dir, filename)
        csr_rp_1_file = os.path.join(csr_rp_1_dir, filename)
        csr_rp_2_file = os.path.join(csr_rp_2_dir, filename)
        csr_rp_3_file = os.path.join(csr_rp_3_dir, filename)
        pred_train_naive_file = os.path.join(pred_train_naive_dir, filename)
        # checking if it is a file
        if os.path.isfile(pred_file):
            #print(pred_file)
            y_pred = pd.read_csv(pred_file, parse_dates = [0], dayfirst = True)
            y_pred['epiweek'] = y_pred['epiweek'].apply(create_epiweek_fromstr)
            y_pred = y_pred.set_index('epiweek')
            
            y_pred_train_naive = pd.read_csv(pred_train_naive_file, parse_dates = [0], dayfirst = True)
            y_pred_train_naive['epiweek'] = y_pred_train_naive['epiweek'].apply(create_epiweek_fromstr)
            y_pred_train_naive = y_pred_train_naive.set_index('epiweek')

            y_pred_combi = pd.read_csv(pred_combi_file, parse_dates = [0], dayfirst = True)
            y_pred_combi['epiweek'] = y_pred_combi['epiweek'].apply(create_epiweek_fromstr)
            y_pred_combi = y_pred_combi.set_index('epiweek')
            y_pred_combi = y_pred_combi.drop(target_var, axis=1)

            y_all = pd.concat([y_pred, y_pred_combi], axis = 'columns')
            
            pred_all_path = os.path.join(target_var, 'pred_all')
            if not os.path.exists(pred_all_path):
                os.makedirs(pred_all_path)
            y_all.to_csv(os.path.join(pred_all_path, filename))

            error_df = generate_error_metrics(y_all, y_pred_train_naive, target_var)

            error_metrics_path = os.path.join(target_var, 'error_metrics')
            if not os.path.exists(error_metrics_path):
                os.makedirs(error_metrics_path)        
            error_df.to_csv(os.path.join(error_metrics_path, filename))

            
            diebold_mariano_dmstat_df, diebold_mariano_pvalue_df = generate_diebold_mariano(y_all, target_var)
            
            dmstat_path = os.path.join(target_var, 'dmstat')
            if not os.path.exists(dmstat_path):
                os.makedirs(dmstat_path)
            diebold_mariano_dmstat_df.to_csv(os.path.join(dmstat_path, filename))

            pvalue_path = os.path.join(target_var, 'pvalue')
            if not os.path.exists(pvalue_path):
                os.makedirs(pvalue_path)
            diebold_mariano_pvalue_df.to_csv(os.path.join(pvalue_path, filename))

            ## ADD y_full to include pred_2 and csr_rp

            y_pred_2 = pd.read_csv(pred_2_file, parse_dates = [0], dayfirst = True)
            y_pred_2['epiweek'] = y_pred_2['epiweek'].apply(create_epiweek_fromstr)
            y_pred_2 = y_pred_2.set_index('epiweek')
            y_pred_2 = y_pred_2.drop(target_var, axis=1)
            
            y_csr_rp_1 = pd.read_csv(csr_rp_1_file, parse_dates = [0], dayfirst = True)
            y_csr_rp_1['epiweek'] = y_csr_rp_1['epiweek'].apply(create_epiweek_fromstr)
            y_csr_rp_1 = y_csr_rp_1.set_index('epiweek')
            y_csr_rp_1 = y_csr_rp_1.drop(target_var, axis=1)
            # change column names of the df
            y_csr_rp_1.rename(columns={'CSL': 'CSR_1', 'RP': 'RP_1'}, inplace=True)
            
            y_csr_rp_2 = pd.read_csv(csr_rp_2_file, parse_dates = [0], dayfirst = True)
            y_csr_rp_2['epiweek'] = y_csr_rp_2['epiweek'].apply(create_epiweek_fromstr)
            y_csr_rp_2 = y_csr_rp_2.set_index('epiweek')
            y_csr_rp_2 = y_csr_rp_2.drop(target_var, axis=1)
            # change column names of the df
            y_csr_rp_2.rename(columns={'CSL': 'CSR_2', 'RP': 'RP_2'}, inplace=True)
            
            y_csr_rp_3 = pd.read_csv(csr_rp_3_file, parse_dates = [0], dayfirst = True)
            y_csr_rp_3['epiweek'] = y_csr_rp_3['epiweek'].apply(create_epiweek_fromstr)
            y_csr_rp_3 = y_csr_rp_3.set_index('epiweek')
            y_csr_rp_3 = y_csr_rp_3.drop(target_var, axis=1)
            # change column names of the df
            y_csr_rp_3.rename(columns={'CSL': 'CSR_3', 'RP': 'RP_3'}, inplace=True)

            y_full = pd.concat([y_pred, y_pred_combi, y_pred_2, y_csr_rp_1, y_csr_rp_2, y_csr_rp_3], axis = 'columns')
            y_full_30 = pd.concat([y_pred, y_pred_combi, y_pred_2, y_csr_rp_1, y_csr_rp_2, y_csr_rp_3], axis = 'columns', join='inner')
#             y_full = y_full.dropna()

            pred_full_path = os.path.join(target_var, 'pred_full')
            pred_full_30_path = os.path.join(target_var, 'pred_full_30')
            if not os.path.exists(pred_full_path):
                os.makedirs(pred_full_path)
            y_full.to_csv(os.path.join(pred_full_path, filename))
            
            if not os.path.exists(pred_full_30_path):
                os.makedirs(pred_full_30_path)
            y_full_30.to_csv(os.path.join(pred_full_30_path, filename))

            error_df_full = generate_error_metrics(y_full, y_pred_train_naive, target_var)  
            error_df_full_30 = generate_error_metrics(y_full_30, y_pred_train_naive, target_var) 

            error_metrics_full_path = os.path.join(target_var, 'error_metrics_full')
            error_metrics_full_30_path = os.path.join(target_var, 'error_metrics_full_30')
            if not os.path.exists(error_metrics_full_path):
                os.makedirs(error_metrics_full_path)        
            error_df_full.to_csv(os.path.join(error_metrics_full_path, filename))
            
            if not os.path.exists(error_metrics_full_30_path):
                os.makedirs(error_metrics_full_30_path)        
            error_df_full_30.to_csv(os.path.join(error_metrics_full_30_path, filename))

            '''
            
            diebold_mariano_dmstat_full_df, diebold_mariano_pvalue_full_df = generate_diebold_mariano(y_full, target_var)
            
            dmstat_full_path = os.path.join(target_var, 'dmstat_full')
            if not os.path.exists(dmstat_full_path):
                os.makedirs(dmstat_full_path)
            diebold_mariano_dmstat_full_df.to_csv(os.path.join(dmstat_full_path, filename))

            pvalue_full_path = os.path.join(target_var, 'pvalue_full')
            if not os.path.exists(pvalue_full_path):
                os.makedirs(pvalue_full_path)
            diebold_mariano_pvalue_full_df.to_csv(os.path.join(pvalue_full_path, filename))
            '''

            #print(error_df)


# In[9]:


def full_combined_error_metrics(target_variables_file, pred_directory, pred_combi_directory, pred_2_directory, csr_rp_1_directory, csr_rp_2_directory, csr_rp_3_directory, pred_train_naive_directory):
    target_variables = []
    with open(target_variables_file, 'r') as file:
        for line in file:
            # Remove linebreak which is the last character of the string
            target_variable = line[:-1]
            # Add item to the list
            target_variables.append(target_variable)
    print(target_variables)

    Parallel(n_jobs=-2, verbose=51)(delayed(combined_error_metrics)(target_var, 
                                                                    pred_directory, 
                                                                    pred_combi_directory,
                                                                    pred_2_directory,
                                                                    csr_rp_1_directory,
                                                                    csr_rp_2_directory,
                                                                    csr_rp_3_directory,
                                                                    pred_train_naive_directory) for target_var in target_variables)
    
full_combined_error_metrics('target_variables.txt', 'pred','pred_combi','pred_2', 'pred_csr_rp_1', 'pred_csr_rp_2', 'pred_csr_rp_3', 'pred_train_naive')


# In[ ]:




