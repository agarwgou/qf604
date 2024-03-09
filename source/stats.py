import pandas as pd
import numpy as np
from scipy.linalg import inv
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

def get_descriptive_stats(df):
    '''
    Helper function to run required descriptive statistics of the data.
    Descriptive statistics include
        - no. of observations
        - mean
        - standard deviation
        - minimum
        - maximum
        - skewness
        - kurtosis
        - Augmented Dickey Fuller Statistics

    Arg:
        df (DataFrame): A combined dataframe containing required data.

    Returns:
        DataFrame: A dataframe of descriptive statistics.

    '''
    cpu = np.array(df['cpu_index'].resample('M').mean())
    emdat = np.array(df['Monthly_Disaster_Freq'].resample('M').mean())
    ng = np.array(df['daily_return'])

    diff_ln_cpu = np.log(cpu[1:]/cpu[:-1])
    diff_ln_emdat = np.log(emdat[1:]/emdat[:-1])

    data_lst = [ng, cpu, diff_ln_cpu, emdat, diff_ln_emdat]
    title_lst = ['Natural gas futures return (daily)',
             'CPU Index',
             'd. ln(CPU Index)',
             'Natural disasters frequency',
             'd. ln(Natural disasters frequency)']

    descriptive_stats = []
    for data, title in zip(data_lst, title_lst):
        descriptive_stats\
            .append(
                {
                    'Data' : title,
                    'Obs' : len(data),
                    'mean' : np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'skewness': stats.skew(data),
                    'kurtosis' : stats.kurtosis(data),
                    'ADF' : adfuller(data)[0]
                }
            )
        
    df_stats =\
        pd\
            .DataFrame(descriptive_stats)\
            .set_index('Data')
    
    return df_stats

def run_kpss_test(df):
    '''
    Helper function to run required KPSS test of the data.
    The KPSS tests are ran on:


    Arg:
        df (DataFrame): A combined dataframe containing required data.

    Returns:
        DataFrame: A dataframe of descriptive statistics.

    '''
    cpu = np.array(df['cpu_index'].resample('M').mean())
    emdat = np.array(df['Monthly_Disaster_Freq'].resample('M').mean())
    ng = np.array(df['daily_return'])

    diff_cpu = cpu[1:]-cpu[:-1]
    diff_emdat = emdat[1:]-emdat[:-1]
    
    diff_ln_cpu = np.log(cpu[1:]/cpu[:-1])
    diff_ln_emdat = np.log(emdat[1:]/emdat[:-1])

    data_lst = [ng, cpu, diff_cpu, diff_ln_cpu, emdat, diff_emdat, diff_ln_emdat]
    title_lst = ['Natural gas futures return (daily)',
             'CPU Index',
             'd. CPU Index',
             'd. ln(CPU Index)',
             'Natural disasters frequency',
             'd. Natural disasters frequency',
             'd. ln(Natural disasters frequency)']

    kpss_stats = []

    for data, title in zip(data_lst, title_lst):
        res = kpss(data)

        kpss_stats\
            .append(
                {
                    'Data' : title,
                    'KPSS Statistic' : res[0],
                    'p-value' : res[1]
                }
            )
    
    df_stats =\
        pd\
            .DataFrame(kpss_stats)\
            .set_index('Data')
    
    return df_stats




def hessian_matrix(fun, theta, args, epsilon=1e-05):
    '''
    Function to calculate the hessian matrix of the model.

    '''
    f = fun(theta, *args)
    h_ = epsilon*np.abs(theta)
    
    h = np.diag(h_)
    K = theta.shape[0]
    
    fp = np.zeros(K)
    fm = np.zeros(K)
    for i in range(K):
        fp[i] = fun(theta+h[i], *args)
        fm[i] = fun(theta-h[i], *args)
        
    fpp = np.zeros((K,K))
    fmm = np.zeros((K,K))
    for i in range(K):
        for j in range(i,K):
            fpp[i,j] = fun(theta + h[i] + h[j],  *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(theta - h[i] - h[j],  *args)
            fmm[j,i] = fmm[i,j]
            
    hh = (np.diag(h))
    hh = hh.reshape((K,1))
    hh = np.dot(hh,hh.T)
    
    H = np.zeros((K,K))
    for i in range(K):
        for j in range(i,K):
            H[i,j] = (fpp[i,j] - fp[i] - fp[j] + f 
                       + f - fm[i] - fm[j] + fmm[i,j])/hh[i,j]/2
            H[j,i] = H[i,j]
    
    return H

def summary_stats(fun, T, params, args, labels, epsilon=1e-05):
    '''
    Function to calculate the p-values of the parameters given the model.

    '''
    H = hessian_matrix(fun, params, args, epsilon = epsilon)
    
    vcv = inv(H/T)/T
    eigenvalues, eigenvectors = np.linalg.eig(vcv)
    eigenvalues[eigenvalues < 0] = 0
    lambda_matrix = np.diag(eigenvalues)
    vcv = np.dot(np.dot(eigenvectors, lambda_matrix), np.linalg.inv(eigenvectors))

    se = np.diag(vcv)**0.5
    t = params/se
    pvalues = stats.t.sf(np.abs(t),args[0].size - params.shape[0])

    df =\
    (
        pd
        .DataFrame(
            data = {'values': params, 
                    'p-val': pvalues}, 
            index = labels
            ).T
    )

    return df

    
