import pandas as pd
import numpy as np
from math import pi, isnan, isinf
from scipy.optimize import minimize
from scipy.linalg import inv
from scipy import stats
from .garchmidas import *


def is_os_split(data, date, K, diff = None):
    date = pd.to_datetime(date)
    train_data = data[data.index < date].copy()
    test_split =  date + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(K+1)
    test_data = data[data.index >= test_split].copy()

    X1_train = get_factors(train_data, 
                           'daily_return', 
                           K, 
                           fun = get_realized_variance, 
                           diff = diff)
    
    X1_test  = get_factors(test_data, 
                           'daily_return', 
                           K, 
                           fun = get_realized_variance,
                           diff = diff)

    X2_train = get_factors(train_data, 'cpu_index', K, diff = diff)
    X2_test = get_factors(test_data, 'cpu_index', K, diff = diff)
    
    X3_train = get_factors(train_data, 'Monthly_Disaster_Freq', K, diff = diff)
    X3_test = get_factors(test_data, 'Monthly_Disaster_Freq', K, diff = diff)

    train_returns = np.array(train_data[['daily_return']])
    test_returns = np.array(test_data[['daily_return']])

    X_train = (X1_train, X2_train, X3_train)
    X_test = (X1_test, X2_test, X3_test)


    return X_train, X_test, train_returns, test_returns

    