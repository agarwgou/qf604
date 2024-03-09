import pandas as pd
import numpy as np
from .garchmidas import *

def is_os_split(data, date_split, K, diff = None, study = 'replicate'):
    if study not in ['replicate', 'extension']:
        raise ValueError('Study has to be either "replicate" or "extension"!')
    
    date_split = pd.to_datetime(date_split)
    train_data = data[data.index < date_split].copy()
    # test_split =  date + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(K+1)
    test_data = data.copy()

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

    if study == 'extension':
        X4_train = get_factors(train_data, 'North', K, diff = diff)
        X4_test = get_factors(test_data, 'North', K, diff = diff)

        X5_train = get_factors(train_data, 'Storage', K, diff = diff)
        X5_test = get_factors(test_data, 'Storage', K, diff = diff)

        X_train += (X4_train, X5_train)
        X_test += (X4_test, X5_test)

    nbr_test_days = data[(data.index >= date_split)].shape[0]

    return X_train, X_test, train_returns, test_returns, nbr_test_days

    