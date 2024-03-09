import numpy as np
from model_confidence_set import ModelConfidenceSet
from .garchmidas import *


def get_SE(N, vol, ht):
    e = vol/np.sqrt(252) - np.sqrt(ht)/100
    e2 = e**2

    return e2

def get_AE(N, vol, ht):
    e = vol/np.sqrt(252) - np.sqrt(ht)/100
    abs_e = np.abs(e)

    return abs_e

def mcs_test(vol, ret_Xtest_K_res_nbr_tuple, lookahead):

    SE = []
    AE = []
    
    for (ret, X_test,K,res, nbr) in ret_Xtest_K_res_nbr_tuple:
    
        if isinstance(X_test, list) or isinstance(X_test, tuple):
            M = len(X_test)-1
        else:
            M = 0
    
        func = [get_onefactor_tau, 
                get_twofactor_tau, 
                get_threefactor_tau, 
                get_fourfactor_tau]

        _, _, _, _, _, _ht, _ =  GARCH_MIDAS(res['x'], 
                                            ret, 
                                            X_test, 
                                            K, 
                                            get_tau=func[M], 
                                            full_output=True)
        ht = _ht[-1*int(nbr):]
        SE.append(get_SE(lookahead, vol[:lookahead], ht[:lookahead]))
        AE.append(get_AE(lookahead, vol[:lookahead], ht[:lookahead]))

    SE= np.array(SE).transpose()
    AE= np.array(AE).transpose()
    
    mcs_mse = ModelConfidenceSet(SE, n_boot=1000, alpha=0.1, show_progress=True)
    mcs_mse.compute()
    mcs_mse_results = mcs_mse.results()
    
    mcs_mae = ModelConfidenceSet(AE, n_boot=1000, alpha=0.1, show_progress=True)
    mcs_mae.compute()
    mcs_mae_results = mcs_mae.results()    
    
        
    return mcs_mse_results,mcs_mae_results