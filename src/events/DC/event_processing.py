import sys,os
if __name__ == '__main__':
    os.chdir('../../')
    sys.path.append('./src/data')
    sys.path.append('./src/events')
    sys.path.append('./src/probability')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from DC.importer import *
from DC.processer import *
from DC.main import get_all_events
from functions import perform_chisq,dc_params_nsi
from scipy.stats import chi2

Ereco = [5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413]
zreco = [-1., -0.75, -0.5 , -0.25,  0., 0.25, 0.5, 0.75, 1.]
Ereco_midpoints = Ereco[0:-1] +np.diff(Ereco)/2 #For scatter plot
zreco_midpoints = zreco[0:-1] +np.diff(zreco)/2 #For scatter plot


def to_hist(arr):
    return arr.tolist() + [arr[-1]]


def get_boundary(arr):
    returned = []
    for i in range(arr.shape[1]):
        try:
            returned.append((np.max(np.nonzero(arr[:,i]==True))+1))
        except (ValueError):
            returned.append(0)
    returned= np.array(returned)
    max_val = arr.shape[0]
    returned[returned >= max_val] = max_val -1 # If a column has all true, set cotour at last row
    return np.array(returned)


def is_precomputed(pid,ndim, dict, check=False, quick=True):
    for anti in [True,False]:
        for flavor_from in ['e','m']:
            for flavor_to in ['e','m','t']:
                try:
                    if quick:
                        get_probabilities_DC(flavor_from, flavor_to, 5,2,dict,anti,pid,ndim)
                    else:
                        for Ebin in range(8):
                            for zbin in range(8):
                                get_probabilities_DC(flavor_from, flavor_to, Ebin,zbin,dict,anti,pid,ndim)
                except (FileNotFoundError,KeyError):
                    if check:
                        return False
                    else:
                        if quick:
                            raise FileNotFoundError(f'P{flavor_from}{flavor_to}, for pid {pid}, dm={dict["dm_41"]}, s24={np.sin(2*dict["theta_24"])**2}, s34={np.sin(2*dict["theta_34"])**2}, not found')
                        else:
                            raise FileNotFoundError(f'P{flavor_from}{flavor_to}, E{Ebin}z{zbin} for pid {pid}, dm={dict["dm_41"]}, s24={np.sin(2*dict["theta_24"])**2}, s34={np.sin(2*dict["theta_34"])**2}, not found')
                return True

def return_precomputed(pid,ndim,params, nsi=False, quick=True):
    params= np.array(params)
    precomputed_list = np.array([is_precomputed(pid,ndim, p, check=True,quick=quick) for p in params])
    mask = precomputed_list == True
    computed_params = params[mask]
    return computed_params

def normalize_events(H0_events,H1_events_list,pid):
    if pid == 1:
        null = pd.read_csv('./src/data/files/DC/2018/track_null.csv', header=None)
    elif pid == 0:
        null = pd.read_csv('./src/data/files/DC/2018/cascade_null.csv', header=None)
    N = np.sum(null[1].values)/np.sum(np.array(H0_events))
    H0_normalized = H0_events*N
    H1_list_normalized = [N*H1 for H1 in H1_events_list]

    return H0_normalized, H1_list_normalized

def get_deltachi(H1_list_normalized,H0_normalized,pid,y_range,x_range, delta_T, sigma = [0.25,0.15], f=0.09, x0=[1,0,0], z_range=None):
    sigma_a = sigma[0]
    sigma_b = sigma[1]
    sigma_g = delta_T
    f = f
    if pid == 1:
        DC_observed = events2018_DC(track=True, cascade=False)
    elif pid == 0:
        DC_observed = events2018_DC(track=False, cascade=True)
    sigma_syst = f*DC_observed
    x0=x0
    chisq_H0, a_H0 = perform_chisq(H0_normalized,DC_observed,sigma_syst=sigma_syst,z=zreco,sigma_a=sigma_a,sigma_b=sigma_b,sigma_g=sigma_g , x0=x0)
    chisq_H1_list  = np.array([perform_chisq(H1_norm, DC_observed,sigma_syst=sigma_syst,z=zreco, sigma_a=sigma_a,sigma_b=sigma_b,sigma_g=sigma_g, x0=x0)[0] for H1_norm in H1_list_normalized])
    delta_chi = chisq_H1_list - np.min(chisq_H1_list)#chisq_H1_list - chisq_H0

    best_fit_index = np.argmin(delta_chi)
    
    if z_range is not None:
        deltachi_reshaped = delta_chi.reshape(len(y_range),len(x_range),len(z_range))
    else:
        deltachi_reshaped = delta_chi.reshape(len(y_range),len(x_range))
    return deltachi_reshaped, best_fit_index, np.min(chisq_H1_list)#, chisq_H0

def get_contour(deltachi, y_range,x_range, df):
    cl_99_bool = np.where(deltachi < chi2.ppf(q = 0.99,df=df),True,False)
    cl_90_bool = np.where(deltachi < chi2.ppf(q = 0.90,df=df),True,False)

    x_cl90_index = get_boundary(cl_90_bool)
    y_cl90_index = np.linspace(0,len(x_cl90_index)-1,len(x_cl90_index)).astype('int')
    x_cl99_index = get_boundary(cl_99_bool)
    y_cl99_index = np.linspace(0,len(x_cl99_index)-1,len(x_cl99_index)).astype('int')


    return x_range[x_cl90_index], x_range[x_cl99_index], y_range[y_cl90_index], y_range[y_cl99_index]

def list_of_params_nsi(dicta,s24_range, emm_range, emt_range=None):
    def update_dict(dict,p):
        dict2 = dicta.copy()
        dict2.update(p)
        return dict2
    if emt_range is None:
        dict_list = [update_dict(dicta,{'e_mm':mm,'theta_24':np.arcsin(np.sqrt(s24))/2}) for mm in emm_range for s24 in s24_range]
    else:
        dict_list = [update_dict(dicta,{'e_mm':mm,'e_mt':mt,'theta_24':np.arcsin(np.sqrt(s24))/2}) for mt in emt_range for mm in emm_range for s24 in s24_range]
    return dict_list
def return_precomputed_nsi(pid,ndim,params):
    params= np.array(params)
    precomputed_list = np.array([is_precomputed_nsi(pid,ndim, p, check=False) for p in params])
    mask = precomputed_list == True
    computed_params = params[mask]
    return computed_params
def is_precomputed_nsi(pid,ndim, dict, check=False):
    for anti in [True,False]:
        for flavor_from in ['e','m']:
            for flavor_to in ['e','m','t']:
                try:
                    get_probabilities_DC(flavor_from, flavor_to, 5,2,dict,anti,pid, ndim, nsi=True)
                except (FileNotFoundError,KeyError):
                    if check:
                        return False
                    else:
                        raise FileNotFoundError(f'P{flavor_from}{flavor_to} {ndim}gen for pid {pid}, dm={dict["dm_41"]}, s24={np.sin(2*dict["theta_24"])**2}, e_mm={dict["e_mm"]},e_mt={dict["e_mt"]}, not found')
                return True
if __name__ == '__main__':
    pass
