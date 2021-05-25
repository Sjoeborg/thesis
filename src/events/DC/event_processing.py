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
from functions import dc_params_nsi, nufit_params_nsi_IO, nufit_params_nsi
from scipy.stats import chi2
from scipy.optimize import minimize
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

def chisq(params,events, data, background,z,sigma_a, sigma_b, sigma_g, sigma_syst):
    z_0 = -np.median(z)
    if len(params) == 4:
        a,c,b, g = params
        S_th = a*(1+b*(z[0:-1]+z_0)+g)*events  + c*background
        penalty = (1-a)**2/sigma_a**2 + b**2 / sigma_b**2 + g**2 /sigma_g**2
    elif len(params) == 3:
        a,c,b = params
        S_th = a*(1+b*(z[0:-1]+z_0))*events + c*background
        penalty = (1-a)**2/sigma_a**2 + b**2 / sigma_b**2 
    elif len(params) == 2:
        a,c = params
        S_th = a*events + c*background
        penalty = (1-a)**2/sigma_a**2
    
    chi2= np.sum((S_th - data)**2/(data + sigma_syst**2))+ penalty
    return chi2

def perform_chisq(events, data,background,sigma_syst, z = np.linspace(-1,1,9), sigma_a=0.25, sigma_b=None, sigma_g =None, x0=[1,1]):
    res = minimize(fun=chisq, x0=x0, args=(events,data, background, z,sigma_a, sigma_b, sigma_g,sigma_syst), method='Nelder-Mead',options={'maxiter': 1e5, 'maxfev':1e5})
    assert res.success, res
    return res.fun, res.x

def get_deltachi(H1_list, delta_T, sigma = [0.25,0.15], f=0.09, x0=[1,0,0]):
    sigma_a = sigma[0]
    sigma_b = sigma[1]
    sigma_g = delta_T
    data = np.array([get_hist(events2018_DC().query(f'pid==0'), 'count_events'),get_hist(events2018_DC().query(f'pid==1'), 'count_events')])
    background =np.array([get_hist(events2018_DC().query(f'pid==0'), 'count_background'),get_hist(events2018_DC().query(f'pid==1'), 'count_background')])
    sigma_syst = f * np.array([get_hist(events2018_DC().query(f'pid==0'), 'abs_uncert'),get_hist(events2018_DC().query(f'pid==1'), 'abs_uncert')])
    chisq_H1_list  = np.array([perform_chisq(H1, data, background,z=zreco, sigma_syst=sigma_syst,sigma_a=sigma_a,sigma_b=sigma_b,sigma_g=sigma_g, x0=x0)[0] for H1 in H1_list])

    best_fit_index = np.argmin(chisq_H1_list)
    
    return chisq_H1_list, best_fit_index


def get_contour(deltachi, y_range,x_range, df):
    cl_99_bool = np.where(deltachi < chi2.ppf(q = 0.99,df=df),True,False)
    cl_90_bool = np.where(deltachi < chi2.ppf(q = 0.90,df=df),True,False)

    x_cl90_index = get_boundary(cl_90_bool)
    y_cl90_index = np.linspace(0,len(x_cl90_index)-1,len(x_cl90_index)).astype('int')
    x_cl99_index = get_boundary(cl_99_bool)
    y_cl99_index = np.linspace(0,len(x_cl99_index)-1,len(x_cl99_index)).astype('int')

    return x_range[x_cl90_index], x_range[x_cl99_index], y_range[y_cl90_index], y_range[y_cl99_index]


def list_of_params_nsi(dicta, dm31_range, th23_range, ett_range, emt_range, eem_range, eet_range):
    def update_dict(dicta,p):
        dict2 = dicta.copy()
        dict2.update(p)
        return dict2
    dict_list = [update_dict(dicta,{'e_em':eem,'e_et':eet,'e_tt':tt,'e_mt':mt,'theta_23':th23, 'dm_31':dm31}) for eet in eet_range for eem in eem_range for mt in emt_range for tt in ett_range for th23 in th23_range for dm31 in dm31_range]
    return dict_list

def marginalize(chisq,dm31_range, th23_range, ett_range, emt_range, eem_range, eet_range):
    from scipy.integrate import simps 
    best_fit_index = chisq.argmin()
    reshaped_chisq = chisq.reshape(len(eet_range),len(eem_range),len(emt_range),len(ett_range), len(th23_range), len(dm31_range))
    N = len(th23_range)*len(dm31_range)
    if len(th23_range) > 1 and len(dm31_range) > 1:
        marginalized_chisq = simps(simps(reshaped_chisq))/N #Integrate out oscillation parameters and "normalize"
    else:
        marginalized_chisq = reshaped_chisq[:,:,:,:,0,0]
    best_eet_index, best_eem_index, best_emt_index, best_ett_index, best_th23_index, best_dm31_index = np.unravel_index(best_fit_index,reshaped_chisq.shape)
    
    deltachi = marginalized_chisq - marginalized_chisq.min()
    return deltachi, best_dm31_index, best_th23_index, best_ett_index, best_emt_index, best_eem_index, best_eet_index

def marginalize_one(deltachi, axis):
    '''
    Assumes 3 axes with same length
    '''
    from scipy.integrate import simps 
    N = deltachi.shape[axis]
    marginalized_chisq = simps(deltachi, axis=axis)/N #Integrate out oscillation parameters
    return marginalized_chisq - marginalized_chisq.min()

def get_param_list(dm31N,th23N, ett_tuple, emt_tuple, eem_tuple, eet_tuple, IO=False):
    ettFrom, ettTo, ettN = ett_tuple
    emtFrom, emtTo, emtN = emt_tuple
    eemFrom, eemTo, eemN = eem_tuple
    eetFrom, eetTo, eetN = eet_tuple
    if IO is False:
        dm31_range = np.linspace(2.435e-3,2.598e-3,dm31N-1)
        th23_range = np.linspace(40.1*np.pi/180,51.7*np.pi/180,th23N-1)
    else: #Nufit gives dm32 for IO, so convert the values and bounds to dm31 explicitly here
        dm31_range = np.linspace(-2.581e-3 - 8.04e-5, -2.414e-3 + 8.04e-5, dm31N-1)
        th23_range = np.linspace(40.3*np.pi/180, 51.8*np.pi/180, th23N-1)
    
    ett_range = np.linspace(ettFrom,ettTo,ettN-1)
    emt_range = np.linspace(emtFrom,emtTo,emtN-1)

    eem_range = np.linspace(eemFrom,eemTo,eemN-1)
    eet_range = np.linspace(eetFrom,eetTo,eetN-1)

    
    if IO:
        nsi_params = nufit_params_nsi_IO.copy()
    else:
        nsi_params = nufit_params_nsi.copy()

    # We want to marg over best-fit points too, so insert them
    if nsi_params['dm_31'] not in dm31_range:
        dm31_range = np.sort(np.append(dm31_range,nsi_params['dm_31']))
    if nsi_params['theta_23'] not in th23_range:
        th23_range = np.sort(np.append(th23_range, nsi_params['theta_23']))
    if nsi_params['e_tt'] not in ett_range:
        ett_range = np.sort(np.append(ett_range, nsi_params['e_tt']))
    if nsi_params['e_mt'] not in emt_range:
        emt_range = np.sort(np.append(emt_range, nsi_params['e_mt']))
    if nsi_params['e_em'] not in eem_range:
        eem_range = np.sort(np.append(eem_range, nsi_params['e_em']))
    if nsi_params['e_et'] not in eet_range:
        eet_range = np.sort(np.append(eet_range, nsi_params['e_et']))

    return dm31_range, th23_range,ett_range, emt_range, eem_range, eet_range

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
                        raise FileNotFoundError(f'P{flavor_from}{flavor_to} {ndim}gen for pid {pid}, dm={dict["dm_31"]}, th23={dict["theta_23"]}, e_mm={dict["e_mm"]},e_mt={dict["e_mt"]}, not found')
                return True

                

def get_marginalized_array(H1,dm31_range,th23_range,ett_range,emt_range,eem_range,eet_range, param_list, nsi_param, sigma_a,sigma_b, f):
    chisq,  best_fit_index = get_deltachi([H for H in H1],0,[sigma_a,sigma_b],f,x0=[1,0])

    reshaped_chisq = chisq.reshape(len(eet_range),
                                len(eem_range),
                                len(emt_range), 
                                len(ett_range), 
                                len(th23_range), 
                                len(dm31_range)) #reshaped_chisq[eet,eem,emt,emm,th23,dm31]
    minimum_oscillation = np.min(reshaped_chisq,axis=(0,1,2,3)).reshape(1,1,1,1,len(th23_range),len(dm31_range)) #Marginalize osc params
    deltachi = reshaped_chisq - minimum_oscillation


    best_eet_index,best_eem_index,best_emt_index,best_ett_index,best_th23_index, best_dm31_index = np.unravel_index(best_fit_index,deltachi.shape)
    best_dm31, best_th23, best_ett, best_emt, best_eem, best_eet = (dm31_range[best_dm31_index], 
                                                                    th23_range[best_th23_index],
                                                                    ett_range[best_ett_index],
                                                                    emt_range[best_emt_index],
                                                                    eem_range[best_eem_index],
                                                                    eet_range[best_eet_index])
    best_fit_params = param_list[best_fit_index]
    assert best_fit_params['dm_31'] == best_dm31 
    assert best_fit_params['theta_23'] == best_th23
    assert best_fit_params['e_tt'] == best_ett
    assert best_fit_params['e_mt'] == best_emt
    assert best_fit_params['e_em'] == best_eem
    assert best_fit_params['e_et'] == best_eet

    marginalized_deltachi = deltachi[:,:,:,:,best_th23_index,best_dm31_index].T # marginalized_deltachi[ett,emt,eem,eet]
    deltachi_ett = marginalized_deltachi[:,best_emt_index, best_eem_index, best_eet_index]
    deltachi_emt = marginalized_deltachi[best_ett_index,:, best_eem_index, best_eet_index]
    deltachi_eem = marginalized_deltachi[best_ett_index,best_emt_index, :, best_eet_index]
    deltachi_eet = marginalized_deltachi[best_ett_index,best_emt_index, best_eem_index, :]
    if nsi_param == 'ett':
        return deltachi_ett
    elif nsi_param == 'emt':
        return deltachi_emt
    elif nsi_param == 'eem':
        return deltachi_eem
    elif nsi_param == 'eet':
        return deltachi_eet
    else: #If no nsi_param is given, return whole array. Used in joint analysis
        return reshaped_chisq.T
if __name__ == '__main__':
    pass
