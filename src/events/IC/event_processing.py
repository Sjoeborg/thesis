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
from IC.importer import *
from IC.processer import *
from IC.main import sim_events, list_of_params, ic_params
from functions import perform_chisq
from scipy.stats import chi2

IC_observed_full = np.array(get_IC_data().T)
E_rate, z_rate = get_flux_factor()
flux_E_factors_full, flux_z_factors_full = bin_flux_factors(E_rate,z_rate)
EFrom,ETo,zFrom,zTo = 0,13,0,20
z_bins = np.arange(zFrom,zTo)
E_bins, z_bins_T = np.arange(EFrom,ETo), np.arange(zFrom,zTo)[:,None]
n_Ebins, n_zbins = len(E_bins), len(z_bins)
flux_E_factors = flux_E_factors_full[E_bins]
flux_z_factors = flux_z_factors_full[z_bins]
flux_factors = np.outer(flux_E_factors, flux_z_factors)
IC_observed = IC_observed_full[E_bins,z_bins_T].T

E_ratios_full = pd.read_csv('./src/data/files/E_ratios.csv', header=None, names=['Ereco', 'Eratio']).Eratio.values
z_ratios_full = pd.read_csv('./src/data/files/z_ratios.csv', header=None, names=['zreco', 'zratio']).zratio.values
IC_per_z_full = np.array(np.sum(IC_observed_full, axis=0))
IC_per_E_full = np.array(np.sum(IC_observed_full, axis=1))
MC_per_E_full = (IC_per_E_full/E_ratios_full)
MC_per_z_full = (IC_per_z_full/z_ratios_full)

MC_ratios_full = np.outer(E_ratios_full, z_ratios_full)
IC_MC_full = IC_observed_full / MC_ratios_full
IC_MC = IC_MC_full[EFrom:ETo+1,z_bins]

Ereco_full = 500*10**np.linspace(0.0,1.3,14)
Ereco_full_midpoints = Ereco_full[0:-1] +np.diff(Ereco_full)/2 #For scatter plot

Ereco = Ereco_full[EFrom:ETo+1]
Ereco_midpoints= Ereco_full_midpoints[3:14]

zreco_full = np.linspace(-1,0,21)
zreco_full_midpoints = zreco_full[0:-1] +np.diff(zreco_full)/2 #For scatter plot

zreco = zreco_full[zFrom:zTo+1]
zreco_midpoints = zreco_full_midpoints[z_bins]


def to_hist(arr):
    return arr.tolist() + [arr[-1]]


def count_plots(H1,H0):
    IC_per_z = np.sum(IC_observed, axis=0)
    IC_per_E = np.sum(IC_observed, axis=1)

    IC_rate_z = IC_per_z /np.sum(H0, axis= 0)
    IC_rate_E = IC_per_E /np.sum(H0, axis= 1)

    H1_per_z_hist = to_hist(np.sum(H1, axis= 0))
    H1_per_E_hist = to_hist(np.sum(H1, axis= 1))
    H0_per_z_hist = to_hist(np.sum(H0, axis= 0))
    H0_per_E_hist = to_hist(np.sum(H0, axis= 1))

    H1_rate_z_hist = to_hist(np.sum(H1, axis= 0) / np.sum(H0, axis= 0))
    H1_rate_E_hist = to_hist(np.sum(H1, axis= 1) / np.sum(H0, axis= 1))
    H0_rate_z_hist = to_hist(np.sum(H0, axis= 0) / np.sum(H0, axis= 0))
    H0_rate_E_hist = to_hist(np.sum(H0, axis= 1) / np.sum(H0, axis= 1))

    fig, ax = plt.subplots(2,2, sharex='col', squeeze=True,gridspec_kw={'width_ratios': [3, 3], 'height_ratios':[3,1]}, figsize=(12,8))
    ax = ax.flatten()

    ax[0].scatter(Ereco_midpoints, IC_per_E, label='IC data', s=10, color='black', zorder=10)
    ax[0].step(Ereco, H1_per_E_hist, label='Sterile',lw=3, where='post', color='blue')
    ax[0].step(Ereco, H0_per_E_hist, label='Null',   lw=1.5, where='post', color='red')
    
    ax[1].scatter(zreco_midpoints, IC_per_z, label='IC data', s=10, color='black',zorder=10)
    ax[1].step(zreco, H1_per_z_hist, label='Sterile',lw=3, where='post', color='blue')
    ax[1].step(zreco, H0_per_z_hist, label='Null',   lw=1.5, where='post', color='red')

    ax[2].scatter(Ereco_midpoints, IC_rate_E, label='IC data',  s=10, color='black',zorder=10)
    ax[2].step(Ereco, H1_rate_E_hist, label='Sterile',lw=3, where='post', color='blue')
    ax[2].step(Ereco, H0_rate_E_hist, label='Null',   lw=1.5, where='post', color='red')
    
    ax[3].scatter(zreco_midpoints, IC_rate_z, label='IC data',  s=10, color='black',zorder=10)
    ax[3].step(zreco, H1_rate_z_hist, label='Sterile',lw=3, where='post', color='blue')
    ax[3].step(zreco, H0_rate_z_hist, label='Null',   lw=1.5, where='post', color='red')


    ax[0].set_xlim((Ereco.min(),Ereco.max()))
    ax[0].set_ylabel('Counts')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[2].set_xscale('log')
    ax[2].set_xlabel(r'$E_\mu^{reco}$ [GeV]')
    ax[2].set_ylabel('Ratio to Null')
    ax[2].grid(True,which='both', axis='both', alpha=0.3)

    ax[3].set_xlim((zreco.min(),zreco.max()))
    ax[3].set_ylim(ax[2].get_ylim())
    ax[3].grid(True,which='both', axis='both', alpha=0.3)
    ax[3].set_xlabel(r'$\cos{(\theta^{reco}_z)}$')
    
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0,0.4))
    plt.subplots_adjust(hspace=0.05)


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

def norm_plot(simulated_events):
    normalization = IC_observed/simulated_events
    n_zbins, n_Ebins = normalization.shape
    fig, ax = plt.subplots()

    fig.set_size_inches(18.5, 10.5)
    im = ax.imshow(normalization,cmap="GnBu",origin='lower', extent= (0, n_zbins, 0, n_Ebins))
    cbar = ax.figure.colorbar(im, ax=ax)
    #cbar.formatter.set_powerlimits((0, 0))
    '''
    for i in range(n_Ebins):
        for j in range(n_zbins):
            text = ax.text(j+0.5, i+0.5, np.round(np.asarray(normalization)[j,i],1),
                ha="center", va="center", color="black")

    ax.set_xlabel(r'$\cos{(\theta_z)}$ bins', fontsize=20)
    ax.set_ylabel(r'$E_{\nu}$ bins', fontsize=20)
    ax.set_title('Ratio of observed IC events to my null hypothesis\n', fontsize=20)
    ax.set_xticks(np.arange(n_zbins+1))
    ax.set_xticklabels(np.round(np.linspace(-0.9,-0.1,n_zbins+1),2), fontsize = 13)
    ax.set_yticks(np.arange(n_Ebins+1))
    E_ticks = 500*10**np.linspace(0.3,1.3,n_Ebins+1)
    ax.set_yticklabels(E_ticks.astype(int), fontsize=11)
    '''

def normalize_bin_by_bin(simulated_events, MC = True, MC_old=False, correct_flux=False):
    if MC_old:
        IC_events = IC_MC_2017
    elif MC:
        IC_events = IC_MC

    if correct_flux:
        simulated_events = flux_E_factors*simulated_events

    normalization = IC_events/simulated_events

    return np.array(normalization)


def is_precomputed(N,ndim, dict, check=False, quick=True):
    for anti in [True,False]:
        for flavor_from in ['e','m']:
            flavor_to  = 'm'
            try:
                if quick:
                    get_probabilities(flavor_from, flavor_to, 5,5,dict,anti,N,ndim)
                else:
                    for Ebin in range(3,13):
                        for zbin in range(0,20):
                            get_probabilities(flavor_from, flavor_to, Ebin,zbin,dict,anti,N,ndim)
            except (FileNotFoundError,KeyError):
                if check:
                    return False
                else:
                    if quick:
                        raise FileNotFoundError(f'P{flavor_from}{flavor_to}, for N={N}, dm={dict["dm_41"]}, s24={np.sin(2*dict["theta_24"])**2}, s34={np.sin(2*dict["theta_34"])**2}, not found')
                    else:
                        raise FileNotFoundError(f'P{flavor_from}{flavor_to}, E{Ebin}z{zbin} for N={N}, dm={dict["dm_41"]}, s24={np.sin(2*dict["theta_24"])**2}, s34={np.sin(2*dict["theta_34"])**2}, not found')
            return True

def return_precomputed(N,ndim,params, nsi=False, quick=True):
    params= np.array(params)
    precomputed_list = np.array([is_precomputed(N,ndim, p, check=True,quick=quick) for p in params])
    mask = precomputed_list == True
    computed_params = params[mask]
    return computed_params

def normalize_events(H0_events,H1_events_list,z_bins):
    norm_factors = normalize_bin_by_bin(H0_events[:,z_bins],MC=True)
    H0_normalized = norm_factors *H0_events[:,z_bins]
    H1_list_normalized = [norm_factors*H1[:,z_bins] for H1 in H1_events_list]

    return H0_normalized, H1_list_normalized

def get_deltachi(H1_list_normalized,H0_normalized,y_range,x_range, delta_T, sigma = [0.25,0.15], f=0.09, x0=[1,0,0], z_range=None):
    sigma_a = sigma[0]
    sigma_b = sigma[1]
    sigma_g = delta_T
    f = f
    sigma_syst = f*IC_observed#f*np.sum(IC_observed, axis=0)
    x0=x0
    chisq_H0, a_H0 = perform_chisq(H0_normalized,IC_observed,sigma_syst=sigma_syst,z=zreco,sigma_a=sigma_a,sigma_b=sigma_b,sigma_g=sigma_g , x0=x0)
    chisq_H1_list  = np.array([perform_chisq(H1_norm, IC_observed,sigma_syst=sigma_syst,z=zreco, sigma_a=sigma_a,sigma_b=sigma_b,sigma_g=sigma_g, x0=x0)[0] for H1_norm in H1_list_normalized])
    delta_chi = chisq_H1_list - np.min(chisq_H1_list)#chisq_H1_list - chisq_H0

    best_fit_index = np.argmin(delta_chi)
    
    if z_range is not None:
        deltachi_reshaped = delta_chi.reshape(len(y_range),len(x_range),len(z_range))
    else:
        deltachi_reshaped = delta_chi.reshape(len(y_range),len(x_range))
    return deltachi_reshaped, best_fit_index, np.min(chisq_H1_list), chisq_H0

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
def return_precomputed_nsi(N,ndim,params, nsi=False):
    params= np.array(params)
    precomputed_list = np.array([is_precomputed_nsi(N,ndim, p, check=False) for p in params])
    mask = precomputed_list == True
    computed_params = params[mask]
    return computed_params
def is_precomputed_nsi(N,ndim, dict, check=False):
    for anti in [True,False]:
        for flavor_from in ['e','m']:
            flavor_to  = 'm'
            try:
                get_probabilities(flavor_from, flavor_to, 5,5,dict,anti,N, ndim)
            except (FileNotFoundError,KeyError):
                if check:
                    return False
                else:
                    raise FileNotFoundError(f'P{flavor_from}{flavor_to} {ndim}gen for N={N}, dm={dict["dm_41"]}, s24={np.sin(2*dict["theta_24"])**2}, e_mm={dict["e_mm"]},e_mt={dict["e_mt"]}, not found')
            return True
if __name__ == '__main__':
    pass