import numpy as np
from dataProcesser import get_flux,get_aeff, generate_probabilities, get_probabilities, get_Etrue, get_interpolators
from dataImporter import *
from multiprocessing import Pool
from functions import ic_params, integrate
import time
from scipy.stats import lognorm


def get_events(E_index, z_index, alpha, npoints, params=ic_params, spectral_shift_parameters=[False, 2e3, 0.02], null=False, tau=False):
    '''
    Assume zr == zt, and thus the zenith resolution function is 1.
    '''
    E_buckets = 500*10**(np.linspace(0.0,1.3,14)) #log-binned in steps of 0.1 from 500 GeV to 500*1e1.3 GeV
    z_buckets = np.linspace(-1,0,21)

    Er = np.logspace(np.log10(E_buckets[E_index]), np.log10(E_buckets[E_index+1]), npoints)
    zr = np.linspace(z_buckets[z_index], z_buckets[z_index+1], npoints)



    Et, mu, sigma = get_Etrue(E_index=E_index,npoints=npoints, left_alpha=alpha, right_alpha=alpha,model=energy_resolution_models)
    resolution_gaussian = lognorm.pdf(Et, s = sigma, scale=np.exp(mu)) #Check doc of scipy.stats.lognorm for explanation of variables
    

    Er_mesh, Et_mesh = np.meshgrid(Er, Et) # meshgrid has shape (Er,zr,Et)


    zr_mesh, Er_mesh, Et_mesh = np.meshgrid(zr, Er, Et)

    aeff_m = interp_aeff(Et_mesh,zr_mesh)
    aeff_mbar = interp_aeff(Et_mesh,zr_mesh)

    if spectral_shift_parameters[0]:
        E_pivot = spectral_shift_parameters[1]
        delta_gamma = spectral_shift_parameters[2]
        factor = spectral_shift_factor(E = Et_mesh, E_pivot = E_pivot, delta_gamma=delta_gamma)   
        flux_m = factor*get_flux('m',Et_mesh,zr_mesh,interp_flux)
        flux_mbar = factor*get_flux('mbar',Et_mesh,zr_mesh,interp_flux)
    else:
        flux_m = get_flux('m',Et_mesh,zr_mesh,interp_flux)
        flux_mbar = get_flux('mbar',Et_mesh,zr_mesh,interp_flux)

    if not null:
        try:
            Pmm = get_probabilities('m', 'm', E_index,z_index,params,False,npoints)
        except FileNotFoundError:
            generate_probabilities('m','m',Et,zr,E_index, z_index, params,False,npoints,ndim=4)
            Pmm = get_probabilities('m', 'm', E_index,z_index,params,False,npoints)

        try:
            P_amam = get_probabilities('m', 'm', E_index,z_index,params,True,npoints)
        except FileNotFoundError:
            generate_probabilities('m','m',Et,zr,E_index, z_index, params,True,npoints,ndim=4)
            P_amam = get_probabilities('m', 'm', E_index,z_index,params,True,npoints)
        
        try:
            Pem = get_probabilities('e', 'm', E_index,z_index,params,False,npoints)
        except FileNotFoundError:
            generate_probabilities('e','m',Et,zr,E_index, z_index, params,False,npoints,ndim=4)
            Pem = get_probabilities('e', 'm', E_index,z_index,params,False,npoints)

        try:
            P_aeam = get_probabilities('e', 'm', E_index,z_index,params,True,npoints)
        except FileNotFoundError:
            generate_probabilities('e','m',Et,zr,E_index, z_index, params,True,npoints,ndim=4)
            P_aeam = get_probabilities('e', 'm', E_index,z_index,params,True,npoints)

        if tau:
            try:
                Pmt = get_probabilities('m', 't', E_index,z_index,params,False,npoints)
            except FileNotFoundError:
                generate_probabilities('m','t',Et,zr,E_index, z_index, params,False,npoints,ndim=4)
                Pmt = get_probabilities('m', 't', E_index,z_index,params,False,npoints)

            try:
                P_amat = get_probabilities('m', 't', E_index,z_index,params,True,npoints)
            except FileNotFoundError:
                generate_probabilities('m','t',Et,zr,E_index, z_index, params,True,npoints,ndim=4)
                P_amat = get_probabilities('m', 't', E_index,z_index,params,True,npoints)

            Pmm = Pmm + 0.1739*Pmt
            P_amam = P_amam + 0.1739*P_amat
    

        flux_e = get_flux('e',Et_mesh,zr_mesh,interp_flux)
        flux_ebar = get_flux('ebar',Et_mesh,zr_mesh,interp_flux)
        integrand = aeff_m*flux_m*Pmm + aeff_mbar*flux_mbar*P_amam + aeff_m*flux_e*Pem + aeff_mbar*flux_ebar*P_aeam
    else:
        integrand = aeff_m*flux_m + aeff_mbar*flux_mbar
    integrand *= resolution_gaussian* 2*np.pi * 240747841
 
    if np.isnan(np.sum(integrand)):
        print('aeff_m:', np.sum(aeff_m),'\n')
        print('flux_m:', np.sum(flux_m),'\n')
        print('aeff_mbar:', np.sum(aeff_mbar),'\n')
        print('flux_mbar:', np.sum(flux_mbar),'\n')

    return integrate(integrand,'simps', Et,zr,Er)


def event_wrapper(param_list):
    E_index,z_index, alpha, params, npoints, null, spectral, tau = param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], param_list[5], param_list[6], param_list[7]
    return get_events(E_index=E_index, z_index=z_index, params=params, npoints=npoints, alpha=alpha, null=null, spectral_shift_parameters=spectral, tau=tau)

def sim_events(alpha, npoints, params=ic_params, null = False,multi=True, spectral_shift=[False, 2e3, 0.02], tau=False):
    res = np.empty((10,20))
    E_z_combinations =[] 
    for i in range(3,13):
        for j in range(20):
            if multi:
                E_z_combinations.append([i,j, alpha, params,npoints, null, spectral_shift, tau])
            if not multi:
                res[i-3][j] = event_wrapper([i,j, alpha, params,npoints, null, spectral_shift,tau])
    if multi:
        p = Pool()
        res = p.map(event_wrapper, E_z_combinations)
    
    res_array = np.array(res).reshape(-1, 20)
    return res_array


def wrap(p):
    return sim_events(0.99,25, multi=True, params=p)


def list_of_params(dict,dm_range, s24_range, s34_range=None, s24_eq_s34=False, s24_2x_s34=False):
    def update_dict(dict,p):
        dict2 = dict.copy()
        dict2.update(p)
        return dict2
    if s24_eq_s34:
        dict_list = [update_dict(dict,{'dm_41':v, 'theta_24': np.arcsin(np.sqrt(k))/2, 'theta_34': np.arcsin(np.sqrt(k))/2}) for k in s24_range for v in dm_range]
    elif s24_2x_s34:
        dict_list = [update_dict(dict,{'dm_41':v, 'theta_24': np.arcsin(np.sqrt(k))/2, 'theta_34': 2*np.arcsin(np.sqrt(k))/2}) for k in s24_range for v in dm_range]
    elif s34_range is not None:
        dict_list = [update_dict(dict,{'dm_41':v, 'theta_24': np.arcsin(np.sqrt(k))/2, 'theta_34': np.arcsin(np.sqrt(j))/2}) for j in s34_range for k in s24_range for v in dm_range]
    else:
        dict_list = [update_dict(dict,{'dm_41':v, 'theta_24': np.arcsin(np.sqrt(k))/2}) for k in s24_range for v in dm_range]
    return dict_list

def spectral_shift_factor(E, E_pivot=2e3, delta_gamma=0.02):
    return  (E/E_pivot)**-delta_gamma


interp_flux, interp_aeff, energy_resolution_models = get_interpolators()
if __name__ == '__main__':

    start = time.time()    
    dm41_range = np.logspace(-1,0,10)
    s24_range = np.logspace(-2,0,10)
    param_list = list_of_params(ic_params,dm41_range, s24_range,s24_range, short=True)
    print(get_events(5, 5, 0.99, 21, params=param_list[0], spectral_shift_parameters=[True, 2e3, 0.1], null=True))
    print(get_events(5, 5, 0.99, 21, params=param_list[0], spectral_shift_parameters=[False, 2e3, 0.1], null=True))