import numpy as np
from dataProcesser import get_flux,get_aeff, generate_probabilities, get_probabilities, get_Etrue, get_interpolators_dc,get_aeff_dc
from dataImporter import *
from multiprocessing import Pool
from functions import ic_params, integrate
import time
from scipy.stats import norm


def get_events(E_index, z_index, alpha, npoints, params=ic_params, spectral_shift_parameters=[False, 2e3, 0.02], null=False, tau=False):
    z_buckets = [-1., -0.75, -0.5 , -0.25,  0., 0.25, 0.5, 0.75, 1.]
    E_buckets = [5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413]


    Et, zt = get_true(flavor,pid,E_bin,z_bin,df)
    

    if spectral_shift_parameters[0]:
        E_pivot = spectral_shift_parameters[1]
        delta_gamma = spectral_shift_parameters[2]
        factor = spectral_shift_factor(E = Et_mesh, E_pivot = E_pivot, delta_gamma=delta_gamma)   
    else:
        factor=1

    if not null:
        try:
            Pmm = get_probabilities('m', 'm', E_index,z_index,params,False,npoints)
        except KeyError:
            generate_probabilities('m','m',Et,zt,E_index, z_index, params,False,npoints,ndim=4)
            gather_specific_prob('Pmm',npoints,E_index,z_index,update=True)
            Pmm = get_probabilities('m', 'm', E_index,z_index,params,False,npoints)

        try:
            P_amam = get_probabilities('m', 'm', E_index,z_index,params,True,npoints)
        except KeyError:
            generate_probabilities('m','m',Et,zt,E_index, z_index, params,True,npoints,ndim=4)
            gather_specific_prob('Pamam',npoints,E_index,z_index,update=True)
            P_amam = get_probabilities('m', 'm', E_index,z_index,params,True,npoints)

        if tau:
            try:
                Pmt = get_probabilities('m', 't', E_index,z_index,params,False,npoints)
            except KeyError:
                generate_probabilities('m','t',Et,zt,E_index, z_index, params,False,npoints,ndim=4)
                gather_specific_prob('Pmt',npoints,E_index,z_index,update=True)
                Pmt = get_probabilities('m', 't', E_index,z_index,params,False,npoints)

            try:
                P_amat = get_probabilities('m', 't', E_index,z_index,params,True,npoints)
            except KeyError:
                generate_probabilities('m','t',Et,zt,E_index, z_index, params,True,npoints,ndim=4)
                gather_specific_prob('Pamat',npoints,E_index,z_index,update=True)
                P_amat = get_probabilities('m', 't', E_index,z_index,params,True,npoints)

            Pmm = Pmm + 0.1739*Pmt
            P_amam = P_amam + 0.1739*P_amat
    

        integrand = aeff_m*flux_m*Pmm + aeff_mbar*flux_mbar*P_amam
    else:
        integrand = aeff_m*flux_m + aeff_mbar*flux_mbar
    integrand *= Eresolution_gaussian * 2*np.pi * 240747841
    scale_factor = np.arccos(zr.max()) - np.arccos(zr.min())
    '''
    if np.isnan(np.sum(integrand)):
        print('aeff_m:', np.sum(aeff_m),'\n')
        print('flux_m:', np.sum(flux_m),'\n')
        print('aeff_mbar:', np.sum(aeff_mbar),'\n')
        print('flux_mbar:', np.sum(flux_mbar),'\n')
    '''
    return -integrate(integrand,'simps', Et,zt,zr,Er) #/ scale_factor


def event_wrapper(param_list):
    E_index,z_index, alpha, params, npoints, null, spectral, tau = param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], param_list[5], param_list[6], param_list[7]
    return get_events(E_index=E_index, z_index=z_index, params=params, npoints=npoints, alpha=alpha, null=null, spectral_shift_parameters=spectral, tau=tau)

def sim_events(alpha, npoints, params=ic_params, null = False,multi=True, spectral_shift=[False, 2e3, 0.02], tau=False, nsi=False):
    res = np.empty((8,4))
    E_z_combinations =[] 
    for E_bin in range(8):
        for z_bin in range(4):
            if multi:
                E_z_combinations.append([E_bin,z_bin, alpha, params,npoints, null, spectral_shift, tau])
            if not multi:
                res[E_bin][z_bin] = event_wrapper([E_bin,z_bin, alpha, params,npoints, null, spectral_shift,tau])
    if multi:
        p = Pool()
        res = p.map(event_wrapper, E_z_combinations)
    
    res_array = np.array(res).reshape(-1, 4)
    return res_array


def wrap(p):
    return sim_events(0.99,25, multi=True, params=p)



def spectral_shift_factor(E, E_pivot=2e3, delta_gamma=0.02):
    return  (E/E_pivot)**-delta_gamma

interp_flux, interp_aeff,=get_interpolators_dc()
if __name__ == '__main__':
    #res = sim_events(0.99, 5, params=ic_params, null = True,multi=False, spectral_shift=[False, 2e3, 0.02], tau=False, nsi=False)
    res = get_events(7,0,0.99,5,null=True)
    print(res)