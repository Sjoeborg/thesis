import numpy as np
from dataProcesser import get_flux,get_aeff, generate_probabilities, get_probabilities, get_Etrue, get_interpolators_dc
from dataImporter import *
from multiprocessing import Pool
from functions import ic_params, integrate
import time
from scipy.stats import norm


def get_events(E_index, z_index, alpha, npoints, params=ic_params, spectral_shift_parameters=[False, 2e3, 0.02], null=False, tau=False):
    '''
    Assume zr == zt, and thus the zenith resolution function is 1.
    '''
    z_buckets = np.linspace(-1,1,9)
    E_buckets = np.logspace(0.75,1.75,9)

    Er = np.logspace(np.log10(E_buckets[E_index]), np.log10(E_buckets[E_index+1]), npoints)
    zr = np.linspace(z_buckets[z_index], z_buckets[z_index+1], npoints)

    Et = np.logspace(np.log10(norm.ppf(1-alpha, scale=0.24*Er[0], loc= Er[0])), 
                         np.log10(norm.ppf(alpha, scale=0.24*Er[-1], loc= Er[-1])),npoints)

    zt = -np.linspace(norm.ppf(1-alpha, scale=-0.1*zr[0], loc= -zr[0]), 
                         norm.ppf(alpha, scale=-0.1*zr[-1], loc= -zr[-1]),npoints)

    Eresolution_gaussian = norm.pdf(Et, scale = 0.24*Et, loc=Er)

    zresolution_gaussian = norm.pdf(-zt, scale = -0.1*zt, loc=-zr)

    zr_mesh,zt_mesh, Er_mesh, Et_mesh = np.meshgrid(zr,zt, Er, Et)

    aeff_m = 10#interp_aeff(Et_mesh,zt_mesh)
    aeff_mbar = 20#interp_aeff(Et_mesh,zt_mesh)

    if spectral_shift_parameters[0]:
        E_pivot = spectral_shift_parameters[1]
        delta_gamma = spectral_shift_parameters[2]
        factor = spectral_shift_factor(E = Et_mesh, E_pivot = E_pivot, delta_gamma=delta_gamma)   
        flux_m = factor*get_flux('m',Et_mesh,zt_mesh,interp_flux)
        flux_mbar = factor*get_flux('mbar',Et_mesh,zt_mesh,interp_flux)
    else:
        flux_m = get_flux('m',Et_mesh,zt_mesh,interp_flux)
        flux_mbar = get_flux('mbar',Et_mesh,zt_mesh,interp_flux)

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
    integrand *= Eresolution_gaussian * zresolution_gaussian* 2*np.pi * 240747841
 
    if np.isnan(np.sum(integrand)):
        print('aeff_m:', np.sum(aeff_m),'\n')
        print('flux_m:', np.sum(flux_m),'\n')
        print('aeff_mbar:', np.sum(aeff_mbar),'\n')
        print('flux_mbar:', np.sum(flux_mbar),'\n')

    from scipy.integrate import simps
    #return simps(integrand, Et_mesh)
    return integrate(integrand,'simps', Et,zt,-zr,Er)


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

def gather_specific_prob(flavor,npoints,En,zn, update=True):
    import os
    filenames=[]
    try:
        for file in os.listdir(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}/'):
            if file.endswith('.npy'):
                filenames.append(file[0:-4])
        try:
            df = pickle.load(open(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}.p','rb'))
        except FileNotFoundError:
            df = pd.DataFrame(index=[f'E{En}z{zn}'], dtype='object')

        for file in filenames:
            array = np.load(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}/{file}.npy')
            try:
                df.insert(loc=0,column=file, value=[array])
            except ValueError: 
                if update:  # If entry already exists, overwrite/update it
                    df[file][f'E{En}z{zn}'] = array
        pickle.dump(df,open(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}.p','wb'))
        for file in filenames:
            os.remove(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}/{file}.npy')
    except FileNotFoundError:
        pass
interp_flux, interp_aeff,=get_interpolators_dc()
if __name__ == '__main__':
    res = get_events(0, 0, 0.99, 5, params=ic_params, spectral_shift_parameters=[False, 2e3, 0.02], null=True, tau=False)
    print(res)