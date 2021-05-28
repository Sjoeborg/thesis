import sys,os
if __name__ == '__main__':
    #os.chdir('../../')
    sys.path.append('./src/probability')
    sys.path.append('./src/data')
import numpy as np
from IC.processer import get_flux, generate_probabilities, get_probabilities, get_Etrue, get_interpolators,generate_probabilitiesOLD
from IC.importer import *
from multiprocessing import Pool
from functions import ic_params, integrate
import time
from scipy.stats import lognorm


def get_events(E_index, z_index, alpha, npoints, params=ic_params, spectral_shift_parameters=[False, 2e3, 0.02], null=False, tau=False, nsi=False, ndim=4):
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
        #try:
        #    Pmm = get_probabilities('m', 'm', E_index,z_index,params,False,npoints,ndim=ndim)
        #except KeyError:
        Pmm = generate_probabilities('m','m',Et,zr,E_index, z_index, params,False,npoints,ndim=ndim,nsi=nsi, save=False)
            #gather_specific_prob('Pmm',npoints,E_index,z_index,update=True,ndim=ndim)
            #Pmm = get_probabilities('m', 'm', E_index,z_index,params,False,npoints,ndim=ndim)

        #try:
        #    P_amam = get_probabilities('m', 'm', E_index,z_index,params,True,npoints,ndim=ndim)
        #except KeyError:
        P_amam = generate_probabilities('m','m',Et,zr,E_index, z_index, params,True,npoints,ndim=ndim,nsi=nsi, save=False)
            #gather_specific_prob('Pamam',npoints,E_index,z_index,update=True,ndim=ndim)
            #P_amam = get_probabilities('m', 'm', E_index,z_index,params,True,npoints,ndim=ndim)
        
        #try:
        #   Pem = get_probabilities('e', 'm', E_index,z_index,params,False,npoints,ndim=ndim)
        #except KeyError:
        Pem = generate_probabilities('e','m',Et,zr,E_index, z_index, params,False,npoints,ndim=ndim,nsi=nsi, save=False)
            #gather_specific_prob('Pem',npoints,E_index,z_index,update=True,ndim=ndim)
            #Pem = get_probabilities('e', 'm', E_index,z_index,params,False,npoints,ndim=ndim)

        #try:
        #    P_aeam = get_probabilities('e', 'm', E_index,z_index,params,True,npoints,ndim=ndim)
        #except KeyError:
        P_aeam=generate_probabilities('e','m',Et,zr,E_index, z_index, params,True,npoints,ndim=ndim,nsi=nsi, save=False)
            #gather_specific_prob('Paeam',npoints,E_index,z_index,update=True,ndim=ndim)
            #P_aeam = get_probabilities('e', 'm', E_index,z_index,params,True,npoints,ndim=ndim)

        if tau:
            #try:
            #    Pmt = get_probabilities('m', 't', E_index,z_index,params,False,npoints,ndim=ndim)
            #except KeyError:
            Pmt=generate_probabilities('m','t',Et,zr,E_index, z_index, params,False,npoints,ndim=ndim,nsi=nsi, save=False)
                #gather_specific_prob('Pmt',npoints,E_index,z_index,update=True,ndim=ndim)
                #Pmt = get_probabilities('m', 't', E_index,z_index,params,False,npoints,ndim=ndim)

            #try:
            #    P_amat = get_probabilities('m', 't', E_index,z_index,params,True,npoints,ndim=ndim)
            #except KeyError:
            P_amat=generate_probabilities('m','t',Et,zr,E_index, z_index, params,True,npoints,ndim=ndim,nsi=nsi, save=False)
                #gather_specific_prob('Pamat',npoints,E_index,z_index,update=True,ndim=ndim)
                #P_amat = get_probabilities('m', 't', E_index,z_index,params,True,npoints,ndim=ndim)

            Pmm = Pmm + 0.1739*Pmt
            P_amam = P_amam + 0.1739*P_amat
    

        flux_e = get_flux('e',Et_mesh,zr_mesh,interp_flux)
        flux_ebar = get_flux('ebar',Et_mesh,zr_mesh,interp_flux)
        integrand = aeff_m*flux_m*Pmm + aeff_mbar*flux_mbar*P_amam + aeff_m*flux_e*Pem + aeff_mbar*flux_ebar*P_aeam
    else:
        integrand = aeff_m*flux_m + aeff_mbar*flux_mbar
    integrand *= resolution_gaussian* 2*np.pi * 240747841

    return integrate(integrand,'simps', Et,zr,Er)


def event_wrapper(param_list):
    E_index,z_index, alpha, params, npoints, null, spectral, tau, ndim = param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], param_list[5], param_list[6], param_list[7], param_list[8]
    return get_events(E_index=E_index, z_index=z_index, params=params, npoints=npoints, alpha=alpha, null=null, spectral_shift_parameters=spectral, tau=tau, ndim=ndim)

def sim_events(alpha, npoints, params=ic_params, null = False,multi=True, spectral_shift=[False, 2e3, 0.02], tau=False, nsi=False, ndim=4):
    if nsi:
        E_offset = 0 #For nsi, include all bins
    else:
        E_offset = 3 # For non-nsi, exclude bottom 3 bins
    res = np.empty((13-E_offset,20))
    E_z_combinations =[] 
    for E_bin in range(E_offset,13):
        for z_bin in range(20):
            if multi:
                E_z_combinations.append([E_bin,z_bin, alpha, params,npoints, null, spectral_shift, tau])
            if not multi:
                res[E_bin-E_offset][z_bin] = event_wrapper([E_bin,z_bin, alpha, params,npoints, null, spectral_shift,tau, ndim])
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

def gather_specific_prob(flavor,npoints,En,zn, update=True, ndim=4):
    import os
    filenames=[]
    try:
        for file in os.listdir(f'./pre_computed/{ndim}gen/{flavor}/{npoints}/E{En}z{zn}/'):
            if file.endswith('.npy'):
                filenames.append(file[0:-4])
        try:
            df = pickle.load(open(f'./pre_computed/{ndim}gen/{flavor}/{npoints}/E{En}z{zn}.p','rb'))
        except FileNotFoundError:
            df = pd.DataFrame(index=[f'E{En}z{zn}'], dtype='object')

        for file in filenames:
            array = np.load(f'./pre_computed/{ndim}gen/{flavor}/{npoints}/E{En}z{zn}/{file}.npy')
            try:
                df.insert(loc=0,column=file, value=[array])
            except ValueError: 
                if update:  # If entry already exists, overwrite/update it
                    df[file][f'E{En}z{zn}'] = array
        pickle.dump(df,open(f'./pre_computed/{ndim}gen/{flavor}/{npoints}/E{En}z{zn}.p','wb'))
        for file in filenames:
            os.remove(f'./pre_computed/{ndim}gen/{flavor}/{npoints}/E{En}z{zn}/{file}.npy')
    except (FileNotFoundError,ValueError):
        print(f'Could not load {flavor} E{En}z{zn}, skipping it')

interp_flux, interp_aeff, energy_resolution_models = get_interpolators()
if __name__ == '__main__':
   print(get_events(5, 11, 0.99, 13))
