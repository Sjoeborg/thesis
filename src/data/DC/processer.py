import sys,os
if __name__ == '__main__':
    sys.path.append('./src/probability')
    sys.path.append('./src/data')
import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator as CT
from scipy.interpolate import NearestNDInterpolator
from functions import mass_dict,dc_params
from DC.importer import systematics2015_DC, get_aeff_df_DC, get_flux_df_DC, events2018_DC, no_osc2018_DC
from dict_hash import sha256
import pandas as pd
from numerical import P_num 
import h5py
from scipy.stats import lognorm
import pickle
import time

pdg_dict={'e':12,'m':14,'t':16}
Ebins_2018 = [5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413]
zbins_2018 = [-1., -0.75, -0.5 , -0.25,  0., 0.25, 0.5, 0.75, 1.]


def get_flux(flavor,E,z,df):
    '''
    Returns flux for a set of flavor, energy [GeV], and z=cos(theta_z).
    '''
    z = np.clip(z,a_min=-1,a_max=0.95)#I dont have flux for this bin, so just set it to 0.95.
    try:
        flux_avg = df[f'{flavor}_flux'][0](E,z)
    except KeyError:
        raise KeyError('NYI for tau flux')
    return np.abs(flux_avg) 


def get_aeff_DC(E,interpolator):
    try:
        return interpolator(np.log10(E))
    except ValueError: #extrapolate
        y1 = interpolator(1.13)
        y2 = interpolator(1.2)
        return (y2-y1)/(1.2-1.13) * np.log10(E)

def interpolate_flux_DC(recompute=False):
    '''
    Returns a df of the interpolated fluxes. 
    '''
    colnames = ['m_flux', 'mbar_flux', 'e_flux', 'ebar_flux']
    if not recompute:
        try:
            inter_df = pickle.load(open('./pre_computed/flux_interpolator.p','rb'))
        except:
            interpolate_flux_DC(recompute=True)
    else:
        df = get_flux_df_DC()
        E = df.GeV
        z_avg = (df.z_min + df.z_max)/2

        points_avg = np.array([E,z_avg]).T

        interp_list=[]
        for flavor in colnames:
            phi = df[flavor]
            values=np.array(phi)

            f_avg = CT(points_avg, values,rescale=True) #Rescale seems to have no effect, but is good according to doc
            interp_list.append([f_avg])

        inter_df = pd.DataFrame(np.transpose(interp_list), columns=colnames)
        pickle.dump(inter_df,open('./pre_computed/flux_interpolator.p','wb'))
    return inter_df


def bin_flux_factors_DC(E_df, z_df):
    z_buckets = np.linspace(-1,1,9)
    E_buckets = np.logspace(0.75,1.75,9)
    E_res=[]
    z_res=[]
    for i in range(9):
        mean_per_bin = E_df[E_df.E.between(E_buckets[i], E_buckets[i+1])].factor.mean()
        E_res.append(mean_per_bin)
    for i in range(9):
        mean_per_bin = z_df[z_df.E.between(z_buckets[i], z_buckets[i+1])].factor.mean()
        z_res.append(
            mean_per_bin)
    return np.array(E_res),np.array(z_res)

def get_probabilities_DC(flavor_from, flavor_to, Ebin, zbin, param_dict,anti,pid,ndim, nsi):
    if not nsi:
        param_dict = dc_params
    hashed_param_name = sha256(param_dict)
    if anti:
        flavor_from = 'a' + flavor_from
        flavor_to = 'a' + flavor_to
    try:
        f = h5py.File(f'./pre_computed/DC/E{Ebin}z{zbin}.hdf5', 'r')
    except OSError:
        raise KeyError(f'E{Ebin}z{zbin}.hdf5 doesnt exist in ./pre_computed/DC/')
    try:
        fh = f[f'{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name}']
    except KeyError:
        f.close()
        raise KeyError(f'{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name} doesnt exist in E{Ebin}z{zbin}.hdf5')
    res = fh[()]
    f.close()
    return res

def generate_probabilities_DC(flavor_from, flavor_to, E_range,z_range,E_bin,z_bin,param_dict,anti,pid, ndim=4, nsi=True, save=True):
    if not nsi:
        param_dict = dc_params
    prob = np.array([P_num(flavor_from=flavor_from, E=E_range[i], ndim = ndim, anti=anti,params=param_dict,zenith=z, nsi=nsi)[mass_dict[flavor_to],-1] for i,z in enumerate(z_range)]).reshape(-1,1)
    if save:
        hashed_param_name = sha256(param_dict)
        if anti:
            flavor_from = 'a' + flavor_from
            flavor_to = 'a' + flavor_to

        f = h5py.File(f'./pre_computed/DC/E{E_bin}z{z_bin}.hdf5', 'a')
        try:
            dset = f.create_dataset(f'{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name}', data=prob, chunks=True)
            for key in param_dict.keys():
                dset.attrs[key] = param_dict[key]
            f.close()
        except RuntimeError:
            if overwrite:
                dset = f[f'{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name}']
                dset[...] = prob    
            else:
                print(f'{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name} already exists, skipping')
            f.close()
            return prob
        if E_bin == 5 and z_bin == 5 and flavor_from == 'am' and flavor_to == 'am' and pid == 1:
            with open(f'./pre_computed/DC/hashed_params.csv','a') as fd:
                fd.write(f'{param_dict};{hashed_param_name}\n')
    return prob

def process_aeff(df_list):
    eff_df = systematics2015_DC()
    
    DOMeff = NearestNDInterpolator(np.array([eff_df.E_avg, eff_df.z_avg]).T, eff_df.DOMeff, rescale=True)
    ICEeff = NearestNDInterpolator(np.array([eff_df.E_avg, eff_df.z_avg]).T, eff_df.ICEeff, rescale=True)

    for df in df_list:
        df.loc[:,'DOMeff'] = DOMeff(df.E_avg,df.z_avg)
        df.loc[:,'ICEeff'] = ICEeff(df.E_avg,df.z_avg)
    return df_list
        

def interpolate_aeff_DC(recompute=False):
    if not recompute:
        try:
            inter = pickle.load(open('./pre_computed/aeff_dc_interpolator.p','rb'))
        except:
            raise FileNotFoundError('File aeff_dc_interpolator.p´ not present in ´./pre_computed/´. Run ´interpolate_aeff_dc()´ with recompute = True to generate it.')
    else:
        aeff_df = get_aeff_df_DC()
        from scipy.interpolate import interp1d
        inter = interp1d(aeff_df.logE, aeff_df.Aeff)
        pickle.dump(inter,open('./pre_computed/aeff_dc_interpolator.p','wb'))
    return inter


def get_binned_DC(pid,E_bin,z_bin,df):
    df1 = (df.query(f'pid=={pid}')
             .query(f'reco_energy<{Ebins_2018[E_bin+1]}')
             .query(f'reco_energy>{Ebins_2018[E_bin]}')
             .query(f'reco_coszen<{zbins_2018[z_bin+1]}')
             .query(f'reco_coszen>{zbins_2018[z_bin]}'))

    return df1




def get_interpolators_DC(recompute_flux=False, recompute_aeff=False):
    interp_flux = interpolate_flux_DC(recompute_flux)
    interp_aeff = interpolate_aeff_DC(recompute_aeff)

    return interp_flux, interp_aeff

def get_hist(df, weight):
    '''retrieve histogram in (energy x coszen) space'''
    hist, _, _ = np.histogram2d(df['reco_energy'], df['reco_coszen'], bins=(Ebins_2018, zbins_2018), weights=df[weight])
    return hist

def multiply_fluxes(MC_df):
    interp_flux = interpolate_flux_DC()
    e_mask = (MC_df['pdg'] == 12)
    m_mask = (MC_df['pdg'] == 14)
    t_mask = (MC_df['pdg'] == 16)

    ebar_mask = (MC_df['pdg'] == -12)
    mbar_mask = (MC_df['pdg'] == -14)
    tbar_mask = (MC_df['pdg'] == -16)

    rate_weights = np.zeros_like(MC_df['weight'])

    rate_weights[e_mask] = MC_df[e_mask]['weight'] * get_flux('e',MC_df[e_mask]['true_energy'], MC_df[e_mask]['true_coszen'], interp_flux)
    rate_weights[m_mask] = MC_df[m_mask]['weight'] * get_flux('m',MC_df[m_mask]['true_energy'], MC_df[m_mask]['true_coszen'], interp_flux)
    rate_weights[ebar_mask] = MC_df[ebar_mask]['weight'] * get_flux('ebar',MC_df[ebar_mask]['true_energy'], MC_df[ebar_mask]['true_coszen'], interp_flux)
    rate_weights[mbar_mask] = MC_df[mbar_mask]['weight'] * get_flux('mbar',MC_df[mbar_mask]['true_energy'], MC_df[mbar_mask]['true_coszen'], interp_flux)

    MC_df['rate_weight'] = rate_weights
    return MC_df

def process_systematics(df, systematics, pid):
    result = []
    pdg_dict = {'e': 12, 'mu': 14, 'tau':16}
    for key in systematics.keys():
        if key != 'nc':
            temp = df.query(f'abs(pdg)=={pdg_dict[key]}')
        else:
            temp = df.query('type == 0')

        raw = get_hist(temp.query(f'pid == {pid}'), weight='rate_weight')
        correction = get_hist(systematics[key].query(f'pid=={pid}'), weight='correction_factor')

        result.append(raw* correction)
    result = np.sum(np.array(result),axis=0)

    return result

def normalization_factors(no_osc_tracks, no_osc_cascades):
    track_null = no_osc2018_DC(pid=1)[1].values
    cascade_null = no_osc2018_DC(pid=0)[1].values
    return track_null/no_osc_tracks, cascade_null/no_osc_cascades

if __name__ == '__main__':
    pass
