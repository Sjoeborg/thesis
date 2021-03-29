import sys,os
if __name__ == '__main__':
    os.chdir('../../')
    sys.path.append('./src/probability')
    sys.path.append('./src/data')
import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator as CT
from functions import mass_dict
from importer import get_flux_df,get_aeff_df,get_flux_df_DC, get_aeff_df_dc
from importerDC import get_systematics, get_aeff_df_dc
from dict_hash import sha256
import pandas as pd
#from numerical import wrapper 
from scipy.stats import lognorm
import pickle
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

def get_flux(flavor,E,z,df):
    '''
    Returns flux for a set of flavor, energy [GeV], and z=cos(theta_z).
    '''
    try:
        flux_avg = df[f'{flavor}_flux'][0](E,z)
    except KeyError:
        raise KeyError('NYI for tau flux')
    return np.abs(flux_avg) 


def get_aeff_dc(E,interpolator):
    try:
        return interpolator(np.log10(E))
    except ValueError: #extrapolate
        y1 = interpolator(1.13)
        y2 = interpolator(1.2)
        return (y2-y1)/(1.2-1.13) * np.log10(E)

def interpolate_flux(recompute=False):
    '''
    Returns a df of the interpolated fluxes. 
    '''
    colnames = ['m_flux', 'mbar_flux', 'e_flux', 'ebar_flux']
    if not recompute:
        try:
            inter_df = pickle.load(open('./pre_computed/flux_interpolator.p','rb'))
        except:
            raise FileNotFoundError('File ´flux_interpolator.p´ not present in ´./pre_computed/´. Rerun with recompute = True to generate it.')
    else:
        df = get_flux_df()
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

def interpolate_flux_DC(recompute=False):
    '''
    Returns a df of the interpolated fluxes. 
    '''
    colnames = ['m_flux', 'mbar_flux', 'e_flux', 'ebar_flux']
    if not recompute:
        try:
            inter_df = pickle.load(open('./pre_computed/flux_interpolator_DC.p','rb'))
        except:
            raise FileNotFoundError('File ´flux_interpolator_DC.p´ not present in ´./pre_computed/´. Rerun with recompute = True to generate it.')
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
        pickle.dump(inter_df,open('./pre_computed/flux_interpolator_DC.p','wb'))
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


def process_aeff(df_list):
    eff_df = get_systematics()
    
    
    from scipy.interpolate import NearestNDInterpolator
    DOMeff = NearestNDInterpolator(np.array([eff_df.E_avg, eff_df.z_avg]).T, eff_df.DOMeff, rescale=True)
    ICEeff = NearestNDInterpolator(np.array([eff_df.E_avg, eff_df.z_avg]).T, eff_df.ICEeff, rescale=True)

    for df in df_list:
        df.loc[:,'DOMeff'] = DOMeff(df.E_avg,df.z_avg)
        df.loc[:,'ICEeff'] = ICEeff(df.E_avg,df.z_avg)
    return df_list
        

def interpolate_aeff_dc(recompute=False):
    if not recompute:
        try:
            inter = pickle.load(open('./pre_computed/aeff_dc_interpolator.p','rb'))
        except:
            raise FileNotFoundError('File aeff_dc_interpolator.p´ not present in ´./pre_computed/´. Run ´interpolate_aeff_dc()´ with recompute = True to generate it.')
    else:
        aeff_df = get_aeff_df_dc()
        from scipy.interpolate import interp1d
        inter = interp1d(aeff_df.logE, aeff_df.Aeff)
        pickle.dump(inter,open('./pre_computed/aeff_dc_interpolator.p','wb'))
    return inter





def get_interpolators_dc(recompute_flux=False, recompute_aeff=False):
    interp_flux = interpolate_flux_DC(recompute_flux)
    interp_aeff = interpolate_aeff_dc(recompute_aeff)

    return interp_flux, interp_aeff

if __name__ == '__main__':
    df_list = get_aeff_df_dc()
    process_aeff(df_list)
