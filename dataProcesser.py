import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator as CT
from functions import mass_dict
from dataImporter import get_flux_df,get_aeff_df
from dict_hash import sha256
import pandas as pd
from numerical import wrapper 
from scipy.stats import lognorm
import pickle
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
def get_flux(flavor,E,z,df):
    '''
    Returns flux for a set of flavor, energy [GeV], and z=cos(theta_z).
    Returns the absolute value since the flux has a small negative dip at the boundary between the interpolated and extrapolated fluxes at 1e4 GeV. TODO:Fix this in fit_flux()
    '''
    try:
        flux_avg = df[f'{flavor}_flux'][0](E,z)
    except KeyError:
        raise KeyError('NYI for tau flux')
    return np.abs(flux_avg) 

def get_aeff(anti,E,z,df_list):
    if anti:
        index = 1
    else:
        index = 0
    aeff = df_list[index](E,z)
    return aeff


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


def bin_flux_factors(E_df, z_df):
    E_bins  = 500*10**(np.linspace(0.0,1.3,14))
    z_bins = np.linspace(-1.,0.,21)
    E_res=[]
    z_res=[]
    for i in range(13):
        mean_per_bin = E_df[E_df.E.between(E_bins[i], E_bins[i+1])].factor.mean()
        E_res.append(mean_per_bin)
    for i in range(20):
        mean_per_bin = z_df[z_df.E.between(z_bins[i], z_bins[i+1])].factor.mean()
        z_res.append(
            mean_per_bin)
    return np.array(E_res),np.array(z_res)



def interpolate_aeff(recompute=False):
    if not recompute:
        try:
            aeff_list = pickle.load(open('./pre_computed/aeff_interpolator.p','rb'))
        except:
            raise FileNotFoundError('File aeff_interpolator.p´ not present in ´./pre_computed/´. Run ´interpolate_aeff()´ with recompute = True to generate it.')
        try:
            df_list = pickle.load(open('./pre_computed/aeff.p','rb'))
        except:
            raise FileNotFoundError('File aeff.p´ not present in ´./pre_computed/´. Run get_aeff_df() to generate it.')
    else:
        df_list = get_aeff_df()
        aeff_list = []
        for df in df_list:
            E = df.Etrue
            z = df.ztrue
            aeff = np.array(df.aeff)
            f = CT(np.array([E,z]).T, aeff)
            aeff_list.append(f)
        pickle.dump(aeff_list,open('./pre_computed/aeff_interpolator.p','wb'))
    return aeff_list

def get_probabilitiesOLD(flavor_from, flavor_to, E_bin,z_bin,params,anti,N, ndim=4):
    '''
    Name of resulting .npy is the sha256 hash of the parameter dictionary used to generate the probablities.
    '''

    filename = sha256(params) #Get hash of parm dict to be used as filename
    if anti:
        file_dir = f'./pre_computed/{ndim}gen/Pa{flavor_from}a{flavor_to}/{N}/E{E_bin}z{z_bin}/'
    else:
        file_dir = f'./pre_computed/{ndim}gen/P{flavor_from}{flavor_to}/{N}/E{E_bin}z{z_bin}/'
    res = np.load(file_dir + filename+'.npy')
    return res

def get_probabilities(flavor_from, flavor_to, E_bin,z_bin,params,anti,N, ndim=4):
    '''
    Name of resulting .npy is the sha256 hash of the parameter dictionary used to generate the probablities.
    '''

    hashed_param_name = sha256(params) #Get hash of parm dict to be used as filename
    if anti:
        file_dir = f'./pre_computed/{ndim}gen/Pa{flavor_from}a{flavor_to}/{N}/E{E_bin}z{z_bin}/'
    else:
        file_dir = f'./pre_computed/{ndim}gen/P{flavor_from}{flavor_to}/{N}/E{E_bin}z{z_bin}/'
    df = pickle.load(open(f'{file_dir}df.p','rb'))
    res = df[hashed_param_name][f'E{E_bin}z{z_bin}']
    return res

def generate_probabilities(flavor_from, flavor_to, E_range,z_range,E_bin,z_bin,params,anti,N, ndim=4):
    '''
    Name of resulting .csv is the sha256 hash of the parameter dictionary used to generate the probablities.
    '''

    filename = sha256(params) #Get hash of parm dict to be used as filename
    if anti:
        file_dir = f'./pre_computed/{ndim}gen/Pa{flavor_from}a{flavor_to}/{N}/E{E_bin}z{z_bin}/'
    else:
        file_dir = f'./pre_computed/{ndim}gen/P{flavor_from}{flavor_to}/{N}/E{E_bin}z{z_bin}/'
    res = np.array([wrapper([flavor_from, E_range,z, anti, params, ndim])[mass_dict[flavor_to]] for z in z_range])
    try:
        np.save(file_dir + filename,res)
    except FileNotFoundError:
        try:
            os.makedirs(file_dir)
        except FileExistsError: #Sometimes, the dir was already created by another thread
            pass
        np.save(file_dir + filename,res)
    if anti:
        with open(f'./pre_computed/{ndim}gen/Pa{flavor_from}a{flavor_to}/{N}/hashed_params.csv','a') as fd:
            fd.write(f'{params};{filename}\n')
    else:
        with open(f'./pre_computed/{ndim}gen/P{flavor_from}{flavor_to}/{N}/hashed_params.csv','a') as fd:
            fd.write(f'{params};{filename}\n')
    

def train_energy_resolution(recompute=False):
    #TODO: Finetune this
    if not recompute:
        try:
            gpr = pickle.load(open("../energy_resolution_models.p", "rb"))
        except:
            raise FileNotFoundError('File energy_resolution_models.p´ not present in ´../´. Run ´train_energy_resolution()´ with recompute=True to generate it.')
    else:
        filename = '~/NuFSGenMC_nominal.dat'
        df = pd.read_csv(filename, delimiter=' ', names= ['pdg', 'Ereco', 'zreco', 'Etrue', 'ztrue', 'mcweight', 'flux_pion', 'flux_kaon'], skiprows=12)
        df.Ereco = np.round(df.Ereco,0)
        df = df.groupby('Ereco').median().reset_index()
        df['Ebin'] = pd.cut(df.Ereco, bins=500*10**np.linspace(0.0,1.3,14))
        if len(df) > 5000:
            df_subsetted = df.sample(5000, random_state=0)
        X = np.array(np.log(df_subsetted.Ereco)).reshape(-1,1)
        y = np.log(df_subsetted.Etrue)
        kernel2  = 1.0 * RBF() + WhiteKernel(noise_level=3)
        gpr = GaussianProcessRegressor(kernel=kernel2,random_state=0).fit(X, y)
        pickle.dump(gpr, open("../energy_resolution_models.p", "wb"))
    return gpr
    

def get_Etrue(model, npoints, left_alpha, right_alpha,E_index=None,Ereco=False):
    E_buckets = 500*10**(np.linspace(0.0,1.3,14))
    if not E_index is None:
        Ereco = np.logspace(np.log10(E_buckets[E_index]), np.log10(E_buckets[E_index+1]), npoints)
    mu_base_e, std_base_e = model.predict(np.log(Ereco).reshape(-1,1), return_std=True)

    Etrue = np.logspace(np.log10(lognorm.ppf(1-left_alpha, s=std_base_e[0], scale= np.exp(mu_base_e[0]))), 
                         np.log10(lognorm.ppf(right_alpha, s=std_base_e[-1], scale= np.exp(mu_base_e[-1]))),npoints)
    return Etrue, mu_base_e, std_base_e

def get_interpolators(recompute_flux=False, recompute_aeff=False, recompute_energy_res=False):
    interp_flux = interpolate_flux(recompute_flux)
    interp_aeff = interpolate_aeff(recompute_aeff)
    from old.deprecated import get_Aeff_df_2012, interpolate_Aeff_2012
    aeff_df = get_Aeff_df_2012()
    interp_aeff = interpolate_Aeff_2012(aeff_df)
    gpr_models = train_energy_resolution(recompute_energy_res)

    return interp_flux, interp_aeff, gpr_models

if __name__ == '__main__':
    pass
