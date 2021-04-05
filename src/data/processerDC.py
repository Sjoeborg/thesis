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

def get_probabilitiesDC(flavor_from, flavor_to, Ebin, zbin, param_dict,anti,N,ndim):
    hashed_param_name = sha256(param_dict)
    if anti:
        flavor_from = 'a' + flavor_from
        flavor_to = 'a' + flavor_to
    try:
        f = h5py.File(f'./pre_computed/DC/E{Ebin}z{zbin}.hdf5', 'r')
    except OSError:
        raise KeyError(f'E{Ebin}z{zbin}.hdf5 doesnt exist in ./pre_computed/DC/')
    try:
        fh = f[f'{ndim}gen/P{flavor_from}{flavor_to}/{N}/{hashed_param_name}']
    except KeyError:
        f.close()
        raise KeyError(f'{ndim}gen/P{flavor_from}{flavor_to}/{N}/{hashed_param_name} doesnt exist in E{Ebin}z{zbin}.hdf5')
    res = fh[()]
    f.close()
    return res

def generate_probabilitiesDC(flavor_from, flavor_to, E_range,z_range,E_bin,z_bin,params,anti,N, ndim=4, nsi=False):
    prob = np.array([wrapper([flavor_from, E_range,z, anti, params, ndim, nsi])[mass_dict[flavor_to]] for z in z_range])
    hashed_param_name = sha256(params)
    if anti:
        flavor_from = 'a' + flavor_from
        flavor_to = 'a' + flavor_to
    f = h5py.File(f'./pre_computed/DC/E{E_bin}z{z_bin}.hdf5', 'a')
    try:
        dset = f.create_dataset(f'{ndim}gen/P{flavor_from}{flavor_to}/{N}/{hashed_param_name}', data=prob, chunks=True)
        for key in params.keys():
            dset.attrs[key] = params[key]
        f.close()
    except RuntimeError:
        print(f'{ndim}gen/P{flavor_from}{flavor_to}/{N}/{hashed_param_name} already exists, skipping')
        f.close()
        return
    if E_bin == 5 and z_bin == 5 and flavor_from == 'am':
        with open(f'./pre_computed/DC/hashed_params.csv','a') as fd:
            fd.write(f'{params};{hashed_param_name}\n')

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

def get_true_models():
    filename = './src/data/files/DC/sample_b/neutrino_mc.csv'
    df = (pd.read_csv(filename)
        .query('pdg == 14 or pdg == -14') #only muon (anti)neutrinos
        .query('pid == 1 ')) # only tracks
    df['Ebin'] = pd.cut(df.reco_energy, bins=Ereco, labels=False)
    df['zbin'] = pd.cut(df.reco_coszen, bins=zreco, labels=False)
    
    def train(df):
        X = np.array([df.reco_coszen, np.log(df.reco_energy)]).reshape(-1, 2)
        y = np.array([df.true_coszen, np.log(df.true_energy)]).reshape(-1, 2)
        kernel2  = 1.0 * RBF() + WhiteKernel(noise_level=3)
        gpr = GaussianProcessRegressor(kernel=kernel2,random_state=0).fit(X, y)
        return gpr

    models = []
    for zbin in range(8):
        for Ebin in range(8):
            print(f'training {Ebin} {zbin}')
            df_sub = df.query(f'Ebin=={Ebin} ad zbin=={zbin}')
            models.append(train(df_sub))
    models = np.array(models).reshape(8,8)
    return models


def get_true(models, npoints, left_alpha, right_alpha,E_bin,z_bin):
    E_buckets = np.array([5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413])
    z_buckets = np.array([-1., -0.75, -0.5 , -0.25,  0., 0.25, 0.5, 0.75, 1.])

    mu_base_e, std_base_e = model.predict(np.array([np.log(E_buckets[E_bin]), z_buckets[z_bin]]).reshape(-1,2), return_std=True)

    Etrue = np.logspace(np.log10(lognorm.ppf(1-left_alpha, s=std_base_e[0,0], scale= np.exp(mu_base_e[0,0]))), 
                         np.log10(lognorm.ppf(right_alpha, s=std_base_e[-1,0], scale= np.exp(mu_base_e[-1,0]))),npoints)
    ztrue = np.linspace(np.log10(lognorm.ppf(1-left_alpha, s=std_base_e[0,1], scale= np.exp(mu_base_e[0,1]))), 
                         np.log10(lognorm.ppf(right_alpha, s=std_base_e[-1,1], scale= np.exp(mu_base_e[-1,1]))),npoints)
    return Etrue, ztrue, mu_base_e, std_base_e



def get_interpolators_dc(recompute_flux=False, recompute_aeff=False):
    interp_flux = interpolate_flux_DC(recompute_flux)
    interp_aeff = interpolate_aeff_dc(recompute_aeff)

    return interp_flux, interp_aeff

if __name__ == '__main__':
    models = get_true_models()
    pickle.dump(models,open('./pre_computed/DC_models.p','wb'))
