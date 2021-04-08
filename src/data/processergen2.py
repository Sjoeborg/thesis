import sys,os
if __name__ == '__main__':
    #os.chdir('../../')
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
from numerical import wrapper 
from multiprocessing import Pool
#from numerical import wrapper 
import h5py
from scipy.stats import lognorm
import pickle
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
pdg_dict={'e':12,'m':14,'t':16}
Ebins_2018 = [5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413]
zbins_2018 = [-1., -0.75, -0.5 , -0.25,  0., 0.25, 0.5, 0.75, 1.]


def get_probabilitiesgen2(flavor_from, flavor_to, Ebin, zbin, param_dict,anti,pid,ndim):
    hashed_param_name = sha256(param_dict)
    if anti:
        flavor_from = 'a' + flavor_from
        flavor_to = 'a' + flavor_to
    try:
        f = h5py.File(f'./pre_computed/gen2/E{Ebin}z{zbin}.hdf5', 'r')
    except OSError:
        raise KeyError(f'E{Ebin}z{zbin}.hdf5 doesnt exist in ./pre_computed/gen2/')
    try:
        fh = f[f'{ndim}gen/P{flavor_from}{flavor_to}/{N}/{hashed_param_name}']
    except KeyError:
        f.close()
        raise KeyError(f'{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name} doesnt exist in E{Ebin}z{zbin}.hdf5')
    res = fh[()]
    f.close()
    return res

def generate_probabilitiesgen2(flavor_from, flavor_to, E_range,z_range,E_bin,z_bin,params,anti,pid, ndim=4, nsi=False):
    prob = np.array([wrapper([flavor_from, E_range,z, anti, params, ndim, nsi])[mass_dict[flavor_to]] for z in z_range])
    hashed_param_name = sha256(params)
    if anti:
        flavor_from = 'a' + flavor_from
        flavor_to = 'a' + flavor_to
    f = h5py.File(f'./pre_computed/gen2/E{E_bin}z{z_bin}.hdf5', 'a')
    try:
        dset = f.create_dataset(f'{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name}', data=prob, chunks=True)
        for key in params.keys():
            dset.attrs[key] = params[key]
        f.close()
    except RuntimeError:
        print(f'{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name} already exists, skipping')
        f.close()
        return prob
    if E_bin == 5 and z_bin == 5 and flavor_from == 'am' and flavor_to == 'am':
        with open(f'./pre_computed/gen2/hashed_params.csv','a') as fd:
            fd.write(f'{params};{hashed_param_name}\n')
    return prob




def get_true_gen2(flavor,anti,pid,E_bin,z_bin,df):
    pdg = pdg_dict[flavor]
    if anti:
        pdg = -pdg
    df1 = (df.query(f'pid=={pid}')
             .query(f'pdg=={pdg}')
             .query(f'reco_energy<{Ebins_2018[E_bin+1]}')
             .query(f'reco_energy>{Ebins_2018[E_bin]}')
             .query(f'reco_coszen<{zbins_2018[z_bin+1]}')
             .query(f'reco_coszen>{zbins_2018[z_bin]}'))

    return df1.true_energy.values, df1.true_coszen.values



if __name__ == '__main__':
    pass
