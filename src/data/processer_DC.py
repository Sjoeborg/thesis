import sys,os
if __name__ == '__main__':
    sys.path.append('./src/probability')
    sys.path.append('./src/data')
import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator as CT
from scipy.interpolate import NearestNDInterpolator
from functions import mass_dict
from importer_DC import systematics2015_DC, get_aeff_df_DC, get_flux_df_DC
from dict_hash import sha256
import pandas as pd
from numerical import wrapper 
import h5py
from scipy.stats import lognorm
import pickle

pdg_dict={'e':12,'m':14,'t':16}
Ebins_2018 = [5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413]
zbins_2018 = [-1., -0.75, -0.5 , -0.25,  0., 0.25, 0.5, 0.75, 1.]

def MC2018_DC(track, cascade):
    #TODO division of CC/NC at end might be redundant.
    interp_flux = interpolate_flux_DC()
    
    df = pd.read_csv(f'./src/data/files/DC/2018/sample_b/neutrino_mc.csv', dtype=np.float64)

    e_mask = (df["pdg"] == 12)
    mu_mask = (df["pdg"] == 14)
    tau_mask = (df["pdg"] == 16)
    ebar_mask = (df["pdg"] == -12)
    mubar_mask = (df["pdg"] == -14)
    taubar_mask = (df["pdg"] == -16)

    track_mask = (df['pid'] == 1)
    cascade_mask = (df['pid'] == 0)

    if track and not cascade:
        pid_mask = track_mask
    elif not track and cascade:
        pid_mask = cascade_mask
    elif track and cascade:
        pid_mask = track_mask | cascade_mask
    else:
        raise ValueError('Specify track and/or cascade')

    e_mask = e_mask & pid_mask
    mu_mask = mu_mask & pid_mask
    tau_mask = tau_mask & pid_mask
    ebar_mask = ebar_mask & pid_mask
    mubar_mask = mubar_mask & pid_mask
    taubar_mask = taubar_mask & pid_mask
    

    rate_weight = np.zeros_like(df["weight"])

    mflux = get_flux('m',df[mu_mask].true_energy,df[mu_mask].true_coszen,interp_flux)
    eflux = get_flux('e',df[e_mask].true_energy,df[e_mask].true_coszen,interp_flux)
    mbarflux = get_flux('mbar',df[mubar_mask].true_energy,df[mubar_mask].true_coszen,interp_flux)
    ebarflux = get_flux('ebar',df[ebar_mask].true_energy,df[ebar_mask].true_coszen,interp_flux)

    rate_weight[e_mask] = eflux * df['weight'][e_mask]
    rate_weight[mu_mask] = mflux * df['weight'][mu_mask]
    rate_weight[ebar_mask] = ebarflux * df['weight'][ebar_mask]
    rate_weight[mubar_mask] = mbarflux * df['weight'][mubar_mask]

    
    df['rate_weight'] = rate_weight
    
    reco_df = df[['reco_coszen', 'reco_energy','true_coszen','true_energy', 'rate_weight','pid','pdg','type']].dropna()
    '''
    dc2018_mc = reco_df.groupby(['reco_coszen','reco_energy','pid','pdg','type']).sum().reset_index()
    
    neutrinos = {}
    neutrinos['nc'] = dc2018_mc[dc2018_mc['type'] == 0].drop('type',axis=1).groupby(['reco_coszen','reco_energy','pid','pdg']).sum().reset_index()
    neutrinos['e'] = dc2018_mc[(dc2018_mc['type'] > 0) & (abs(dc2018_mc['pdg']) == 12)].drop('type',axis=1).groupby(['reco_coszen','reco_energy','pid','pdg']).sum().reset_index()
    neutrinos['mu'] = dc2018_mc[(dc2018_mc['type'] > 0) & (abs(dc2018_mc['pdg']) == 14)].drop('type',axis=1).groupby(['reco_coszen','reco_energy','pid','pdg']).sum().reset_index()
    neutrinos['tau'] = dc2018_mc[(dc2018_mc['type'] > 0) & (abs(dc2018_mc['pdg']) == 16)].drop('type',axis=1).groupby(['reco_coszen','reco_energy','pid','pdg']).sum().reset_index()
    '''
    return reco_df#neutrinos


def get_flux(flavor,E,z,df):
    '''
    Returns flux for a set of flavor, energy [GeV], and z=cos(theta_z).
    '''
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

def get_probabilities_DC(flavor_from, flavor_to, Ebin, zbin, param_dict,anti,pid,ndim):
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

def generate_probabilities_DC(flavor_from, flavor_to, E_range,z_range,E_bin,z_bin,params,anti,pid, ndim=4, nsi=False):
    prob = np.array([wrapper([flavor_from, [E_range[i]],z, anti, params, ndim, nsi])[mass_dict[flavor_to]] for i,z in enumerate(z_range)])
    hashed_param_name = sha256(params)
    if anti:
        flavor_from = 'a' + flavor_from
        flavor_to = 'a' + flavor_to
    f = h5py.File(f'./pre_computed/DC/E{E_bin}z{z_bin}.hdf5', 'a')
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
        with open(f'./pre_computed/DC/hashed_params.csv','a') as fd:
            fd.write(f'{params};{hashed_param_name}\n')
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

def get_true_models():
    Ereco = np.array([5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413])
    zreco = np.array([-1., -0.75, -0.5 , -0.25,  0.])
    filename = './src/data/files/DC/sample_b/neutrino_mc.csv'
    df = (pd.read_csv(filename)
        .query('pdg == 14 or pdg == -14') #only muon (anti)neutrinos
        .query('pid == 1 ')) # only tracks
    df['Ebin'] = pd.cut(df.reco_energy, bins=Ereco, labels=False)
    df['zbin'] = pd.cut(df.reco_coszen, bins=zreco, labels=False)
    df = df.sample(100000)
    print(len(df))
    def train(df):
        X = np.array([df.reco_coszen, np.log(df.reco_energy)]).reshape(-1, 2)
        y = np.array([df.true_coszen, np.log(df.true_energy)]).reshape(-1, 2)
        kernel2  = 1.0 * RBF() + WhiteKernel(noise_level=3)
        gpr = GaussianProcessRegressor(kernel=kernel2,random_state=0).fit(X, y)
        return gpr

    return train(df)


def get_true_DC(flavor,anti,pid,E_bin,z_bin,df):
    pdg = pdg_dict[flavor]
    if anti:
        pdg = -pdg
    df1 = (df.query(f'pid=={pid}')
             .query(f'pdg=={pdg}')
             .query(f'reco_energy<{Ebins_2018[E_bin+1]}')
             .query(f'reco_energy>{Ebins_2018[E_bin]}')
             .query(f'reco_coszen<{zbins_2018[z_bin+1]}')
             .query(f'reco_coszen>{zbins_2018[z_bin]}'))

    return df1




def get_interpolators_DC(recompute_flux=False, recompute_aeff=False):
    interp_flux = interpolate_flux_DC(recompute_flux)
    interp_aeff = interpolate_aeff_DC(recompute_aeff)

    return interp_flux, interp_aeff

if __name__ == '__main__':
    pass
