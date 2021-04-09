import sys,os
if __name__ == '__main__':
    os.chdir('../../')
import numpy as np
import pandas as pd
import warnings
import pickle
from DC.importer import get_flux, interpolate_flux_DC #DC flux can be used


def PINGU_MC(track, cascade):
    interp_flux = interpolate_flux_DC()
    df = pd.read_csv(f'./src/data/files/gen2/neutrino_mc.csv', dtype=np.float64)
    df['reco_coszen'] = np.cos(df['reco_zenith'])
    df['true_coszen'] = np.cos(df['true_zenith'])
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
    
    reco_df = df[['reco_coszen', 'reco_energy','true_coszen','true_energy', 'rate_weight','pid','pdg']]
    #grouped_mc = reco_df.groupby(['reco_coszen', 'reco_energy','true_coszen','true_energy', 'rate_weight','pid','pdg']).sum().reset_index()
    
    return reco_df#grouped_mc


if __name__ == '__main__':
    pass