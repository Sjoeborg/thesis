import sys,os
if __name__ == '__main__':
    sys.path.append('./src/probability')
    sys.path.append('./src/data')
import numpy as np 
import pandas as pd 
from PINGU.importer import MC_PINGU
from PINGU.processer import get_binned_PINGU, generate_probabilities_PINGU, get_probabilities_PINGU
from DC.processer import get_interpolators_DC, get_flux
from functions import dc_params_nsi,dc_params, nufit_params_nsi
df = MC_PINGU()
interp_flux,_ = get_interpolators_DC()

def get_events(Ebin,zbin,params,pid,nsi,save=True):
    binned_df = get_binned_PINGU(pid,Ebin,zbin,df)
    events = 0
    
    rate_weight = np.zeros_like(binned_df["weight"])

    masks = {}
    masks['e'] = (binned_df["pdg"] == 12)
    masks['m'] = (binned_df["pdg"] == 14)
    masks['t'] = (binned_df["pdg"] == 16)

    anti_masks = {}
    anti_masks['e'] = (binned_df["pdg"] == -12)
    anti_masks['m'] = (binned_df["pdg"] == -14)
    anti_masks['t'] = (binned_df["pdg"] == -16)

    for flavor_to, mask in masks.items():
        Etrue = binned_df[mask]['true_energy'].values
        ztrue = binned_df[mask]['true_coszen'].values
        e_flux = get_flux('e',Etrue,ztrue,interp_flux)
        m_flux = get_flux('m',Etrue,ztrue,interp_flux)

        try:
            Pe = get_probabilities_PINGU('e', flavor_to, Ebin,zbin,params,anti=False,pid=pid,ndim=3,nsi=nsi).reshape(-1,)
        except KeyError:
            Pe = generate_probabilities_PINGU('e', flavor_to, Etrue, ztrue, Ebin, zbin, params, anti=False, pid=pid, ndim=3, nsi=nsi, save=save).reshape(-1,)
        try:
            Pm = get_probabilities_PINGU('m', flavor_to, Ebin,zbin,params,anti=False,pid=pid,ndim=3,nsi=nsi).reshape(-1,)
        except KeyError:
            Pm = generate_probabilities_PINGU('m', flavor_to, Etrue, ztrue, Ebin, zbin, params, anti=False, pid=pid, ndim=3, nsi=nsi, save=save).reshape(-1,)

        rate_weight[mask] = binned_df[mask]['weight'] * (m_flux*Pm + e_flux*Pe)

    for flavor_to, mask in anti_masks.items():
        Etrue = binned_df[mask]['true_energy'].values
        ztrue = binned_df[mask]['true_coszen'].values
        ebar_flux = get_flux('ebar',Etrue,ztrue,interp_flux)
        mbar_flux = get_flux('mbar',Etrue,ztrue,interp_flux)

        try:
            Pebar = get_probabilities_PINGU('e', flavor_to, Ebin,zbin,params,anti=True,pid=pid,ndim=3,nsi=nsi).reshape(-1,)
        except KeyError:
            Pebar = generate_probabilities_PINGU('e', flavor_to, Etrue, ztrue, Ebin, zbin, params, anti=True, pid=pid, ndim=3, nsi=nsi, save=save).reshape(-1,)
        try:
            Pmbar = get_probabilities_PINGU('m', flavor_to, Ebin,zbin,params,anti=True,pid=pid,ndim=3,nsi=nsi).reshape(-1,)
        except KeyError:
            Pmbar = generate_probabilities_PINGU('m', flavor_to, Etrue, ztrue, Ebin, zbin, params, anti=True, pid=pid, ndim=3, nsi=nsi, save=save).reshape(-1,)
        
        rate_weight[mask] = binned_df[mask]['weight'] * (mbar_flux*Pmbar + ebar_flux*Pebar)
    binned_df["rate_weight"] = rate_weight
    return np.sum(binned_df["rate_weight"])


def get_all_events(params, nsi,save=True):
    cascades = np.array([get_events(Ebin,zbin,params,0,nsi,save) for Ebin in range(8) for zbin in range(8)]).reshape(8,8)
    tracks = np.array([get_events(Ebin,zbin,params,1,nsi,save) for Ebin in range(8) for zbin in range(8)]).reshape(8,8)
    return np.array([cascades, tracks])