import sys,os
if __name__ == '__main__':
    sys.path.append('./src/probability')
    sys.path.append('./src/data')
import numpy as np 
import pandas as pd 
from DC.importer import MC2018_DC, systematics2018_DC
from DC.processer import get_binned_DC, generate_probabilities_DC, get_probabilities_DC, get_flux, get_interpolators_DC,process_systematics
from functions import dc_params_nsi,dc_params
df = MC2018_DC()
interp_flux,_ = get_interpolators_DC()
systematics = systematics2018_DC()

def get_events(Ebin,zbin,params,pid,nsi, no_osc=False):
    binned_df = get_binned_DC(pid,Ebin,zbin,df)
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
            Pe = get_probabilities_DC('e', flavor_to, Ebin,zbin,params,anti=False,pid=pid,ndim=3,nsi=nsi).reshape(-1,)
        except KeyError:
            Pe = generate_probabilities_DC('e', flavor_to, Etrue, ztrue, Ebin, zbin, params, anti=False, pid=pid, ndim=3, nsi=nsi).reshape(-1,)
        try:
            Pm = get_probabilities_DC('m', flavor_to, Ebin,zbin,params,anti=False,pid=pid,ndim=3,nsi=nsi).reshape(-1,)
        except KeyError:
            Pm = generate_probabilities_DC('m', flavor_to, Etrue, ztrue, Ebin, zbin, params, anti=False, pid=pid, ndim=3, nsi=nsi).reshape(-1,)
        if not no_osc:
            rate_weight[mask] = binned_df[mask]['weight'] * (m_flux*Pm + e_flux*Pe)
        else:
            if flavor_to == 'e':
                rate_weight[mask] = binned_df[mask]['weight'] * e_flux
            elif flavor_to == 'm':
                rate_weight[mask] = binned_df[mask]['weight'] * m_flux

    for flavor_to, mask in anti_masks.items():
        Etrue = binned_df[mask]['true_energy'].values
        ztrue = binned_df[mask]['true_coszen'].values
        ebar_flux = get_flux('ebar',Etrue,ztrue,interp_flux)
        mbar_flux = get_flux('mbar',Etrue,ztrue,interp_flux)

        try:
            Pebar = get_probabilities_DC('e', flavor_to, Ebin,zbin,params,anti=True,pid=pid,ndim=3,nsi=nsi).reshape(-1,)
        except KeyError:
            Pebar = generate_probabilities_DC('e', flavor_to, Etrue, ztrue, Ebin, zbin, params, anti=True, pid=pid, ndim=3, nsi=nsi).reshape(-1,)
        try:
            Pmbar = get_probabilities_DC('m', flavor_to, Ebin,zbin,params,anti=True,pid=pid,ndim=3,nsi=nsi).reshape(-1,)
        except KeyError:
            Pmbar = generate_probabilities_DC('m', flavor_to, Etrue, ztrue, Ebin, zbin, params, anti=True, pid=pid, ndim=3, nsi=nsi).reshape(-1,)
        
        if not no_osc:
            rate_weight[mask] = binned_df[mask]['weight'] * (mbar_flux*Pmbar + ebar_flux*Pebar)
        else:
            if flavor_to == 'e':
                rate_weight[mask] = binned_df[mask]['weight'] * ebar_flux
            elif flavor_to == 'm':
                rate_weight[mask] = binned_df[mask]['weight'] * mbar_flux
    
    binned_df["rate_weight"] = rate_weight
    res = process_systematics(binned_df, systematics,pid)
    return np.sum(res) #np.sum(binned_df["rate_weight"])


def get_all_events(params, pid, nsi, no_osc=False):
    result = [get_events(Ebin,zbin,params,pid,nsi,no_osc) for Ebin in range(8) for zbin in range(8)]
    return np.array(result).reshape(8,8)