import sys,os
if __name__ == '__main__':
    os.chdir('../../')
import numpy as np
import pandas as pd
import warnings
import pickle

def get_aeff_df_dc():
    flavor_list=[]
    df_list=[]
    for flavor in ['E','EBar','Mu','Mubar', 'Tau', 'TauBar','X','XBar']:
        df_list.append(flavor_aeff_df_dc(flavor))
    return df_list



def flavor_aeff_df_dc(flavor):
    filename = f'./src/data/files/DC/2015/CC_Nu{flavor}.txt'
    if flavor[0] == 'X':
        filename = f'./src/data/files/DC/2015/NC_Nu{flavor}.txt'
    df_chunks = pd.read_csv(filename, skiprows=1, delimiter='\t', names=['z', 'E', f'aeff_{flavor}'],skip_blank_lines=False)
    df_list = np.split(df_chunks, df_chunks[df_chunks.isnull().all(1)].index)
    _=df_list.pop(-1)
    new_list=[]
    for df in df_list:  
        df = (df.dropna()
                .reset_index(drop=True)
                .drop(0)
                .convert_dtypes())
        df.z = df.z.str.replace('[','')
        df.E = df.E.str.replace('[','')
        df.z = df.z.str.replace(']','')
        df.E = df.E.str.replace(']','')

        z_ranges = pd.DataFrame(df.z.str.split(',', expand=True))
        E_ranges = pd.DataFrame(df.E.str.split(',', expand=True))
        df['zmin'] = z_ranges[0].astype(np.float64)
        df['zmax'] = z_ranges[1].astype(np.float64)
        df['Emin'] = E_ranges[0].astype(np.float64)
        df['Emax'] = E_ranges[1].astype(np.float64)
        df[f'aeff_{flavor}'] = df[f'aeff_{flavor}'].astype(np.float64)

        df['E_avg'] = (df['Emax'] + df['Emin'])/2
        df['z_avg'] = (df['zmax'] + df['zmin'])/2
        new_list.append(df)
    return new_list

def get_systematics():
    best_fit_optical_eff = 1.015
    best_fit_hole = 0.02
    eval_DOM = lambda p:eval(p.replace('^','**').replace(' x',f'*{best_fit_optical_eff}'))
    eval_ICE = lambda p:eval(p.replace('^','**').replace(' x',f'*{best_fit_hole}'))
    df = pd.read_csv('./src/data/files/DC/2015/DOMeff.txt', skiprows=3, delimiter='\t', names=['E', 'z', 'del','DOMeff','del1'])
    df_ICE = pd.read_csv('./src/data/files/DC/2015/HoleIce.txt', skiprows=3, delimiter='\t', names=['E', 'z', 'del','ICEeff','del1'])
    df_ICE = df_ICE['ICEeff']
    df.z = df.z.str.replace('cosZreco=\[','')
    df.E = df.E.str.replace('logEreco=\[','')
    df.z = df.z.str.replace('=\[','')
    df.z = df.z.str.replace(']','')
    df.E = df.E.str.replace(']','')
    df = df.drop(['del','del1'],axis=1)
    z_ranges = pd.DataFrame(df.z.str.split(',', expand=True))
    E_ranges = pd.DataFrame(df.E.str.split(',', expand=True))
    df['zmin'] = z_ranges[0].astype(np.float64)
    df['zmax'] = z_ranges[1].astype(np.float64)
    df['Emin'] = E_ranges[0].astype(np.float64)
    df['Emax'] = E_ranges[1].astype(np.float64)
    df['E_avg'] = (df['Emax'] + df['Emin'])/2
    df['z_avg'] = (df['zmax'] + df['zmin'])/2

    df.DOMeff = df.DOMeff.apply(eval_DOM)
    df['ICEeff']= df_ICE.apply(eval_ICE)
    return df[['E_avg','z_avg','DOMeff','ICEeff']]

def DC2018_MC():
    from processerDC import get_flux, interpolate_flux_DC
    interp_flux = interpolate_flux_DC()
    df = pd.read_csv(f'./src/data/files/DC/2018/sample_b/neutrino_mc.csv', dtype=np.float64)

    nue_mask = (df["pdg"] == 12)
    numu_mask = (df["pdg"] == 14)
    nutau_mask = (df["pdg"] == 16)
    nuebar_mask = (df["pdg"] == -12)
    numubar_mask = (df["pdg"] == -14)
    nutaubar_mask = (df["pdg"] == -16)

    nc_mask = (df["type"] == 0)
    cc_mask = (df["type"] == 1)
    track_mask = (df['pid'] == 1)
    cascade_mask = (df['pid'] == 0)

    nue_cc_mask = nue_mask & cc_mask
    numu_cc_mask = numu_mask & cc_mask
    nutau_cc_mask = nutau_mask & cc_mask

    nuebar_cc_mask = nuebar_mask & cc_mask
    numubar_cc_mask = numubar_mask & cc_mask
    nutaubar_cc_mask = nutaubar_mask & cc_mask

    rate_weight = np.zeros_like(df["weight"])

    mflux = get_flux('m',df[numu_mask].true_energy,df[numu_mask].true_coszen,interp_flux)
    eflux = get_flux('e',df[nue_mask].true_energy,df[nue_mask].true_coszen,interp_flux)
    mbarflux = get_flux('mbar',df[numubar_mask].true_energy,df[numubar_mask].true_coszen,interp_flux)
    ebarflux = get_flux('ebar',df[nuebar_mask].true_energy,df[nuebar_mask].true_coszen,interp_flux)

    rate_weight[nue_mask] = eflux * df['weight'][nue_mask]
    rate_weight[numu_mask] = mflux * df['weight'][numu_mask]
    rate_weight[nuebar_mask] = ebarflux * df['weight'][nuebar_mask]
    rate_weight[numubar_mask] = mbarflux * df['weight'][numubar_mask]

    df['rate_weight'] = rate_weight
    livetime = 1022*24*3600
    reco_df = df[['reco_coszen', 'reco_energy', 'rate_weight']]
    dc2018_mc = reco_df.groupby(['reco_coszen','reco_energy']).sum().reset_index().pivot(index='reco_coszen',columns='reco_energy')*livetime
    return dc2018_mc.query('reco_coszen < 0')

def DC2015_MC():
    bins = np.arange(1,9)
    MC_factors =[]
    for bin in bins:
        df = pd.read_csv(f'./src/data/files/DC/2015/DC2015_MC_bin{bin}.csv', skiprows=2, names=['z1','rates','z2','events'], dtype=np.float64)
        MC = df.events/df.rates
        MC_factors.append(MC.to_numpy())
    return np.array(MC_factors)

def DC2018_event_data():
    df = pd.read_csv('./src/data/files/DC/2018/sample_b/data.csv').query('pid==1')
    df = df.pivot(index='reco_coszen', columns='reco_energy')
    return df['count'].query('reco_coszen < 0')

def DC2015_event_data():
    df = pd.read_csv('./src/data/files/DC/2015/DataCounts.txt', skiprows=2, delimiter='\t', names=['z','E','events'])
    background = pd.read_csv('./src/data/files/DC/2015/AtmMuons_fromData.txt', skiprows=2, delimiter='\t', names=['z','E','events'])
    df['background'] = background.events
    df.z = df.z.str.replace('[','')
    df.E = df.E.str.replace('[','')
    df.z = df.z.str.replace(']','')
    df.E = df.E.str.replace(']','')
    z_ranges = pd.DataFrame(df.z.str.split(',', expand=True))
    E_ranges = pd.DataFrame(df.E.str.split(',', expand=True))
    df['zmin'] = z_ranges[0].astype(np.float64)
    df['zmax'] = z_ranges[1].astype(np.float64)
    df['Emin'] = E_ranges[0].astype(np.float64)
    df['Emax'] = E_ranges[1].astype(np.float64)
    df['Eavg'] = (df['Emax'] + df['Emin'])/2
    df['zavg'] = (df['zmax'] + df['zmin'])/2
    df = df[['zavg','Eavg','events', 'background']]
    df = df.pivot(index='zavg', columns='Eavg')
    return df

def get_flux_df_DC():
    '''
    Reads the files files/spl-nu-20-01-000.d and files/spl-nu-20-01-n3650.d which contain the solar min and max atm fluxes. Averages these for each zenith angle range

    Files are from http://www.icrr.u-tokyo.ac.jp/~mhonda/nflx2014/index.html section 2.6
    '''
    file1 = './src/data/files/spl-nu-20-01-000.d'
    file2 = './src/data/files/spl-nu-20-01-n3650.d'
    colnames = ['GeV', 'm_flux', 'mbar_flux', 'e_flux', 'ebar_flux']

    text_rows = np.append(np.arange(0,2500,103),(np.arange(1,2500,103)))

    df1 = pd.read_csv(file1, skiprows=text_rows, header=None, names=colnames, dtype = np.float64, sep = ' ', chunksize=101)
    df2 = pd.read_csv(file2, skiprows=text_rows, header=None, names=colnames, dtype = np.float64, sep = ' ', chunksize=101)

    df_list = [] #List of the dataframes for each zenith angle bin. df_list[i] is the df for angle theta_range[i]
    for left,right in zip(df1,df2):
        left = left.set_index('GeV')
        right = right.set_index('GeV')
        df_concat = pd.concat([left, right])
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = by_row_index.mean()
        df_means.reset_index(inplace=True)
        df_list.append(df_means)
    df_list.append(df_list[-1])
    binned_df = z_bins_DC(df_list)
    df = pd.concat(binned_df)
    return df

def z_bins_DC(df_list):
    '''
    Doubles the number of dataframes, and puts half of the initial flux in each one.
    Also puts the new z-bin limits in columns
    '''
    new_theta_range =np.round(np.linspace(1,-1.1,22),2) #See comment in get_flux_df
    new_df_list = []
    for i,df in enumerate(df_list):
        
        df['z_min'] = new_theta_range[i+1]
        df['z_max'] = new_theta_range[i]

        new_df_list.append(df)
    return new_df_list

if __name__ == '__main__':
    get_flux_factor()