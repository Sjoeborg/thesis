import sys,os
if __name__ == '__main__':
    sys.path.append('./src/data')
    sys.path.append('./src/probability')
import numpy as np
import pandas as pd
import warnings
import pickle

def get_aeff_df_DC():
    return [flavor_aeff_df_dc(flavor) for flavor in ['E','EBar','Mu','Mubar', 'Tau', 'TauBar','X','XBar']]

def MC2018_DC():
    return pd.read_csv(f'./src/data/files/DC/2018/sample_b/neutrino_mc.csv', dtype=np.float64)

def no_osc2018_DC(pid):
    if pid == 1:
        df = pd.read_csv(f'./src/data/files/DC/2018/track_noosc.csv', dtype=np.float64, header=None) 
    else:
        df = pd.read_csv(f'./src/data/files/DC/2018/cascade_noosc.csv', dtype=np.float64, header=None)
    return df

def flavor_aeff_df_DC(flavor):

    filename = f'./src/data/files/DC/2015/NC_Nu{flavor}.txt' if flavor[0] == 'X' else f'./src/data/files/DC/2015/CC_Nu{flavor}.txt'
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

def systematics2015_DC():
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


def MC2015_DC():
    bins = np.arange(1,9)
    MC_factors =[]
    for bin in bins:
        df = pd.read_csv(f'./src/data/files/DC/2015/DC2015_MC_bin{bin}.csv', skiprows=2, names=['z1','rates','z2','events'], dtype=np.float64)
        MC = df.events/df.rates
        MC_factors.append(MC.to_numpy())
    return np.array(MC_factors)

def events2018_DC():
    df = pd.read_csv('./src/data/files/DC/2018/sample_b/data.csv')
    df1 = pd.read_csv('./src/data/files/DC/2018/sample_b/muons.csv')

    merged_df = pd.merge(df,df1, on =['pid', 'reco_coszen','reco_energy'], suffixes=('_events','_background'))
    return merged_df

def events2015_DC():
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
    df['Emin'] = 10**(E_ranges[0].astype(np.float64))
    df['Emax'] = 10**(E_ranges[1].astype(np.float64))
    df['reco_energy'] = (df['Emax'] + df['Emin'])/2
    df['reco_coszen'] = (df['zmax'] + df['zmin'])/2
    df = df[['reco_coszen','reco_energy','events', 'background']]
    #df = df.pivot(index='reco_coszen', columns='reco_energy')
    return df#['events'], df['background']

def systematics2018_DC():
    hyperplanes = {}
    hyperplanes['e'] = pd.read_csv("./src/data/files/DC/2018/sample_b/hyperplanes_nue_cc.csv")
    hyperplanes['mu'] = pd.read_csv("./src/data/files/DC/2018/sample_b/hyperplanes_numu_cc.csv")
    hyperplanes['tau'] = pd.read_csv("./src/data/files/DC/2018/sample_b/hyperplanes_nutau_cc.csv")
    hyperplanes['nc'] = pd.read_csv("./src/data/files/DC/2018/sample_b/hyperplanes_all_nc.csv")

    # bestfit point of detector systematics from Phys. Rev. D 99, 032007 (2019), Table 2
    bestfit = {}
    bestfit['ice_scattering'] = 0.974
    bestfit['ice_absorption'] = 1.021
    bestfit['opt_eff_overall'] = 1.05
    bestfit['opt_eff_lateral'] = -0.25
    bestfit['opt_eff_headon'] = -1.15

    # calculate the correction factors from detector systematics at bestfit point
    for hyperplane in hyperplanes.values():
        hyperplane['correction_factor'] = (
            hyperplane['offset'] +
            hyperplane['ice_absorption'] * (bestfit['ice_absorption'] - 1.) * 100. +
            hyperplane['ice_scattering'] * (bestfit['ice_scattering'] - 1.) * 100. +
            hyperplane['opt_eff_overall'] * (bestfit['opt_eff_overall']) +
            hyperplane['opt_eff_lateral'] * ((bestfit['opt_eff_lateral'] * 10) + 25) +
            hyperplane['opt_eff_headon'] * (bestfit['opt_eff_headon'])
            )
    return hyperplanes



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
    z_bins_left = np.linspace(-1,0.9,20)
    z_bins_right = np.linspace(-0.9,1,20)
    df = pd.DataFrame()
    for i,(left,right) in enumerate(zip(df1,df2)):
        new = (left + right)/2
        new['GeV'] = left['GeV']
        new['z_min'] = np.round(z_bins_left[i],2)
        new['z_max'] = np.round(z_bins_right[i],2)
        df = pd.concat([df,new])
    new_leftmost = df[df['z_min'] == -1].copy()
    new_rightmost = df[df['z_max'] == 1].copy()

    new_leftmost['z_min'] = -1.1
    new_leftmost['z_max'] = -1
    new_rightmost['z_min'] = 1
    new_rightmost['z_max'] = 1.1
    df = pd.concat([new_leftmost,df,new_rightmost])
    from IC.importer import extrapolate_flux
    df = extrapolate_flux(df)
    return df

if __name__ == '__main__':
    pass