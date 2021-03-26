import sys,os
if __name__ == '__main__':
    os.chdir('../../')
import numpy as np
import pandas as pd
import warnings
import pickle


def extrapolate_aeff_edges(df):
    E_averages = df.E_avg.unique()
    for E in E_averages:
        aeff_at_edge = df[df.E_avg == E].query('z_min <= -0.99').reset_index()
        extrapolated_aeff = 2*aeff_at_edge.loc[1,'Aeff']-aeff_at_edge.loc[2,'Aeff']
        constant_aeff = aeff_at_edge.loc[1,'Aeff']
        aeff_at_edge.loc[0,'Aeff'] = extrapolated_aeff
        df.iloc[aeff_at_edge['index'][0]] = aeff_at_edge.iloc[0]

    return df.sort_values(by=['E_avg','z_avg'])

def get_aeff_df_dc():
    df_chunks = pd.read_csv('../src/data/files/DC/CC_NuE.txt', skiprows=1, delimiter='\t', names=['z', 'E', 'aeff'],skip_blank_lines=False)
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

        df['E_avg'] = (df['Emax'] + df['Emin'])/2
        df['z_avg'] = (df['zmax'] + df['zmin'])/2
        new_list.append(df)
    return new_list



def flux_parametrization(x, K, b, c, a):
    '''
    Source: https://arxiv.org/abs/hep-ph/0203272v2
    '''
    return K * np.power(x + b * np.exp(-c * np.sqrt(x)),-a)
    #return K * np.power(x,-a)





def DC_event_data():
    '''
    Data from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.102.052009 fig2
    Double-checked and confirmed to sum to 305735
    '''
    events = np.array([[32, 85,  143, 186, 208, 179, 119,  67],  
                       [66, 101, 162, 223, 271, 276, 230, 147],
                       [119, 150, 216, 288, 355, 366, 310, 211], 
                       [179, 193, 266, 356, 431, 440, 401, 295], 
                       [248, 240, 291, 353, 440, 468, 445, 332], 
                       [252, 236, 260, 279, 338, 343, 347, 262], 
                       [218, 192, 211, 206, 220, 215, 218, 186], 
                       [182, 197, 206, 206, 171, 137, 128, 113]])
    z_range = np.linspace(-1,-0.25,8)
    E_buckets = np.logspace(0.75,1.5,8)

    df = pd.DataFrame(events, columns= E_buckets, index=z_range)
    return df

def get_flux_df_DC():
    '''
    Reads the files files/spl-nu-20-01-000.d and files/spl-nu-20-01-n3650.d which contain the solar min and max atm fluxes. Averages these for each zenith angle range and returns the fluxes for zenith between -1.05 to 0.05, extrapolated to 1e5 GeV.

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


def extrapolate_flux(flux_df):
    '''
    Extrapolates the fluxes to 1e5 GeV
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") #Ignore warning about covariance not estimated
        for zmin in flux_df.z_min.unique():
            res = fit_flux(flux_df,zmin)
            flux_df = flux_df.append(res)
    return flux_df

def fit_flux(flux_df,zmin):
    from sklearn.linear_model import LinearRegression as LR
    '''
    Fits all four fluxes for a given z_min 
    '''
    df = flux_df[flux_df.z_min == zmin]

    x = df.query('GeV > 5e3').GeV
    y_m = df.query('GeV > 5e3').m_flux
    y_mbar = df.query('GeV > 5e3').mbar_flux
    y_e = df.query('GeV > 5e3').e_flux
    y_ebar = df.query('GeV > 5e3').ebar_flux


    model_m = LR().fit(np.array(np.log10(x)).reshape(-1, 1), np.log10(y_m))
    model_mbar = LR().fit(np.array(np.log10(x)).reshape(-1, 1), np.log10(y_mbar))
    model_e = LR().fit(np.array(np.log10(x)).reshape(-1, 1), np.log10(y_e))
    model_ebar = LR().fit(np.array(np.log10(x)).reshape(-1, 1), np.log10(y_ebar))
    x_new = np.logspace(np.log10(1e4),np.log10(1e6),100)
    y_new_m = model_m.predict(np.log10(x_new.reshape(-1,1)))
    y_new_mbar = model_mbar.predict(np.log10(x_new.reshape(-1,1)))
    y_new_e = model_e.predict(np.log10(x_new.reshape(-1,1)))
    y_new_ebar = model_ebar.predict(np.log10(x_new.reshape(-1,1)))

    return pd.DataFrame(np.transpose([x_new, 10**y_new_m, 10**y_new_mbar, 10**y_new_e, 10**y_new_ebar, zmin*np.ones(100), (zmin + 0.1)*np.ones(100)]),columns=['GeV', 'm_flux','mbar_flux','e_flux', 'ebar_flux', 'z_min','z_max'])
if __name__ == '__main__':
    get_flux_factor()