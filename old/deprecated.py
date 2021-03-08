from scipy.interpolate import CloughTocher2DInterpolator as CT
import numpy as np
import pandas as pd

def extrapolate_aeff_edges(df):
    E_averages = df.E_avg.unique()
    for E in E_averages:
        aeff_at_edge = df[df.E_avg == E].query('z_min <= -0.99').reset_index()
        extrapolated_aeff = 2*aeff_at_edge.loc[1,'Aeff']-aeff_at_edge.loc[2,'Aeff']
        constant_aeff = aeff_at_edge.loc[1,'Aeff']
        aeff_at_edge.loc[0,'Aeff'] = extrapolated_aeff
        df.iloc[aeff_at_edge['index'][0]] = aeff_at_edge.iloc[0]

    return df.sort_values(by=['E_avg','z_avg'])

    
def get_Aeff_df_2012():
    '''
    Data from 2012
    https://icecube.wisc.edu/science/data/PS-3years
    Returns a df with cols 'E_min', 'E_max', 'z_min', 'z_max', 'Aeff', 'E_avg', 'z_avg'

    Cuts off energies above 20 000 GeV and z above 0 

    '''
    file1 = 'data/IC86-2012-TabulatedAeff.txt'
    colnames = ['E_min', 'E_max', 'z_min', 'z_max', 'Aeff']

    A = pd.read_csv(file1, header=None, skiprows=1,names=colnames, dtype = np.float64, skipinitialspace=True, sep=' ')

    #Constant extrapolation of aeff for z edge
    import warnings
    df_edges = A.query('z_min == -1.00')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_edges['z_max'] = np.round(df_edges['z_max'],2).replace(-0.99,-1.0)
    df_fixed = pd.concat([df_edges, A])

    df_fixed['E_avg'] = (df_fixed.E_min + df_fixed.E_max)/2
    df_fixed['z_avg'] = (df_fixed.z_min + df_fixed.z_max)/2
    df_fixed = df_fixed.query('E_max <= 1e6') #Remove E_max over 1e6 GeV
    df_fixed = df_fixed.query('E_min <= 1e6') #Remove E_min over 1e6 GeV
    df_fixed = df_fixed.query('z_max <= 0.1') #Remove z_max above 0.1

    df = df_fixed.reset_index(drop=True)
    extrapolated_df = extrapolate_aeff_edges(df)
    return extrapolated_df

def get_energy_resolutionOLD(Er, GPR_model):
    mu_base_e, std_base_e = GPR_model.predict(np.log(Er).reshape(-1,1), return_std=True)
    
    mu = np.exp(mu_base_e+ std_base_e**2/2)
    sigma = np.sqrt(mu**2*(np.exp(std_base_e**2)-1))
    return mu, sigma
def train_energy_resolutionOLD():
    from joblib import dump, load
    #TODO: Finetune this
    try:
        gpr = load('energy_resolution_model.joblib')
    except:
        filename = '~/NuFSGenMC_nominal.dat'
        df = pd.read_csv(filename, delimiter=' ', names= ['pdg', 'Ereco', 'zreco', 'Etrue', 'ztrue', 'mcweight', 'flux_pion', 'flux_kaon'], skiprows=12)
        df = df[['Ereco','Etrue']]
        df.Ereco = np.round(df.Ereco,0)
        df = df.groupby('Ereco').median().reset_index()

        df['Ebin'] = pd.cut(df.Ereco, bins=500*10**np.linspace(0.0,1.3,14))
        df_grouped = df.groupby('Ebin').median().dropna()

        if len(df_grouped) > 1e5:
            df_grouped = df_grouped.sample(1e5, random_state=0)
        X = np.log(np.array(df_grouped.Ereco).reshape(-1,1))
        y = np.log(df_grouped.Etrue)
        kernel1 = 1.0 * RBF() + WhiteKernel(noise_level=3)
        gpr = GaussianProcessRegressor(kernel=kernel1,random_state=0).fit(X, y)
        dump(gpr, 'energy_resolution_model.joblib') 
    return gpr
def interpolate_true_to_reco(df):
    '''
    Returns a interpolator using interp1d NN with extrapolation
    '''
    inter = interpolate.interp1d(df['Etrue'], df['Ereco'],fill_value='extrapolate',kind='linear')
    return inter

def interpolate_reco_to_true(df):
    '''
    Returns a interpolator using interp1d NN with extrapolation
    '''
    inter = interpolate.interp1d(df['Ereco'], df['Etrue'],fill_value='extrapolate',kind='linear')
    return inter

def interpolate_Aeff_2012(df):
    E = df.E_avg
    z_avg = df.z_avg
    aeff = np.array(df.Aeff)


    points_avg = np.array([E,z_avg]).T
    f_avg = CT(points_avg, aeff,rescale=True)
    return f_avg


def get_E_prob(deposited, reconstructed):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    datafile  = 'reconstruction.txt'

    data = np.loadtxt(datafile, dtype=float, delimiter=",")
    data = data.transpose()
    rtwolog = np.sqrt(2*np.log(2))
    rtwo = 1./np.sqrt(2*np.pi)

    get_sigma = interp1d(data[0], data[1])


    def get_odds_energy(deposited, reconstructed):

        sigma = get_sigma(deposited)*deposited*0.01
        
        s2 = np.log(1+ (sigma/deposited)**2)
        mu = np.log((deposited**2)/np.sqrt(deposited**2  + sigma**2))
        # now, we assume that the uncertainty follows a log normal distribution, and calculate the PDF here

        #prob = rtwo*(1./sigma)*exp(-0.5*((reconstructed - deposited)/sigma)**2)
        prob = rtwo*(1./np.sqrt(s2))*np.exp(-0.5*((np.log(reconstructed) - mu)**2)/s2)/reconstructed

        return(prob)
    return get_odds_energy(deposited, reconstructed)
    
def get_energy_resolution_df_new():
    '''
    Returns a tuple num,nue of the extractd energy resolutions in http://arxiv.org/abs/1311.4767.
    Energy resolutions are a standard deviation, expressed as a percentage of the Etrue at that point.
    '''
    file1 = 'data/energy_resolution.csv'

    num = pd.read_csv(file1, skiprows=0, header=None, names=['Etrue', 'Ereco'], dtype = np.float64, skipinitialspace=True, sep=',')
    return num

def generate_probabilitiesOLD(flavor_from, flavor_to, E_range,z_range,params,anti, ndim=4):
    '''
    Name of resulting .csv is the sha256 hash of the parameter dictionary used to generate the probablities.
    '''

    filename = sha256(params) #Get hash of parm dict to be used as filename
    if anti:
        file_dir = f'./simulations/{ndim}gen/Pa{flavor_from}a{flavor_to}/'
    else:
        #file_dir = f'./simulations/{ndim}gen/P{flavor_from}{flavor_to}/'
        file_dir = f'./simulations/test/P{flavor_from}{flavor_to}/'
    try: # Try to open pre-computed df in order to append the new P's
        df = pd.read_csv(file_dir+f'{filename}.csv', index_col=0, header=0, dtype=np.float64)
        df.index = df.index.astype(np.float64)
        df.columns = df.columns.astype(np.float64)
        #df.reset_index().groupby('z').max()
        #df.to_csv(file_dir+'/backup/'+f'{filename}.csv', index=True)
    except FileNotFoundError: # Else, just create it from scrath
        with open(file_dir+'hashed_params.csv','a') as fd:
            fd.write(f'{params};{filename}\n')
        df = pd.DataFrame(columns=E_range, index=z_range, dtype=np.float64)
        df.columns = df.columns.astype(np.float64)
        df.index = df.index.astype(np.float64)
        df.index.name = 'z'
    for z in z_range: # Compute probs and insert
        res = wrapper([flavor_from, E_range,z, anti, params, ndim])[mass_dict[flavor_to]]
        #print(filename, z, rounded_E_range)
        assert type(np.sum(res)) == np.float64, f'Got {res} as probs for {flavor_from}{flavor_to}, {E_range},{z}'
        df.loc[z,E_range] = res
        print('Successfully appended', res, z, E_range)
    #Make sure indices are rounded and unique
    #df1 = df.reset_index()
    #df1.z = np.round(df1.z.astype(np.float64),6).astype('str')
    #df2 = df1.groupby('z').fillna(0).max()
    #df2.to_csv(file_dir+f'{filename}.csv', index=True)
    df.to_csv(file_dir+f'{filename}.csv', index=True)



def get_probabilitiesOLD(flavor_from, flavor_to, E_range,z_range,params,anti, ndim=4):
    '''
    Returns the precomputed probabilities stored as .csv files in the ./simulations/Pab/ directory as an array of shape (E,z).

    E and z are rounded to remove floating point errors in column/index names. This shouldn't affect the returned probabilities unless very fine step sizes are used. Even then, the differences between the returned pre-computed probabilities and the true ones are negligible.

    Some assertions and except clauses to make sure that the df found has the correct shape and contains no nans.
    '''
    filename = sha256(params)
    if anti:
        flavor_from = 'a' + flavor_from
        flavor_to = 'a' + flavor_to
    #df = pd.read_csv(f'./simulations/{ndim}gen/P{flavor_from}{flavor_to}/{filename}'+'.csv', index_col=0)
    df = pd.read_csv(f'./simulations/test/P{flavor_from}{flavor_to}/{filename}'+'.csv', index_col=0, header=0, dtype=np.float64)
    df.index = df.index.astype(np.float64)
    df.columns = df.columns.astype(np.float64)
    try:
        subsetted_df = df.loc[df.index.isin(z_range),E_range]
        if subsetted_df.isnull().values.any(): #For debugging
            print(z_range, E_range, flavor_from, flavor_to, filename)
            print(subsetted_df)
        assert not subsetted_df.isnull().values.any()
    except (KeyError,AssertionError):
        raise AssertionError('At least one of the energies in E_range are not precomputed')
    assert np.shape(subsetted_df)[1] ==  len(E_range), f'{len(E_range)-np.shape(subsetted_df)[1]} of the energies in E_range are not precomputed'
    assert np.shape(subsetted_df)[0] ==  len(z_range), f'{len(z_range)-np.shape(subsetted_df)[0]} of the zeniths in z_range are not precomputed'
    return np.array(subsetted_df)

def extrapolate_fluxOLD(flux_df):
    '''
    Extrapolates the fluxes to 1e5 GeV
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") #Ignore warning about covariance not estimated
        for zmin in flux_df.z_min.unique():
            res = fit_flux(flux_df,zmin)
            flux_df = flux_df.append(res)
    return flux_df

def double_z_bins(df_list):
    '''
    Doubles the number of dataframes, and puts half of the initial flux in each one.
    Also puts the new z-bin limits in columns
    '''
    new_theta_range =np.round(np.linspace(0.1,-1.1,25),2) #See comment in get_flux_df
    new_df_list = []
    for i,df in enumerate(df_list):
        df_left = df/2 #Average flux
        df_right = df/2 #Average flux
        df_left['GeV'] =df['GeV'] # But we dont want to average the energy
        df_right['GeV'] =df['GeV'] # But we dont want to average the energy

        
        df_left['z_min'] = new_theta_range[2*i+1]
        df_left['z_max'] = new_theta_range[2*i]

        df_right['z_min'] = new_theta_range[2*i+2]
        df_right['z_max'] = new_theta_range[2*i+1]

        new_df_list.append(df_left)
        new_df_list.append(df_right)
    return new_df_list
def get_flux_dfOLD():
    '''
    Reads the files data/spl-nu-20-01-000.d and data/spl-nu-20-01-n3650.d which contain the solar min and max atm fluxes. Averages these for each zenith angle range and returns the fluxes for zenith between -1.05 to 0.05, extrapolated to 1e5 GeV.

    Files are from http://www.icrr.u-tokyo.ac.jp/~mhonda/nflx2014/index.html section 2.6
    '''
    file1 = 'data/spl-nu-20-01-000.d'
    file2 = 'data/spl-nu-20-01-n3650.d'
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
    df_list.append(df_list[-1]) #Put the fluxes for z=-0.9 to -1 at the end and reinterpret them as z=-1 to -1.1 since flux should be symmetric about z=-1.
    doubled_df_list = double_z_bins(df_list[9:21]) #Split the bins in two since IC 2020 has 20 bins instead of 10 in our range.
    single_df = pd.concat(doubled_df_list) #Make it a single df instead
    df = single_df.query('z_max != 0.1').query('z_min!=-1.1') #Remove two redundant fluxes
    flux_df = extrapolate_flux(single_df) #Extrapolate flux to 1e5 GeV
    return flux_df
def get_Aeff_2015_df():
    '''
    Data from 2015. Paper: doi:10.3847/1538-4357/835/2/151
    Returns a df with cols 'E_min', 'E_max', 'Aeff'
    '''
    file1 = 'data/7yrPS_data/IC2012-2015_PS_effA.dat'
    colnames = ['E_min', 'E_max', 'D_min', 'D_max', 'Aeff']

    df = pd.read_csv(file1, header=None, skiprows=1,names=colnames, dtype = np.float64, skipinitialspace=True, sep=' ')
    df['D_avg'] = (df['D_min']+df['D_max'])/2
    df['E_min'] = df['E_min']*1e3 #Convert TeV to GeV
    df['E_max'] = df['E_max']*1e3 #Convert TeV to GeV
    #df['z_avg'] = np.sin(np.pi/180*df['D_avg']) #cos(zenith) = sin(decl)
    #df = df[df.z_avg <= 0.]
    #df1 = df.groupby(['E_min', 'E_max'], as_index=False).mean()
    return df

def get_Aeff_2015_df_old():
    '''
    Data from 2015. Paper: doi:10.3847/1538-4357/835/2/151
    Returns a df with cols 'E_min', 'E_max', 'Aeff'
    '''
    file1 = 'data/7yrPS_data/IC2012-2015_PS_effA.dat'
    colnames = ['E_min', 'E_max', 'D_min', 'D_max', 'Aeff']

    df = pd.read_csv(file1, header=None, skiprows=1,names=colnames, dtype = np.float64, skipinitialspace=True, sep=' ')
    #df['z_avg'] = np.sin(np.pi/180*df['D_avg']) #cos(zenith) = sin(decl)
    df = df[df.D_max <= -5] #Horizon has decl =-5 according to paper
    df['E_min'] = df['E_min']*1e3 #Convert TeV to GeV
    df['E_max'] = df['E_max']*1e3 #Convert TeV to GeV
    df1 = df.groupby(['E_min', 'E_max'], as_index=False).mean()
    return df1[['E_min','E_max','Aeff']]

def get_energy_resolution_df():
    '''
    Returns a tuple num,nue of the extractd energy resolutions in http://arxiv.org/abs/1311.4767.
    Energy resolutions are a standard deviation, expressed as a percentage of the Etrue at that point.
    '''
    file1 = 'data/num_energy_resolution.csv'
    file2 = 'data/nue_energy_resolution_high.csv'
    file3 = 'data/nue_energy_resolution_low.csv'

    num = pd.read_csv(file1, skiprows=0, header=None, names=['exponent', 'num_sigma'], dtype = np.float64, skipinitialspace=True, sep=',')
    num['Etrue'] = 10**num['exponent']

    nue_high = pd.read_csv(file2, skiprows=0, header=None, names=['exponent', 'nue_sigma'], dtype = np.float64, skipinitialspace=True, sep=',')

    nue_low = pd.read_csv(file3, skiprows=0, header=None, names=['exponent', 'nue_sigma'], dtype = np.float64, skipinitialspace=True, sep=',')
    nue_low['exponent'] = nue_low['exponent']/10

    nue = pd.concat([nue_high, nue_low])
    nue['Etrue'] = 10**nue['exponent']
    return num, nue

def aeff_weight(z):
    '''
    Returns a tuple of the relative distance to the next interpolator, together with the interpolator index.
    '''
    decl = 90 - np.arccos(z)*180/np.pi
    D = np.array([-90,-30,-5])
    index = np.searchsorted(D, decl)
    try:
        return D[index]/decl, index-1
    except IndexError:
        return 1, 2

def get_Aeff_2015(E_range, z_range,interp_list):
    '''
    Accepts E_range as a mesh, and z_range as an array.
    Returns a weighted interpolation of the Aeff interpolators closest to each zenith in z_range.
    '''
    assert np.ndim(z_range) == 1, 'z needs to be an array of dimension 1'
    result = []
    for z in z_range:
        weight,i = aeff_weight(z)
        aeff = weight*interp_list[i](E_range) + (1-weight)*interp_list[i-1](E_range)
        result.append(aeff)
    res = np.array(result)
    return res[:,0,:] #Removes a redundant dimension

def interpolate_Aeff_2015_old(df):
    '''
    Returns a single interpolator. 
    Uses 1D interpolator cubic since 2015 Aeff data is 1D
    TODO: check that cubic extrapolator is good
    '''
    df['E_avg'] = (df['E_min']+df['E_max'])/2
    z_avg = interpolate.interp1d(df['E_avg'],df['Aeff'], bounds_error=False, fill_value='extrapolate', kind='cubic')
    return z_avg

def interpolate_Aeff_2015(df):
    '''
    Returns a single interpolator. 
    Uses 1D interpolator cubic since 2015 Aeff data is 1D
    TODO: check that cubic extrapolator is good
    '''
    D_min = np.array([-90.0,-30.0,-5.0])
    interpolators=[]
    for Dmin in D_min:
        df1 = df[df.D_min == Dmin]
        df1['E_avg'] = (df1['E_min']+df1['E_max'])/2
        z_avg = interpolate.interp1d(df1['E_avg'],df1['Aeff'], bounds_error=False, fill_value='extrapolate', kind='cubic')
        interpolators.append(z_avg)
    return interpolators

def interpolate_flux_old(df):
    '''
    Returns a df of the interpolated fluxes. 
    '''
    colnames = ['m_flux', 'mbar_flux', 'e_flux', 'ebar_flux']

    # Add fluxes for tails (z=-1,0)  so that these can be interpolated
    import warnings

    
    df_edges = df.query('z_max == 0.00 | z_min == -1.00')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_edges['z_min'] = np.round(df_edges['z_min'],2).replace(-0.05,0.00)
        df_edges['z_max'] = np.round(df_edges['z_max'],2).replace(-0.95,-1.0)
    df_fixed = pd.concat([df_edges, df])

    E = df_fixed.GeV
    z_avg = (df_fixed.z_min + df_fixed.z_max)/2

    points_avg = np.array([E,z_avg]).T

    interp_list=[]
    for flavor in colnames:
        phi = df_fixed[flavor]
        values=np.array(phi)

        f_avg = CT(points_avg, values,rescale=True) #Rescale seems to have no effect, but is good according to doc
        interp_list.append([f_avg])

    inter_df = pd.DataFrame(np.transpose(interp_list), columns=colnames)
    return inter_df

def get_Aeff_2015_old(E_range, f_avg):
    '''
    Returns the effective area averaged over declination (which is related to zenith).
    '''
    return f_avg(E_range)


def interpolate_energy_resolution(df):
    '''
    Returns a interpolator using interp1d NN with extrapolation
    '''
    sigma = df.iloc[:, 1] # Second column contains the sigma values
    inter = interpolate.interp1d(df['Etrue'], sigma,fill_value='extrapolate',kind='cubic')
    return inter

def get_angErr_df():
    '''
    DEPRECATED. Use get_angErr_2015_df()
    '''
    print('DEPRECATED')
    file1 = 'data/IC86-2012-events.txt'
    colnames = ['days', 'logE', 'angErr', 'RA', 'dec', 'azi', 'theta']

    df = pd.read_csv(file1, header=None, skiprows=1,names=colnames, dtype = np.float64, skipinitialspace=True, sep=' ')
    #df_filter = df.query('theta >= 90').query('theta<= 270').query('angErr <=2.5')
    df_filter = df.query('angErr <=2.5')
    df_filter['E'] = 10 ** np.round(df_filter['logE'],2)
    
    df1 = df_filter.groupby(by='E').median().reset_index()
    df1['z'] = np.cos(df1['theta']*np.pi/180)
    return df1[['E','logE','angErr','theta']]

def get_x_df():
    '''
    Reads the files data/XCC.dat and data/XNC.dat and returns the tuple (CC,NC) of cross-sections.

    Files from Globes site https://www.mpi-hd.mpg.de/personalhomes/globes/glb/
    '''

    file1 = 'data/XCC.dat'
    file2 = 'data/XNC.dat'
    colnames = ['gamma', 'e_X', 'm_X', 't_X', 'ebar_X','mbar_X','tbar_X']

    CC = pd.read_csv(file1, skiprows=30, header=None, names=colnames, dtype = np.float64, skipinitialspace=True, sep=' ')
    NC = pd.read_csv(file2, skiprows=30, header=None, names=colnames, dtype = np.float64, skipinitialspace=True, sep=' ')

    NC['GeV'] = 10**NC['gamma']
    NC = (NC.drop('gamma', axis=1))
    CC['GeV'] = 10**CC['gamma']
    CC = (CC.drop('gamma', axis=1))
    return CC,NC

def get_angErr_2015_df():
    '''
    Data from 2015. Not zenith dependent.
    E in [GeV], Aeff in [m^2]
    '''
    file1 = 'data/7yrPS_data/IC2012-2015_PS_PSF.dat'
    colnames = ['E', 'angErr']

    df = pd.read_csv(file1, header=None, skiprows=1,names=colnames, dtype = np.float64, skipinitialspace=True, sep=' ')
    df['E'] = 1e3*df['E'] # Convert to GeV
    df['angErr'] = df['angErr'] * np.pi/180 # Convert to rad
    return df

    
def compare_angErr():
    '''
    Use z-independent data from 2015 after discussion with S
    '''
    df1 = get_angErr_df()
    df2 = get_angErr_2015_df()
    df1_rounded = df1
    df1_rounded['logE'] = np.round(df1_rounded['logE'],1)
    df1_rounded = df1_rounded.groupby('logE',as_index=False).mean()
    x = df1_rounded.logE
    y= df1_rounded.z
    z = df1_rounded.angErr
    z1 = interpolate.interp1d(x,z, bounds_error=False, fill_value=1)
    plt.scatter(x,z,s=3, label='averaged 2012 data')
    E_range = np.logspace(2,5,100)
    plt.plot(np.log10(E_range),z1(np.log10(E_range)), c='r', label='interpolated 2012 data')
    plt.scatter(np.log10(df2.E),df2.angErr,s=3,c='k', label='2015 data, no zenith')
    #plt.xlim((2,5))
    #plt.ylim((0,2))
    plt.legend()
    plt.show()


def interpolate_angErr(df):
    '''
    DEPRECATED
    Trains the angErr interpolator on E and z using an interpolator
    TODO: fine-tune interpolator. Cubic doesnt work because cubic extrapolation sucks at z > -0.8
    '''
    print('DEPRECATED')
    x = 10**df['logE']
    y = df['theta']
    z = df['angErr']

    points = np.array([x,y]).T
    values = np.array(z)
    #f = SBS(x,y,z)
    #f = CT(points,values,rescale=False,fill_value=0)
    def grid_f(x1,y1):
        interp_points = np.array([x1,y1]).T
        #f = interpolate.griddata(points, values, interp_points, method='linear', fill_value=0, rescale=True)
        rbfi = interpolate.Rbf(x,y,z,smooth=10, function='linear')
        f = rbfi(x1,y1)
        return f
    return grid_f

def interpolate_angErr_2015(df):
    '''
    Uses 1D interpolator cubic since 2015 angErr data is 1D
    '''
    x = df['E']
    z = df['angErr']

    interp = interpolate.interp1d(x,z, bounds_error=False, fill_value='extrapolate', kind='cubic')
    return interp



def get_angErr(E_range,z_range,f):
    '''
    DEPRECATED. Use get_angErr_2015()

    E_range = energy proxy
    z_range = reconstructed
    Returns the values from the interpolator.
    Uses np.meshgrid() if not already passed in E_range,z_range
    '''
    try:
        values_inter=f(E_range,z_range)
    except ValueError:
        X, Y = np.meshgrid(E_range,z_range)
        values_inter=f(X,Y)
    return values_inter

def get_angErr_2015(E_range,f):
    '''
    Returns the angular error in radians for a E_range [GeV]
    '''
    return f(E_range)

    

def get_Aeff(E_range,z_range,f_max, f_min):
    '''
    DEPRECATED. Use get_Aeff_2015()

    Returns the average interpolated Aeff for the values in the arrays E_range and z_range.
    For the z edges 0 and -1, one of the interpolators fail, so then we just use the other one.
    '''
    print('DEPRECATED')
    points = np.array([E_range,z_range]).T
    values_max = f_max(points) 
    values_min = f_min(points)
    # interpolator returns nan at z=0 and -1. 
    if np.isnan(f_min(points)).any() | np.isnan(f_max(points)).any():
        try:
            a = fill_NN(values_min)
            b=fill_NN(values_max)
            return (a.T+b.T)/2
        except:
            try:
                a = fill_NN(values_min)
                return a.T
            except:
                return values_max.T
            try:
                a=fill_NN(values_max)
                return a.T
            except:
                return values_min.T
    else:
        return (values_max.T + values_min.T)/2

def interpolate_X(CC,NC):
    '''
    Uses cubic spline interp
    olation to interpolate X, and linear interp1d to extrapolate X.
    Returns the tuple with dfs containing functions (CC_list, NC_list)
    
    CC_list and NC_list has the structure [flavor]
    '''
    E_CC = CC.GeV
    E_NC = NC.GeV

    CC_list = []
    NC_list =[]
    for flavor in ['e_X', 'm_X', 't_X', 'ebar_X','mbar_X','tbar_X']:
        interp_CC = interpolate.interp1d(E_CC, CC[flavor],fill_value=CC[flavor].iloc[-1],kind='cubic', bounds_error=False)
        interp_NC = interpolate.interp1d(E_NC, NC[flavor],fill_value=NC[flavor].iloc[-1], kind='cubic', bounds_error=False)
        CC_list.append(interp_CC)
        NC_list.append(interp_NC)
    CC_df = pd.DataFrame([CC_list],columns=['e_X', 'm_X', 't_X', 'ebar_X','mbar_X','tbar_X'])
    NC_df = pd.DataFrame([NC_list],columns=['e_X', 'm_X', 't_X', 'ebar_X','mbar_X','tbar_X'])

    return CC_df, NC_df

