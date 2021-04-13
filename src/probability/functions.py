import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simps,romb
mass_dict = {'e': 0, 'm': 1, 't': 2, 's1': 3, 's2': 4, 0: 'e', 1: 'm', 2: 't', 3: 's1', 4: 's2'}
#---Natural constants-----
GF = 1.16637876e-5 #GeV^-2
N_A = 6.0221409e23
c = 299792458
hbar = 1.054571800e-34
e = 1.6021766209e-19
r_earth = 6371.0 # km
r_core = 3480.0 #km

#--- Neutrino parameters ------
theta_12 = 33.44 * np.pi/180 #Nufit
theta_13 = 8.57 * np.pi/180 #Nufit
theta_23 = 49.0 * np.pi/180 #Nufit
delta_ij = 195 * np.pi/180 #Nufit
dm_21 = 7.42e-5 # eV^2 Nufit
dm_31 = 2.514e-3 #eV^2 Nufit

theta_14 = np.arcsin(np.sqrt(0.00)) 
theta_24 = np.arcsin(np.sqrt(0.04)) 
theta_34 = np.arcsin(np.sqrt(0.00)) 


dm_41 = -1



GeVtocm1 = e / (c*hbar) *1e7 # 
GeV2tokm1 = e / (c*hbar) *1e-6 # 
param_dict= {'theta_12': theta_12,
             'theta_13': theta_13,
             'theta_23': theta_23,
             'theta_34': theta_34,
             'theta_24': theta_24,
             'theta_14': theta_14,
             'delta_ij': delta_ij,
             'dm_21': dm_21,
             'dm_31': dm_31,
             'dm_41': dm_41}
ic_params = param_dict.copy() #IC2020 companion paper cites PDG 2014
ic_params.update({'theta_12': np.arcsin(np.sqrt(0.846))/2, 'theta_23': np.arcsin(np.sqrt(0.999))/2, 'theta_13': np.arcsin(np.sqrt(9.3e-2))/2, 'dm_21': 7.53e-5, 'dm_31': 7.53e-5 + 2.44e-3})
ic_params.update({'theta_14': 0, 'theta_24': np.arcsin(np.sqrt(0.10))/2, 'theta_34': 0, 'dm_41': 4.5})
ic_params_nsi = ic_params.copy()
ic_params_nsi.update({'e_ee':0,'e_me':0,'e_et':0,'e_mm':0,'e_mt':0,'e_tt':0,'e_es':0,'e_ms':0,'e_ts':0,'e_ss':0})

dc_params = param_dict.copy() #From demidov 2020
dc_params.update({'theta_12': 0.5839958715755919, 'theta_23': 0.8054035005744429, 'theta_13': 0.14784723758432042, 'dm_21': 7.53e-5, 'dm_31': 7.53e-5 + 2.29e-3})
dc_params_nsi = dc_params.copy()
dc_params_nsi.update({'e_ee':0,'e_me':0,'e_et':0,'e_mm':0,'e_mt':0,'e_tt':0,'e_es':0,'e_ms':0,'e_ts':0,'e_ss':0})

#---- Mixing parameters and matrices -------
def dm(first, second,params=param_dict): #dm_ij = dm_kj - dm_ki, dm_ij = -dm_ji
    if first == 1: # m_1j
        if second == 2: #m_k2
            return -params['dm_21']
        elif second == 3: #m_k3
            return -params['dm_31']
        elif second == 4:
            return -params['dm_41']
        elif second == 5:
            return -params['dm_51']
        
    
    if first == 2: # m_2j
        if second == 1: #m_k1
            return params['dm_21']
        elif second == 3: #m_k3
            try:
                return params['dm_23']
            except:
                try:
                    return -params['dm_32']
                except:
                    return params['dm_21'] - params['dm_31']
        elif second == 4:
            try:
                return params['dm_24']
            except:
                try:
                    return -params['dm_42']
                except:
                    return params['dm_21'] - params['dm_41']
        elif second == 5:
            return params['dm_21'] - params['dm_51']
    
    if first == 3: # m_3j
        if second == 1: #m_k1
            return params['dm_31']
        elif second == 2: #m_k2
            try:
                return params['dm_32']
            except:
                try:
                    return -params['dm_23']
                except:
                    return params['dm_31'] - params['dm_21']
        elif second == 4:
            try:
                return params['dm_34']
            except:
                try:
                    return -params['dm_43']
                except:
                    return params['dm_31'] - params['dm_41']
        elif second == 5:
            return params['dm_31'] - params['dm_51']
    
    if first == 4: # m_4j
        if second == 1: #m_k1
            return params['dm_41']
        elif second == 2: #m_k2
            try:
                return params['dm_42']
            except:
                try:
                    return -params['dm_24']
                except:
                    return params['dm_41'] - params['dm_21'] 
        elif second == 3:
            try:
                return params['dm_43']
            except:
                try:
                    return -params['dm_34']
                except:
                    return params['dm_41'] - params['dm_31']
        elif second == 5:
            return params['dm_41'] - params['dm_51']
    if first == 5:
        if second == 1:
            return params['dm_51']

    if first == second:
        return 0
    
    else:
        raise ValueError(f'dm() recieved faulty args {first}, {second}')




def theta(first, second, params=param_dict):
    if first == 1: # th_1j
        if second == 2: #th_k2
            th = params['theta_12']
        elif second == 3: #th_k3
            th = params['theta_13']
        elif second == 4: #th_k4
            th = params['theta_14']
        elif second == 5: #th_k5
            th = params['theta_15']
        else:
            raise ValueError(f'theta() recieved faulty args {first}, {second}')
    
    elif first == 2: # th_2j
        if second == 1: #th_k1
            th = -params['theta_12']
        elif second == 3: #th_k3
            th = params['theta_23']
        elif second == 4: 
            th = params['theta_24']
        elif second == 5: #th_k5
            th = params['theta_25']
        else:
            raise ValueError(f'theta() recieved faulty args {first}, {second}')
    
    elif first == 3: # th_3j
        if second == 1: #th_k1
            th = -params['theta_13']
        elif second == 2: #th_k2
            th = -params['theta_23']
        elif second == 4:
            th = params['theta_34']
        elif second == 5: #th_k5
            th = params['theta_35']
        else:
            raise ValueError(f'theta() recieved faulty args {first}, {second}')
    
    elif first == 4:
        if second == 1:
            th = -params['theta_14']
        elif second == 2:
            th = -params['theta_24']
        elif second == 3:
            th = -params['theta_34']
        elif second == 5: #th_k5
            th = params['theta_45']
        else:
            raise ValueError(f'theta() recieved faulty args {first}, {second}')
    
    else:
        raise ValueError(f'theta() recieved faulty args {first}, {second}')
    return th

def R23_3(theta):
  return np.array([[ 1,             0,             0],
                   [ 0, np.cos(theta),  np.sin(theta)],
                   [ 0, -np.sin(theta), np.cos(theta)]])
def R13_3(theta):
  return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0,             1,             0],
                   [-np.sin(theta), 0, np.cos(theta)]])
def R12_3(theta):
  return np.array([[ np.cos(theta),  np.sin(theta), 0],
                   [ -np.sin(theta), np.cos(theta), 0],
                   [ 0             , 0            , 1 ]])

def R23_4(theta):
  return np.array([[ 1,             0,             0, 0],
                   [ 0, np.cos(theta),  np.sin(theta),0],
                   [ 0, -np.sin(theta), np.cos(theta),0],
                   [ 0,             0,             0, 1]])
def R13_4(theta):
  return np.array([[ np.cos(theta),  0,np.sin(theta), 0 ],
                   [ 0             , 1            , 0, 0],
                   [ -np.sin(theta), 0,np.cos(theta), 0],
                   [ 0             , 0            , 0, 1]])
def R12_4(theta):
  return np.array([[ np.cos(theta),  np.sin(theta), 0, 0],
                   [ -np.sin(theta), np.cos(theta), 0, 0],
                   [ 0             , 0            , 1, 0],
                   [ 0             , 0            , 0, 1]])
def R34_4(theta):
  return np.array([[ 1,             0,             0, 0],
                   [ 0,             1,             0, 0],
                   [ 0,             0,np.cos(theta),  np.sin(theta)],
                   [ 0,             0,-np.sin(theta), np.cos(theta)]])
def R24_4(theta):
  return np.array([[ 1,0             ,             0, 0],
                   [ 0, np.cos(theta),0,  np.sin(theta)],
                   [ 0,0             ,1            , 0],
                   [ 0, -np.sin(theta), 0,np.cos(theta)]])
def R14_4(theta):
  return np.array([[ np.cos(theta),  0,0,  np.sin(theta)],
                   [ 0             , 1            , 0, 0],
                   [ 0             , 0            , 1, 0],
                   [ -np.sin(theta), 0,0,  np.cos(theta)]])

def U_4(params=param_dict):
    '''
    Returns the 4x4 mixing matrix as defined by Eq 1 in the 2020 IC companion paper, with the 3x3 mixing matrix defined in PDG.
    '''
    return R34_4(params['theta_34'])@ R24_4(params['theta_24'])@R14_4(params['theta_14']) @ R23_4(params['theta_23']) @ R13_4(params['theta_13']) @ R12_4(params['theta_12'])

def U_3(params=param_dict):
    '''
    Returns the 3x3 mixing matrix defined in PDG.
    '''
    return R23_3(params['theta_23']) @ R13_3(params['theta_13']) @ R12_3(params['theta_12'])

def U_5(params=param_dict):
    return  V_matrix(3,5, n = 2, params=params) @ V_matrix(2,5, n = 2, params=params) @ V_matrix(1,5, n = 2, params=params) @ V_matrix(3,4, n = 2, params=params) @ V_matrix(2,4, n = 2, params=params) @ V_matrix(1,4, n = 2, params=params)@ V_matrix(2,3, n = 2, params=params)@ V_matrix(1,3, n = 2, params=params)@ V_matrix(1,2, n = 2, params=params)

def V_ijab(i,j,a,b, A=0, params=param_dict): # Blennow 78:807
    if a == b:
        if a == i or a == j:
            return np.cos(theta(i,j, params=params))
        else:
            return 1
    else:
        if a == i and b == j:
            return np.sin(theta(i,j, params=params))
        elif a == j and b == i:
            return -np.sin(theta(i,j, params=params))
        else:
            return 0


def V_matrix(i,j, n = 0, params=param_dict): # Blennow 78:807
    result = np.empty(shape=(3+n,3 + n), dtype=np.complex64)
    for a in range(1, 4+n):
        elem = (V_ijab(i,j,a,b, params=params) for b in range(1, 3 + n + 1))
        V_row = np.fromiter(elem, dtype=np.complex64, count=3 + n)
        result[a-1] = V_row
    return result

# ------ Matter effects -------
def rho_earth(r):# lä 1306.2903 och 0612285 ocg kuo
    '''
    Calculates the density in g/cm-3 of the point at a distance r in km from the Earth's center using data from PREM https://www.cfa.harvard.edu/~lzeng/papers/PREM.pdf
    Returns the density in gm/cm-3
    '''
    if r == -1: #Neutrino doesn't traverse Earth
        return 0
    if not np.isscalar(r): #If r is array, return list of densities.
        return [rho_earth(item) for item in r]
    
    x = r / r_earth #Normalized Earth radius in km
    if 0 <= r < 1221.5:
        return 13.0885 - 8.8381*x**2
    elif 1221.5 <= r < 3480.0:
        return 12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3
    elif 3480.0 <= r  < 5701.0:
        return 7.9565 - 6.4761*x + 5.5283*x**2 - 3.0807*x**3
    elif 5701.0 <= r < 5771.0:
        return 5.3197 - 1.4836*x
    elif 5771.0 <= r < 5971.0:
        return 11.2494 - 8.0298*x
    elif 5971.0 <= r < 6151.0:
        return 7.1089 - 3.8045*x
    elif 6151.0 <= r < 6346.6:
        return 2.6910 + 0.6924*x
    elif 6346.6 <= r < 6356.0:
        return 2.9
    elif 6356.0 <= r < 6368.0:
        return 2.6
    elif 6368.0 <= r < r_earth:
        return 1.020
    else:
        return 0.0


def V(r, material='earth'):
    '''
    Takes the distance from the Earth's core in [km] and returns the CC potential(+) and NC potential(-) in [GeV].
    '''
    if material == 'earth':
        rho = rho_earth(r)
    elif type(material) is float:
        rho = material
    elif material == 'vac':
        rho = 0
    Y = get_electron_density(r)
    V_cc = np.sqrt(2) * GF * Y * N_A * rho * (1/GeVtocm1)**3 #0709.1937 eq 6
    V_nc =  -1/np.sqrt(2)* GF * Y * N_A * rho * (1/GeVtocm1)**3 #0709.1937 eq 7
    return V_cc, V_nc 


def baseline(theta_i):
    h = 15 # Production height
    if theta_i <= np.pi/2:
        r = np.sqrt((r_earth+h)**2 - r_earth**2*np.sin(theta_i)**2) - r_earth*np.cos(theta_i)
        r= 2*r_earth*np.cos(theta_i)
    elif theta_i > np.pi/2:
        r = h/np.cos(np.pi-theta_i)
    return r

def get_radial_distance(x,theta_i):
    '''
    Returns the distance to the Earth core in [km] for a baseline x [km] and an angle theta_i
    '''
    if theta_i <= np.pi/2:
        L = baseline(theta_i)
        r = np.sqrt((L-x)**2 + r_earth**2 - 2*(L-x)*r_earth*np.cos(theta_i))
    else: # Neutrino doesn't traverse Earth
        r = -1 
    return r

def get_electron_density(r):
    '''
    Returns the fraction of electrons per nucleon Y_e for a radial distance to the Earth core in [km]
    '''
    if r >= r_core:
        return 0.4957
    elif r < r_core and r > 0:
        return 0.4656
    elif r == -1: #Neutrino doesn't traverse Earth
        return 0

def chisq(params,events, data,z,sigma_a, sigma_b, sigma_g, sigma_syst):
    z_0 = -np.median(z)

    if len(params) == 3:
        a,b, g = params
        S_th = a*(1+b*(z[0:-1]+z_0)+g)*events 
        penalty = (1-a)**2/sigma_a**2 + b**2 / sigma_b**2 + g**2 /sigma_g**2
    elif len(params) == 2:
        a,b = params
        S_th = a*(1+b*(z[0:-1]+z_0))*events
        penalty = (1-a)**2/sigma_a**2 + b**2 / sigma_b**2 
    elif len(params) == 1:
        a = params
        S_th = a*events
        penalty = (1-a)**2/sigma_a**2
    chi2= np.sum((S_th - data)**2/(data + sigma_syst**2))+ penalty
    return chi2

def perform_chisq(events, data,sigma_syst, z = np.linspace(-1,0,21), sigma_a=0.25, sigma_b=None, sigma_g =None, x0=[1]):
    res = minimize(fun=chisq, x0=x0, args=(events,data,z,sigma_a, sigma_b, sigma_g, sigma_syst), method='Nelder-Mead',options={'maxiter': 1e5, 'maxfev':1e5})
    assert res.success, res
    return res.fun, res.x


def integrate(array,method='simps',*args):
    '''
    Args can be meshgrids or arrays.
    '''
    assert np.ndim(array) == len(args), 'Not enough args to integrate over'
    if method == 'romb':
        dx0 = (args[0].max() - args[0].min())/(len(args[0]-1))
        dx1 = (args[1].max() - args[1].min())/(len(args[1]-1))
        dx2 = (args[2].max() - args[2].min())/(len(args[2]-1))
        return romb(romb(romb(array, dx = dx0), dx = dx1), dx = dx2)

    if np.ndim(array) == 1:
        return simps(array,args[0])
    elif np.ndim(array) == 2:
        return simps(simps(array,args[0]),args[1])
    elif np.ndim(array) == 3:
        return simps(simps(simps(array,args[0]),args[1]),args[2])
    elif np.ndim(array) == 4:
        return simps(simps(simps(simps(array,args[0]),args[1]),args[2]),args[3])
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    z = 1
    print(baseline(np.pi - np.arccos(z)))
    print(get_radial_distance(500, np.pi - np.arccos(z)))
