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
U_e4 = 0.0225 # abs^2 1202.1024
U_m4 = 0.0289 # abs^2 1202.1024
# 2008.12769 DUNE 90% CL
a_ee = 0.3
a_mm = 0.2
a_tt = 0.8
a_me = 0.04
a_te = 0.7
a_tm = 0.2


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

#---- Mixing parameters and matrices -------
def dm_vac(first, second,params=param_dict): #dm_ij = dm_kj - dm_ki, dm_ij = -dm_ji
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


def dm(first, second,A=0, params=param_dict): # 0709.1937 99.67
    '''
    Takes A in GeV^2, converts it to eV^2 and returns the dm_M in eV^2
    '''
    dm_v = dm_vac(first,second, params=params)
    if A == 0: # do this check because sqrt removes sign. Accefts ordering
        return dm_v
    else: 
        print('dm_M only valid for 2 generations')
        return np.sqrt( (dm_v * np.cos(2*theta(second, first,params=params)) -1e18*A)**2 + (dm_v * np.sin(2*theta(second, first, params=params)))**2)


def theta(first, second, A = 0, params=param_dict):
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
    if A != 0: #0709.1937 eq 9
        th_M = np.arcsin(np.sin(2*th) * dm(second, first, params=params) / dm(second, first, A=A, params=params)) / 2
        return th_M
    else:
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


def U_nu(flavor_from=None, flavor_to=None, A = 0, ndim=3, params=param_dict):
    if ndim == 3:
        atm_lbl = np.array([[1, 0, 0],
                       [0, np.cos(theta(2,3,A, params=params)), np.sin(theta(2,3,A, params=params))],
                       [0, -np.sin(theta(2,3,A, params=params)), np.cos(theta(2,3,A, params=params))]])
        reactor = np.array([[np.cos(theta(1,3,A, params=params)), 0, np.sin(theta(1,3,A, params=params))],
                            [0, 1, 0],
                            [-np.sin(theta(1,3,A, params=params)), 0, np.cos(theta(1,3,A, params=params))]])
        solar = np.array([[np.cos(theta(1,2,A, params=params)), np.sin(theta(1,2,A, params=params)), 0],
                        [-np.sin(theta(1,2,A, params=params)), np.cos(theta(1,2,A, params=params)), 0],
                        [0, 0, 1]])
        return atm_lbl @ reactor @ solar
        
    elif ndim == 2:
        alpha = mass_dict[flavor_from]
        beta = mass_dict[flavor_to]
        if alpha == beta:
            raise ValueError('U_nu only takes different flavors for 2D. To get survival, use 1-P_ab instad')
        return np.array([[np.cos(theta(alpha+1,beta+1, A, params=params)), np.sin(theta(alpha+1,beta+1, A, params=params))],
                        [-np.sin(theta(alpha+1,beta+1, A, params=params)), np.cos(theta(alpha+1,beta+1, A, params=params))]])

    elif ndim == 4: # Blennow 78:807 eq 2
        return V_matrix(3,4,A,1, params=params) @ V_matrix(2,4,A,1, params=params) @ V_matrix(1,4,A,1, params=params) @ V_matrix(2,3,A,1, params=params) @ V_matrix(1,3,A,1, params=params)@ V_matrix(1,2,A,1, params=params)

    elif ndim == 5: # Kopp eq 2.6
        return V_matrix(3,5,A,2, params=params) @ V_matrix(2,5,A,2, params=params) @ V_matrix(1,5,A,2, params=params) @ V_matrix(3,4,A,2, params=params) @ V_matrix(2,4,A,2, params=params) @ V_matrix(1,4,A,2, params=params) @ V_matrix(2,3,A,2, params=params) @ V_matrix(1,3,A,2, params=params)@ V_matrix(1,2,A,2, params=params)


def V_ijab(i,j,a,b, A=0, params=param_dict): # Blennow 78:807
    if a == b:
        if a == i or a == j:
            return np.cos(theta(i,j,A, params=params))
        else:
            return 1
    else:
        if a == i and b == j:
            return np.sin(theta(i,j,A, params=params))
        elif a == j and b == i:
            return -np.sin(theta(i,j,A, params=params))
        else:
            return 0


def V_matrix(i,j, A=0, n = 0, params=param_dict): # Blennow 78:807
    result = np.empty(shape=(3+n,3 + n), dtype=np.complex64)
    for a in range(1, 4+n):
        elem = (V_ijab(i,j,a,b,A, params=params) for b in range(1, 3 + n + 1))
        V_row = np.fromiter(elem, dtype=np.complex64, count=3 + n)
        result[a-1] = V_row
    return result

# ------ Matter effects -------
def rho_earth(r):# lä 1306.2903 och 0612285 ocg kuo
    '''
    Calculates the density in g/cm-3 of the point at a distance r in km from the Earth's center using data from PREM https://www.cfa.harvard.edu/~lzeng/papers/PREM.pdf
    Returns the tuple (integrated density, density in gcm-3, derivative of density)
    '''
    
    if not np.isscalar(r): #If r is array, return list of densities.
        return [rho_earth(item) for item in r]
    
    x = r / r_earth #Normalized Earth radius in km
    if 0 <= r < 1221.5:
        return (13.0885*x - 8.8381*x**3/3, 13.0885 - 8.8381*x**2, - 2*8.8381*x)
    elif 1221.5 <= r < 3480.0:
        return (12.5815*x - 1.2638*x**2/2 - 3.6426*x**3/3 - 5.5281*x**4/4, 12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3, - 1.2638 - 2*3.6426*x - 3*5.5281*x**2)
    elif 3480.0 <= r  < 5701.0:
        return (7.9565*x - 6.4761*x**2/2 + 5.5283*x**3/3 - 3.0807*x**4/4, 7.9565 - 6.4761*x + 5.5283*x**2 - 3.0807*x**3, - 6.4761 + 2*5.5283*x - 3*3.0807*x**2)
    elif 5701.0 <= r < 5771.0:
        return (5.3197*x - 1.4836*x**2/2, 5.3197 - 1.4836*x, - 1.4836)
    elif 5771.0 <= r < 5971.0:
        return (11.2494*x - 8.0298*x**2/2, 11.2494 - 8.0298*x, - 8.0298)
    elif 5971.0 <= r < 6151.0:
        return (7.1089*x - 3.8045*x**2/2, 7.1089 - 3.8045*x, - 3.8045)
    elif 6151.0 <= r < 6346.6:
        return (2.6910*x + 0.6924*x**2/2, 2.6910 + 0.6924*x, 0.6924)
    elif 6346.6 <= r < 6356.0:
        return (2.9*x, 2.9, 0)
    elif 6356.0 <= r < 6368.0:
        return (2.6*x, 2.6, 0)
    elif 6368.0 <= r < r_earth:
        return (1.020*x, 1.020, 0)
    else:
        return (0.0, 0.0, 0.0)


def V(r, material='earth'):
    '''
    Takes the distance from the Earth's core in [km] and returns the CC potential(+) and NC potential(-) in [GeV].
    '''
    if material == 'earth':
        rho = rho_earth(r)[1]
    elif type(material) is float:
        rho = material
    elif material == 'vac':
        rho = 0
    Y = get_electron_density(r)
    V_cc = np.sqrt(2) * GF * Y * N_A * rho * (1/GeVtocm1)**3 #0709.1937 eq 6
    V_nc =  -1/np.sqrt(2)* GF * Y * N_A * rho * (1/GeVtocm1)**3 #0709.1937 eq 7
    return V_cc, V_nc 


def get_radial_distance(x,theta_i):
    '''
    Returns the distance to the Earth core in [km] for a baseline x [km] and a zenith angle theta_i
    '''
    x = 2*r_earth - x # Flip, otherwise we go from Earth out to infinity.
    L = 2*r_earth / (np.cos(theta_i) + np.sin(theta_i)*np.tan(theta_i))
    r = np.sqrt((L-x)**2 + r_earth**2 - 2*(L-x)*r_earth*np.cos(theta_i))

    return r


def baseline(theta):
    '''
    Returns the distance [km] travelled. Should be equivalent to get_radial_distance for h=0, L=2*r_earth
    TODO: maybe fix this
    '''
    production_height = 15. # Assuming neutrino produced 15 km above surface
    detector_depth = 1. # Assuming detector depth of 1 km
    z = np.cos(theta)
    return np.sqrt((r_earth + production_height)**2 - r_earth**2*np.sin(theta)**2) - r_earth*np.cos(theta)
    #return -r_earth*z +  np.sqrt( (r_earth + production_height)**2 - r_earth**2*np.sin(theta)**2 )
    #np.sqrt((r_earth + h)**2 - r_earth**2*np.sin(theta)**2) - r_earth*np.cos(theta)


def get_electron_density(r):
    '''
    Returns the fraction of electrons per nucleon Y_e for a radial distance to the Earth core in [km]
    '''
    if r >= r_core:
        return 0.4957
    elif r < r_core:
        return 0.4656
    else:
        raise ValueError(f'{r} not numerical')

def chisq_S(params,events, data,sigma_a, sigma_b, sigma_g, sigma_syst):
    if len(params) == 3:
        a,b, g = params
        S_th = events * (1 + a*sigma_a + b*sigma_b + g*sigma_g)
        penalty = a**2 + b**2 + g**2
    elif len(params) == 2:
        a,b = params
        S_th = events * (1 + a*sigma_a + b*sigma_b)
        penalty = a**2 + b**2
    elif len(params) == 1:
        a = params
        S_th = events * (1 + a*sigma_a)
        penalty = a**2 
    chi2= np.sum((S_th - data)**2/(data + sigma_syst**2))+ penalty
    return chi2


def chisq(params,events, data,z,sigma_a, sigma_b, sigma_g, sigma_syst):
    z_0 = -np.median(z)
    if len(params) == 3:
        a,b, g = params
        S_th = a*(1+b*(z+z_0)+g)*events 
        penalty = (1-a)**2/sigma_a**2 + b**2 / sigma_b**2 + g**2 /sigma_g**2
    elif len(params) == 2:
        a,b = params
        S_th = a*(1+b*(z+z_0))*events
        penalty = (1-a)**2/sigma_a**2 + b**2 / sigma_b**2 
    elif len(params) == 1:
        a = params
        S_th = a*events
        penalty = a**2 
    chi2= np.sum((S_th - data)**2/(data + sigma_syst**2))+ penalty
    return chi2

def perform_chisq(events, data,sigma_syst, z = np.linspace(-1,0,21), sigma_a=0.25, sigma_b=None, sigma_g =None, x0=[1]):
    res = minimize(fun=chisq, x0=x0, args=(events,data,z,sigma_a, sigma_b, sigma_g, sigma_syst), method='Nelder-Mead',options={'maxiter': 1e5, 'maxfev':1e5})
    assert res.success, res
    return res.fun, res.x

def perform_chisq_S(events, data,sigma_syst, z = np.linspace(-1,0,21), sigma_a=0.25, sigma_b=None, sigma_g =None, x0=[1]):
    res = minimize(fun=chisq_S, x0=x0, args=(events,data,sigma_a, sigma_b, sigma_g, sigma_syst), method='Nelder-Mead',options={'maxiter': 1e5, 'maxfev':1e5})
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
    events = np.array([[3,4],[5,1]])
    events_gamma = np.array([[4,6],[6,2]])
    sigma_g = (np.sum(events)- np.sum(events_gamma))/np.sum(events)
    data = np.array([[4,8],[4,1]])
    #print(chisq(params=np.array([1,0,0]),events=events, data=data,z = np.array([-0.5,-0.6]),sigma_a=0.25, sigma_b=0.15, sigma_g=sigma_g, sigma_syst=0))
    print(perform_chisq(x0=np.array([1,0,0]),events=events, data=data,z = np.array([-0.5,-0.6]),sigma_a=0.25, sigma_b=0.15, sigma_g=sigma_g, sigma_syst=0))
    print(perform_chisq_S(x0=np.array([1,0,0]),events=events, data=data,z = np.array([-0.5,-0.6]),sigma_a=0.25, sigma_b=0.15, sigma_g=sigma_g, sigma_syst=0))