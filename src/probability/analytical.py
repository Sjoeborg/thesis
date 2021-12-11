import numpy as np
from functions import V,dm, mass_dict,theta,param_dict,GeV2tokm1,U_3,r_earth, nufit_params_nsi, baseline
        
def P_an(flavor_from, flavor_to, param, E=None, L=None,  param_min=None, param_max=None, material='vac',earth_start = 0, ndim = 3, anti=False, theta_i=0, npoints=500, params=param_dict):
    '''
    Returns the analytical probability as an array for a collection of L[m/km] and E[MeV/GeV]
    '''
    if param == 'E':
        E = np.logspace(np.log10(param_min), np.log10(param_max), npoints)
        assert(L is not None)
    elif param == 'L':
        L = np.linspace(param_min, param_max, npoints)
        assert(E is not None)
    imag_sign = 1
    if anti is True: # Imaginary part get a - for antineutrinos
        imag_sign = -1
    alpha = mass_dict[flavor_from]
    beta = mass_dict[flavor_to]

    if ndim == 2:
        if param == 'E':
            probs = [P_two(alpha, beta, E=en, L =L, theta_i=theta_i , material=material, params=param_dict) for en in E]
        elif param == 'L':
            probs = [P_two(alpha, beta, E=E, L =ell, theta_i=theta_i , material=material, params=param_dict) for ell in L]
        return np.array(probs)
    else:
        if param == 'E':
            probs = [P_n(alpha, beta, E=en, L =L, ndim=ndim, params=param_dict) for en in E]
            return np.array(probs)
        elif param == 'L':
            probs = [P_n(alpha, beta, E=E, L =ell, ndim=ndim, params=param_dict) for ell in L]
            return np.array(probs)


def P_n(alpha,  beta, E, ndim, L=2*r_earth, params=param_dict): 
    '''
    Returns the analytical probability as a scalar for L[m/km] and E[MeV/GeV]
    '''
    P_list = []

    U = U_3(params=params)
    U_conj = np.conj(U)
    for i in range(0, ndim):
        for j in range(0, i): #1605.08607 eq3, giunti 7.38 and 7.51
            m = dm(i+1, j+1, params=params)
            U_product = U_conj[alpha][i] * U[beta][i] * U[alpha][j] * U_conj[beta][j]
            real_part = np.real(U_product)*np.sin(GeV2tokm1*m*L/(4*E)) ** 2 
            P_list.append(-4*real_part)
    if  alpha == beta:
        return 1 + np.sum(P_list)
    else:
        return np.sum(P_list)

def P_two(alpha, beta, E, L, material, theta_i, params=param_dict):#giunti 9.77
    flavor_from=mass_dict[alpha]
    flavor_to=mass_dict[beta]
    r = baseline(theta_i)
    A = 2*E*V(r, material)[0]

    th_M = theta(alpha+1,beta+1, params=params)
    dm_M = dm(alpha+1, beta+1, params=params)

    return np.sin(2*th_M)**2 * np.sin(GeV2tokm1*dm_M*L/(4*E))**2


if __name__ == '__main__':
    print(P_n('m',  'm', 1e2, 3, L=2*r_earth, params=nufit_params_nsi))