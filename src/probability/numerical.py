from scipy.integrate import solve_ivp
import numpy as np
from multiprocessing import Pool
from probability.functions import dm,theta,V_matrix, V, mass_dict,r_earth, param_dict,GeV2tokm1,get_radial_distance, baseline,ic_params, U_4, U_3,U_5

def H_2(flavor_from, flavor_to, A=0, params=param_dict): # giunti 9.73
    i = mass_dict[flavor_from]
    j = mass_dict[flavor_to]
    m_M = dm(i+1,j+1, A, params=params)
    th_M = theta(i+1,j+1, A, params=params)
    H = np.array([ #kanske samma som U_nu(ndim=2,A)  M @ U_conj +
        [-m_M*np.cos(2*th_M), m_M*np.sin(2*th_M)],
        [m_M*np.sin(2*th_M),  m_M*np.cos(2*th_M)]])

    return H 


def H_3(A_cc,U,M, params=param_dict):
    A = 1e18*np.array([[A_cc, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]])
    U_conj=np.conj(U)
    H = U @ M @ U_conj.T + A 
    return H

def E_3(A_cc, params):
    H_eps =np.array([[params['e_ee'], params['e_me'], params['e_et']],
                     [params['e_me'], params['e_mm'], params['e_mt']],
                     [params['e_et'], params['e_mt'], params['e_tt']]])
    return 1e18*A_cc*H_eps

def E_4(A_cc, A_nc, params):
    H_eps =np.array([[params['e_ee'], params['e_me'], params['e_et'], params['e_es']],
                     [params['e_me'], params['e_mm'], params['e_mt'], params['e_ms']],
                     [params['e_et'], params['e_mt'], params['e_tt'], params['e_ts']],
                     [params['e_es'], params['e_ms'], params['e_ts'], params['e_ss']]])
    return 1e18*A_cc*H_eps

def H_3_nsi(A_cc,U,M, params):
    H = H_3(A_cc,U,M,params)
    H_eps = E_3(A_cc,params)

    return H + H_eps

def H_4_nsi(A_cc,A_nc,U,M, params):
    H = H_4(A_cc,A_nc,U,M,params)
    H_eps = E_4(A_cc,A_nc,params)

    return H + H_eps

def H_4(A_cc,A_nc,U, M, params=param_dict):
    A = 1e18*np.array([[A_cc, 0, 0, 0], #1702.05160 eq 8
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, A_nc]])
    U_conj = np.conj(U)
    H = U @ M @ U_conj.T + A 
    return H


def H_5(A_cc,A_nc,U,M, params=param_dict):#0709.1937 eq 5

    A = 1e18*np.array([[A_cc, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, A_nc, 0],
                        [0, 0, 0, 0, A_nc]])
    U_conj = np.conj(U)
    H = U @ M @ U_conj.T + A 
    return H



def evolve(t, state, flavor_from, flavor_to,E, U,M,ndim, theta_i, material='vac',anti=False,params=param_dict, nsi=False):
    if anti:
        sign = -1
    else:
        sign = 1
    r = get_radial_distance(t/GeV2tokm1, theta_i)
    V_cc, V_nc = V(r, material) # returns 0 if material==vac
    A_cc = sign * 2 * E * V_cc #giunti 332.
    A_nc = -sign * 2 * E * V_nc #sandhya convention
    if ndim == 3:
        if nsi:
            H = H_3_nsi(A_cc,U=U, M=M, params=params)
        else:
            H = H_3(A_cc,U=U, M=M, params=params)
        RHS = -1j/(2*E) * H
    elif ndim == 4:
        if nsi:
            H = H_4_nsi(A_cc,A_nc,U=U, M=M, params=params)
        else:
            H = H_4(A_cc,A_nc, U=U, M=M, params=params)
        RHS = -1j/(2*E) * H
    elif ndim == 2: #giunti 9.59
        H = H_2(flavor_from, flavor_to,A_cc, params=params)
        RHS = -1j/(4*E) * H 
    elif ndim == 5: #0709.1937
        H = H_5(A_cc,A_nc,U=U, M=M, params=params)
        RHS = -1j/(2*E) * H

    return RHS.dot(state)


def P_num(flavor_from, flavor_to=None,ndim = 3, E=None,L_max=None,vacuum=False,anti = False,zenith=-1, params=param_dict, eval_at=None,solver = 'RK45', material='earth', nsi=False, tols=(1e-3,1e-6)):
    '''
    Returns the numerical probabilities for all flavors as a list for L[km] and E[GeV].

    Integrates through vacuum until 'earth_start' (until 'param_max' if vacuum=True)

    Integrates through Earth at earth_start [km] if vacuum=False.

    Integrates from earth_start + 2*r_earth to param_max if vacuum = False
    '''
    theta_i = np.pi - np.arccos(zenith)
    if flavor_to is None and ndim==2:
        raise ValueError('flavor_to needs to be specified for 2 generations')
    state = np.array([0.0]*ndim, dtype=np.complex64)
    state[mass_dict[flavor_from]] = 1.0

    if ndim == 4:
        M = np.array([[0, 0, 0, 0],
                    [0, dm(2,1, params=params), 0, 0],
                    [0, 0, dm(3,1, params=params), 0],
                    [0, 0, 0, dm(4,1, params=params)]])
        U = U_4(params=params)
    elif ndim ==3:
        M = np.array([[0, 0, 0],
                      [0, dm(2,1, params=params), 0],
                      [0, 0, dm(3,1, params=params)]])
        U = U_3(params=params)
    elif ndim==5:
        M = np.array([[0, 0, 0, 0,0],
                    [0, dm(2,1, params=params), 0, 0,0],
                    [0, 0, dm(3,1, params=params), 0,0],
                    [0, 0, 0, dm(4,1, params=params),0],
                    [0, 0, 0,0,dm(5,1, params=params)]])
        U = U_5(params=params)

    if eval_at is not None: #Fix unit
        eval_at=[eval_at*GeV2tokm1]
    rtol,atol=tols # put to rtol 1e-4, atol 1e7 whenplotting for better sims (otherwise P can be slightly above unity.)  At the cost of roughly 2x the time

    if vacuum is True:
        solver = solve_ivp(fun = evolve, t_span = [0, GeV2tokm1*L_max], method=solver, y0 = state, args=(flavor_from,flavor_to, E,U,M, ndim, theta_i,'vac',anti,params,nsi))     
    else:
        solver = solve_ivp(fun = evolve, t_span = [0, GeV2tokm1*baseline(theta_i)], method=solver, y0 = state, t_eval=eval_at,rtol=rtol,atol=atol, args=(flavor_from,flavor_to, E,U,M, ndim, theta_i,material,anti,params,nsi))

    return np.abs(solver.y)**2


def P_num_over_E(flavor_from, E, flavor_to=None, L=2*r_earth, zenith=-1,earth_start = 0, ndim = 3,vacuum=False, eval_at=None,anti=False,params=param_dict, material='earth', nsi=False,tols=(1e-3,1e-6)):
    '''
    Returns the range of energies and the list of all flavour oscillation probabilities. Uses a single core
    '''
    P_list = [[]]*ndim
    for en in E:
        #print(f'Solving P{flavor_from} for {np.round(en,0)} GeV, theta = {np.round(theta_i,2)}')
        probs = P_num(flavor_from=flavor_from, flavor_to=flavor_to, E=en, L_max=L, ndim = ndim, vacuum = vacuum, anti=anti,params=params, eval_at=eval_at, zenith=zenith, material=material, nsi=nsi,tols=tols)
        for n in range(ndim):
            P_list[n] = np.append(P_list[n], probs[n][-1])
    return np.array(P_list)

def wrapper(p):
    flavor,E_range,z,anti, params, ndim, nsi = p 
    P=P_num_over_E(flavor, E=E_range, zenith=z, anti = anti, params=params, ndim=ndim, nsi=nsi)
    return P

if __name__== '__main__':
    from functions import ic_params
    params=ic_params
    print(P_num_over_E('m', E=[8e2],ndim=4, params=ic_params, zenith=-1))
