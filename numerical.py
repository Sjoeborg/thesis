from scipy.integrate import solve_ivp
import numpy as np
from multiprocessing import Pool
from functions import dm,theta,V_matrix, V, mass_dict,r_earth, U_nu, param_dict,GeV2tokm1,get_radial_distance, baseline,ic_params

def H_2(flavor_from, flavor_to, A=0, params=param_dict): # giunti 9.73
    i = mass_dict[flavor_from]
    j = mass_dict[flavor_to]
    m_M = dm(i+1,j+1, A, params=params)
    th_M = theta(i+1,j+1, A, params=params)
    H = np.array([ #kanske samma som U_nu(ndim=2,A)  M @ U_conj +
        [-m_M*np.cos(2*th_M), m_M*np.sin(2*th_M)],
        [m_M*np.sin(2*th_M),  m_M*np.cos(2*th_M)]])

    return H 


def H_3(A_cc, params=param_dict):
    M  = np.array([[0, 0, 0],
                   [0, dm(2,1,A=0,params=params), 0],
                   [0, 0, dm(3,1,A=0,params=params)]])
    A = 1e18*np.array([[A_cc, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]])
    U = U_nu(ndim=3, A=0,params=params)
    U_conj=np.conj(U)
    H = U @ M @ U_conj.T + A 
    return H

def H_4(A_cc,A_nc, params=param_dict):
    M  = np.array([[0, 0, 0, 0],
                   [0, dm(2,1, params=params), 0, 0],
                   [0, 0, dm(3,1, params=params), 0],
                   [0, 0, 0, dm(4,1, params=params)]])

    A = 1e18*np.array([[A_cc, 0, 0, 0], #1702.05160 eq 8
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, A_nc]])
    
    U = U_nu(ndim=4, params=params)
    U_conj = np.conj(U)
    H = U @ M @ U_conj.T + A 
    return H


def H_5(A_cc,A_nc, params=param_dict):#0709.1937 eq 5
    M  = np.array([[0, 0, 0, 0, 0],
                   [0, dm(2,1, params=params), 0, 0, 0],
                   [0, 0, dm(3,1, params=params), 0, 0],
                   [0, 0, 0, dm(4,1, params=params), 0],
                   [0, 0, 0, 0, dm(5,1, params=params)]])

    A = 1e18*np.array([[A_cc, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, A_nc, 0],
                        [0, 0, 0, 0, A_nc]])
    
    U = U_nu(ndim=5, params=params)
    U_conj = np.conj(U)
    H = U @ M @ U_conj.T + A 
    return H



def evolve(t, state, flavor_from, flavor_to,E, ndim, theta_i, material='vac',anti=False,params=param_dict):
    if anti:
        sign = -1
    else:
        sign = 1
    r = get_radial_distance(t/GeV2tokm1, theta_i)
    V_cc, V_nc = V(r, material) # returns 0 if material==vac
    A_cc = sign * 2 * E * V_cc #giunti 332.
    A_nc = -sign * 2 * E * V_nc #sandhya convention
    if ndim == 3:
        H = H_3(A_cc,params)
        RHS = -1j/(2*E) * H
    elif ndim == 4:
        H = H_4(A_cc,A_nc,params)
        RHS = -1j/(2*E) * H
    elif ndim == 2: #giunti 9.59
        H = H_2(flavor_from, flavor_to,A_cc,params)
        RHS = -1j/(4*E) * H 
    elif ndim == 5: #0709.1937
        H = H_5(A_cc,A_nc,params)
        RHS = -1j/(2*E) * H

    return RHS.dot(state)


def P_num(flavor_from, flavor_to=None,L=None,  param_min=0, param_max=2*r_earth,earth_start = 0, ndim = 3, E=None,vacuum=False,anti = False,theta_i=0, params=param_dict, eval_at=None,solver = 'RK45', material='earth'):
    '''
    Returns the numerical probabilities for all flavors as a list for L[km] and E[GeV].

    Integrates through vacuum until 'earth_start' (until 'param_max' if vacuum=True)

    Integrates through Earth at earth_start [km] if vacuum=False.

    Integrates from earth_start + 2*r_earth to param_max if vacuum = False
    '''
    if flavor_to is None and ndim==2:
        raise ValueError('flavor_to needs to be specified for 2 generations')
    state = np.array([0.0]*ndim, dtype=np.complex64)
    state[mass_dict[flavor_from]] = 1.0
    earth_end = earth_start + 2*r_earth
    x_list = np.array([])
    P_list = [[]]*ndim
    if eval_at is not None: #Fix unit
        eval_at=[eval_at*GeV2tokm1]
    if L is None:
        L = earth_start + 2*r_earth
    #rtol=1e-2 #Can use some tweaking, compare with an sol
    #atol=1e-3 #Can use some tweaking, compare with an sol
    rtol = 1e-3
    atol=1e-6
    ''' try this solver
    numPrec = 5e-4
    #implement hard coded jacobian?
    solver = integrate.ode(f).set_integrator('zvode', method='adams', order=5, with_jacobian=False,
                                                                      nsteps=1200000, atol=numPrec*2e-3, rtol=numPrec*2e-3)
                                                                      '''
    if param_min != earth_start or vacuum is True:
        if vacuum is True:
            #print(f'Solving for {np.round(E,1)} GeV in vacuum between {np.round(param_min,1)} and {np.round(param_max,1)}')
            vac_solver = solve_ivp(fun = evolve, t_span = [GeV2tokm1*param_min, GeV2tokm1*param_max], method=solver, y0 = state, args=(flavor_from,flavor_to, E, ndim, theta_i,'vac',anti,params))
        else:
            #print(f'Solving for {np.round(E,1)} GeV in vacuum between {np.round(param_min,1)} and {np.round(earth_start,1)}')
            vac_solver = solve_ivp(fun = evolve, t_span = [GeV2tokm1*param_min, GeV2tokm1*earth_start], method=solver, y0 = state, args=(flavor_from,flavor_to, E, ndim, theta_i,'vac',anti,params))

        x_list = np.concatenate((x_list,vac_solver.t/GeV2tokm1))
        for n in range(ndim):
            state[n] = np.array([vac_solver.y[n][-1]])
            P_list[n] = vac_solver.y[n]
            
    if vacuum is False:
        if param_max < earth_end:
            #print(f'Solving for {np.round(E,1)} GeV between Earth start at {np.round(earth_start,1)} to {np.round(param_max,1)}')
            earth_solver = solve_ivp(fun = evolve, t_span = [0, GeV2tokm1*param_max], method=solver, y0 = state, args=(flavor_from,flavor_to, E, ndim, theta_i,material,anti,params))
        else:
            #print(f'Solving for {np.round(E,1)} GeV between Earth start at {np.round(earth_start,1)} to {np.round(earth_end,1)}')
            earth_solver = solve_ivp(fun = evolve, t_span = [0, GeV2tokm1*earth_end], method=solver, y0 = state, t_eval=eval_at,rtol=rtol,atol=atol, args=(flavor_from,flavor_to, E, ndim, theta_i,material,anti,params))

        x_list = np.concatenate((x_list,earth_solver.t/GeV2tokm1))
        for n in range(ndim):
            state[n] = np.array([earth_solver.y[n][-1]])
            P_list[n] = np.append(P_list[n], earth_solver.y[n])

    if param_max > earth_end and vacuum is False:
        #print(f'Solving for {np.round(E,1)} GeV in vacuum between {np.round(earth_end,1)} and {np.round(param_max,1)}')
        final_solver = solve_ivp(fun = evolve, t_span = [GeV2tokm1*earth_end, GeV2tokm1*param_max], method=solver, y0 = state, args=(flavor_from,flavor_to, E, ndim, theta_i,'vac',anti,params))

        x_list = np.concatenate((x_list,final_solver.t/GeV2tokm1))
        for n in range(ndim):
            state[n] = np.array([final_solver.y[n][-1]])
            P_list[n] = np.append(P_list[n], final_solver.y[n])

    return np.abs(P_list)**2


def P_num_over_E_single(flavor_from, E, flavor_to=None, L=2*r_earth, theta_i=0,earth_start = 0, ndim = 3,vacuum=False, eval_at=2*r_earth,anti=False,params=param_dict, material='earth'):
    '''
    Returns the range of energies and the list of all flavour oscillation probabilities. Uses a single core
    '''
    P_list = [[]]*ndim
    for en in E:
        #print(f'Solving P{flavor_from} for {np.round(en,0)} GeV, theta = {np.round(theta_i,2)}')
        probs = P_num(flavor_from=flavor_from, flavor_to=flavor_to, E=en, L=L,  param_min=0, param_max=L,earth_start = earth_start, ndim = ndim, vacuum = vacuum, anti=anti,params=params, eval_at=eval_at, theta_i=theta_i, material=material)
        for n in range(ndim):
            P_list[n] = np.append(P_list[n], probs[n][-1])
    return P_list

def wrapper(p):
    flavor,E_range,z,anti, params, ndim = p 
    P=P_num_over_E_single(flavor, E=E_range, theta_i=np.pi - np.arccos(z),earth_start = 0, anti = anti, params=params, ndim=ndim)
    return P

if __name__== '__main__':
    import cProfile
    cProfile.run('P_num_over_E_single("m", [1e1,1e2,1e3])')