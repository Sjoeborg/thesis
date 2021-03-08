import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from constants import *
GF = 1.16637876 * 10e-5 * 10e-18 #eV

theta = theta(1,2)
m = dm(2,1)
def evolve(t, state,E, N_e, ndim, N_n = 0):
    A_cc = 2 * np.sqrt(2) * E * GF * (N_e - N_n/2) #giunti 332. same as adding A_nc for s!=0

    H = 1/(4*E)*np.array([[-m*np.cos(2*theta) + A_cc, m*np.sin(2*theta)],
                       [m*np.sin(2*theta), m*np.cos(2*theta) - A_cc]])

    RHS = -1j * H

    return RHS.dot(state)


def propagate(initial_state, N_e, E, Lmin, Lmax, ndim, t_eval=None):
    solver = integrate.solve_ivp(fun = evolve, t_span = [Lmin, Lmax], t_eval=t_eval, method='BDF', y0 = initial_state, args=(E, N_e, ndim))
    return solver.t, np.abs(solver.y)**2


def analytical(E, Lmin, Lmax, N_e,L=None):
    A_cc = 2 * np.sqrt(2) * E * GF * N_e
    if L is None:
        L = np.linspace(Lmin, Lmax, 500)
    LE = L/E

    m_M = np.sqrt(
        (m * np.cos(2*theta) - A_cc)**2 + (m * np.sin(2*theta))**2
    )
    cos_M = (m * np.cos(2*theta) - A_cc) / m_M
    sin_M = (m * np.sin(2*theta)) / m_M

    return L, sin_M ** 2*np.sin(m_M*LE/4)**2

if __name__== '__main__':
    #N_e =6e11 # 1202.1024
    E = 12500e-3
    L = 12500e3
    N_e = 6e13
    state = np.array([1, 0], dtype=np.complex64)
    ndim = state.shape[0]
    
    t_vac, P_num_vac = propagate(state,N_e = 0, E = E, Lmin=0, Lmax = L, ndim = ndim)

    t_vac, P_num_M = propagate(state,N_e = N_e, E = E, Lmin=0, Lmax = L, t_eval=t_vac, ndim = ndim)

    t_vac, P_anal_mat = analytical(E=E, Lmin=0, Lmax=L, N_e = N_e, L = t_vac)
    t_vac, P_anal_vac = analytical(E=E, Lmin=0, Lmax=L, N_e = 0, L = t_vac)

    plt.subplot(2, 1, 1)
    plt.scatter(t_vac, P_num_vac[1], color='blue', s = 7)
    plt.plot(t_vac, P_anal_vac, label='em num vac', color='blue')
    plt.title('Vacuum')

    

    plt.subplot(2, 1, 2)
    plt.scatter(t_vac, P_num_M[1], color='black', s= 7)
    plt.plot(t_vac, P_anal_mat, label='em num mat', color='black')
    plt.title(f'Matter')
    plt.show()