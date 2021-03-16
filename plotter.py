import numpy as np
import matplotlib.pyplot as plt
from numerical import P_num, P_num_over_E
from analytical import P_an
from functions import ic_params,r_earth, mass_dict
import matplotlib
from multiprocessing import Pool
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
np.set_printoptions(precision=3)

def an_plots(flavor_from_list=['e','m','t'], flavor_to_list=['e','m','t'], param='L', material='vac',E=None, L=None,  param_min=None, param_max=None, earth_start = 0, ndim = 3, anti=False):
    styles = ['solid','dotted','dashed']
    colors = ['blue','red','green']
    ncols= len(flavor_from_list)
    nrows = len(flavor_to_list)
    fig, ax = plt.subplots(nrows,ncols, sharex=True, sharey=True)
    for col,flavor_from in enumerate(flavor_from_list):
        for row,flavor_to in enumerate(flavor_to_list):
            x_an,y_an = P_an(flavor_from, flavor_to, param, E, L,  param_min, param_max, material,earth_start, ndim)
            ax[col,row].plot(x_an,y_an, linestyle=styles[0])
            ax[col,row].set_title(f'{flavor_from} to {flavor_to}')
    if param == 'E':
        plt.suptitle(rf'Vacuum oscillations in Earth core, $E = \in [{param_min},{param_max}]$ GeV, $L = {L}$ km')
    elif param == 'L':
        plt.suptitle(rf'Vacuum oscillations in Earth core, $E = {E}$ GeV, $L \in [{param_min},{param_max}]$ km')
    plt.tight_layout()
    plt.show()

def compare_an_nu(flavor_from_list, flavor_to_list, param, material,E=None, L=None,  param_min=None, param_max=None, earth_start = 0, ndim = 3, anti=False):
    styles = ['solid','dotted','dashed']
    colors = ['blue','red','green']
    ncols= len(flavor_from_list)
    nrows = len(flavor_to_list)
    fig, ax = plt.subplots(nrows,ncols, sharex=True, sharey=True)
    for col,flavor_from in enumerate(flavor_from_list):
        for row,flavor_to in enumerate(flavor_to_list):
            x_nu,y_nu = P_num(earth_start=earth_start, E=E, param_min=param_min, param_max=param_max, ndim=ndim, flavor_from=flavor_from, flavor_to=flavor_to)
            x_an,y_an = P_an(flavor_from, flavor_to, param, E, L,  param_min, param_max, material,earth_start, ndim)
            ax[col,row].plot(x_an,y_an, linestyle=styles[0])
            ax[col,row].plot(x_nu,y_nu[row], linestyle=styles[1], c='black')
            ax[col,row].set_title(f'{flavor_from} to {flavor_to}')
    if param == 'E':
        plt.suptitle(rf'Vacuum oscillations in Earth core, $E = \in [{param_min},{param_max}]$ GeV, $L = {L}$ km')
    elif param == 'L':
        plt.suptitle(rf'Vacuum oscillations in Earth core, $E = {E}$ GeV, $L \in [{param_min},{param_max}]$ km')
    plt.tight_layout()
    plt.show()





def P_over_E_parameter(flavor_from, param_dict_list, E, zenith = -1, ndim = 3, anti=False, nsi=False, tols=(1e-4,1e-7)):
    '''
    Returns the range of energies and the list of all flavour oscillation probabilities. Uses all cores locally or on techila (type='local'/'cloud')
    '''
    global P_num_over_E_wrapper # Needed for wrapper to work with p.map
    def P_num_over_E_wrapper(p):
        return P_num_over_E(flavor_from=flavor_from, E=E, ndim = ndim,params=p, anti=anti, zenith=zenith,nsi=nsi, tols=tols)
    #p = Pool()
    res = []
    for p in param_dict_list:
        res.append(P_num_over_E_wrapper(p))
    #res = map(P_num_over_E_wrapper, param_dict_list)

    P_list = []
    for i in range(len(param_dict_list)): # Splits result list into x and y
            P_list.append(res[i])
    return np.array(P_list)


def plot_P_E_params(x,P, ax,flavor_to='m',colors=None,legend_name='', legend_values='', ylabel='', xlabel='',title=''):
    if colors is None:
        colors = ['black','blue','red','green','brown']
    beta = mass_dict[flavor_to]
    for i in range(len(legend_values)):
        ax.plot(x/1e3, P[i][beta],label=f'{legend_name} = {np.round(legend_values[i],2)}', color=colors[i])
        ax.tick_params(axis='y', which='major', length = 5, direction='in',right=True)
        ax.tick_params(axis='y', which='minor', length = 2, direction='in',right=True)
    ax.set_xlabel(f'{xlabel}')
    ax.set_ylabel(f'{ylabel}')
    ax.set_xlim((np.min(x[0]/1e3),np.max(x[0]/1e3)))
    ax.set_xscale('log')
    ax.set_ylim((0,1.1))
    ax.set_title(f'{title}')
    ax.legend()

def wrap(z):
    return P_num_over_E(flavor_from='m', E=np.logspace(3,4,50), zenith=z, ndim = 4,anti=False,params=ic_params)
def oscillogram(z):
    p = Pool() 
    res = p.map(wrap, z)
    Pmm = np.array(res)[:,1,:]
    p.close()
    return Pmm.T



if __name__ == '__main__':
    pass