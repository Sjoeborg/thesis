import numpy as np
import matplotlib.pyplot as plt
from numerical import P_num, P_num_over_E
from analytical import P_an
from functions import ic_params,r_earth, mass_dict
import matplotlib
from multiprocessing import Pool
from DC.main import get_interpolators_DC, get_flux
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

    args = [(flavor_from, E, None, 2*r_earth, zenith, 0,ndim,False, None, anti, p,'earth',nsi, tols) for p in param_dict_list]
    p = Pool()
    #res = []
    #for p in param_dict_list:
    #    res.append(P_num_over_E_wrapper(p))
    res = p.starmap(P_num_over_E, args)

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

def wrap(flavor_from, E, zenith, ndim, anti,params,nsi):
    return P_num_over_E(flavor_from=flavor_from, E=E, zenith=zenith, ndim = ndim,anti=anti,params=params, nsi=nsi)

def _oscillogram(p_list):
    '''
    p = [flavor_from, E, zenith, ndim, anti, params]
    '''
    p = Pool() 
    res = p.starmap(wrap, p_list)
    p.close()
    Pxm = np.array(res)[:,1,:]
    Pxe = np.array(res)[:,0,:]
    Pxt = np.array(res)[:,2,:]
    return Pxe,Pxm,Pxt

def oscillogram(E_range, z_range, params):
    '''
    Returns (Pex, Pmx, Paeax, Pamax)
    '''
    lista_mbar = [('m', E_range, z, 3, True, params,False) for z in z_range]
    lista_ebar = [('e', E_range, z, 3, True, params,False) for z in z_range]
    lista_m = [('m', E_range, z, 3, False, params,False) for z in z_range]
    lista_e = [('e', E_range, z, 3, False, params,False) for z in z_range]

    Pmx = _oscillogram(lista_m)
    Pex = _oscillogram(lista_e)
    Pamax = _oscillogram(lista_mbar)
    Paeax = _oscillogram(lista_ebar)

    return Pex, Pmx, Paeax, Pamax
def nsi_oscillogram(E_range, z_range, params):
    '''
    Returns (Pex_nsi, Pmx_nsi, Paeax_nsi, Pamax_nsi)
    '''
    lista_mbar_nsi = [('m', E_range, z, 3, True, params,True) for z in z_range]
    lista_ebar_nsi = [('e', E_range, z, 3, True, params,True) for z in z_range]
    lista_m_nsi = [('m', E_range, z, 3, False, params,True) for z in z_range]
    lista_e_nsi = [('e', E_range, z, 3, False, params,True) for z in z_range]

    

    Pmx_nsi = _oscillogram(lista_m_nsi)
    Pex_nsi = _oscillogram(lista_e_nsi)
    Pamax_nsi = _oscillogram(lista_mbar_nsi)
    Paeax_nsi = _oscillogram(lista_ebar_nsi)

    return Pex_nsi, Pmx_nsi, Paeax_nsi, Pamax_nsi


def flux_oscillogram(E_range, z_range, params, nsi=False):
    '''
    Returns 1-flux_final/flux_initial for Pxm
    '''
    lista_mbar = [('m', E_range, z, 3, True, params,False) for z in z_range]
    lista_ebar = [('e', E_range, z, 3, True, params,False) for z in z_range]
    lista_m = [('m', E_range, z, 3, False, params,False) for z in z_range]
    lista_e = [('e', E_range, z, 3, False, params,False) for z in z_range]
    Pmm = _oscillogram(lista_m)
    Pem = _oscillogram(lista_e)
    Pamam = _oscillogram(lista_mbar)
    Paeam = _oscillogram(lista_ebar)

    
    interp_flux, _ = get_interpolators_DC()
    E_mesh, z_mesh = np.meshgrid(E_range, z_range)
    flux_m = get_flux('m',E_mesh,z_mesh,interp_flux)
    flux_e = get_flux('e',E_mesh,z_mesh,interp_flux)
    flux_mbar = get_flux('mbar',E_mesh,z_mesh,interp_flux)
    flux_ebar = get_flux('ebar',E_mesh,z_mesh,interp_flux)

    flux_initial = flux_mbar + flux_ebar + flux_m + flux_e
    flux_final = flux_mbar*Pamam + flux_ebar*Paeam + flux_m*Pmm + flux_e*Pem

    return 1-flux_final/flux_initial


def nsi_flux_oscillogram(E_range, z_range, params, only_m=True):
    '''
    Returns (flux_final/flux_initial, flux_bar_final/flux_bar_initial, flux_both_final/flux_both_initial)
    for either only Pxm flux (only_m is True), or for only Pxe+Pxt flux (only_m is False)
    '''
    lista_mbar_nsi = [('m', E_range, z, 3, True, params,True) for z in z_range]
    lista_ebar_nsi = [('e', E_range, z, 3, True, params,True) for z in z_range]
    lista_m_nsi = [('m', E_range, z, 3, False, params,True) for z in z_range]
    lista_e_nsi = [('e', E_range, z, 3, False, params,True) for z in z_range]

    lista_mbar = [('m', E_range, z, 3, True, params,False) for z in z_range]
    lista_ebar = [('e', E_range, z, 3, True, params,False) for z in z_range]
    lista_m = [('m', E_range, z, 3, False, params,False) for z in z_range]
    lista_e = [('e', E_range, z, 3, False, params,False) for z in z_range]

    Pmx_nsi = _oscillogram(lista_m_nsi)
    Pex_nsi = _oscillogram(lista_e_nsi)
    Pamax_nsi = _oscillogram(lista_mbar_nsi)
    Paeax_nsi = _oscillogram(lista_ebar_nsi)

    Pmx = _oscillogram(lista_m)
    Pex = _oscillogram(lista_e)
    Pamax = _oscillogram(lista_mbar)
    Paeax = _oscillogram(lista_ebar)
    
    interp_flux, _ = get_interpolators_DC()
    E_mesh, z_mesh = np.meshgrid(E_range, z_range)
    flux_m = get_flux('m',E_mesh,z_mesh,interp_flux)
    flux_e = get_flux('e',E_mesh,z_mesh,interp_flux)
    flux_mbar = get_flux('mbar',E_mesh,z_mesh,interp_flux)
    flux_ebar = get_flux('ebar',E_mesh,z_mesh,interp_flux)

    if only_m:
        flux_final =  flux_m*Pmx[1] + flux_e*Pex[1]
        flux_final_nsi = flux_m*Pmx_nsi[1] + flux_e*Pex_nsi[1]

        flux_final_bar =  flux_mbar*Pamax[1] + flux_ebar*Paeax[1]
        flux_final_nsi_bar = flux_mbar*Pamax_nsi[1] + flux_ebar*Paeax_nsi[1]
    
    else:
        flux_final =  flux_m*(Pmx[0] + Pmx[2]) + flux_e*(Pex[0] + Pex[2])
        flux_final_nsi = flux_m*(Pmx_nsi[0] + Pmx_nsi[2]) + flux_e*(Pex_nsi[0] + Pex_nsi[2])

        flux_final_bar =  flux_mbar*(Pamax[0] + Pamax[2]) + flux_ebar*(Paeax[0] + Paeax[2])
        flux_final_nsi_bar = flux_mbar*(Pamax_nsi[0] + Pamax_nsi[2]) + flux_ebar*(Paeax_nsi[0] + Paeax_nsi[2])

    return flux_final_nsi/flux_final, flux_final_nsi_bar/flux_final_bar, (flux_final_nsi + flux_final_nsi_bar)/(flux_final+ flux_final_bar)



if __name__ == '__main__':
    pass