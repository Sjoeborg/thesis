# File for plots in 0509197
from main import *

def fig1():
    #Fig 1 in sandhyas paper
    fig, ax = plt.subplots(3,2,sharex=True,sharey=True)
    ax= ax.flatten()
    for i,E in enumerate(range(2,8)):
        x_mat,y_mat=P_num(flavor_from='m', flavor_to='m', E=E, L=None,  param_min=0, param_max=L,earth_start = 0, ndim = 3)
        x_vac,y_vac=P_num(flavor_from='m', flavor_to='m', E=E, L=None,  param_min=0, param_max=L,earth_start = 0, ndim = 3, vacuum=True)
        ax[i].plot(x_mat,y_mat[1], label='mm mat', color='black')
        ax[i].plot(x_vac,y_vac[1], label='mm vac', color='red', linestyle='dotted')
        ax[i].set_xlim((2000, 12000))
        ax[i].set_ylim((0, 1))
        ax[i].set_title(f'E={E} GeV')
    plt.suptitle(r'3 gen matter and vacuum $P_{\mu\mu}$ as a function of L with fixed E.')
    plt.show()


def fig2():
    #Fig 2 in sandhyas paper    
    L_range_Eres = np.linspace(100, 10000, 100)
    E_baseline = []
    E_res_all=[]
    for ell in L_range_Eres:
        Er = []
        for th in theta_i:
            r = get_radial_distance(ell,th)
            if np.round(r,5) <= r_earth:
                Eres = 1e-18 * 2e-3 * (1-2*0.1) / (2*V(r)[0])
                Er.append(Eres)
            else:
                #Er.append(0)
                pass
        E_baseline.append(np.mean(Er))
    E_spmin1 = [2/np.pi * 1.27 * 2e-3*x for x in L_range_Eres]
    E_spmax = [1/np.pi * 1.27 * 2e-3*x for x in L_range_Eres]
    E_spmin2 = [2/(3*np.pi) * 1.27 * 2e-3*x for x in L_range_Eres]
    
    plt.plot(L_range_Eres, E_baseline, label=r'$E_{res,avg}$')
    plt.plot(L_range_Eres, E_spmin1, label=r'$E_{SPMIN1}$')
    plt.plot(L_range_Eres, E_spmax, label=r'$E_{SPMAX}$')
    plt.plot(L_range_Eres, E_spmin2, label=r'$E_{SPMIN2}$')
    plt.ylabel('E [GeV]') 
    plt.legend()
    plt.xscale('log')
    
    plt.show()


def fig3():
    #Fig 3 sandhya
    fig, ax = plt.subplots(3,2, sharex=True,sharey=True)
    ax = ax.flatten()
    lengths = [1000, 7000, 3000, 9000, 5000, 11000]
    for i in range(6):
        x,y = P_num_over_E_single(flavor_from='m', flavor_to='m', L=lengths[i],  param_min=1, param_max=11,earth_start = 0, ndim = 3, npoints=100, cores=8)
        ax[i].plot(x,y[1], color='black')

        ax[i].annotate(f'L = {lengths[i]} km', xy=(7,0.5))

        x,y = P_num_over_E_single(flavor_from='m', flavor_to='m', L=lengths[i],  param_min=1, param_max=11,earth_start = lengths[i], ndim = 3, npoints=100, cores=8, vacuum=True)
        ax[i].plot(x,y[1], color='red',linestyle='dotted')
    plt.suptitle(r'$P_{\mu\mu}$ vs E for 3 generations for different baselines')
    plt.xlabel('E (GeV)')
    plt.show()


if __name__ == '__main__':    
    E = 5 #GeV
    L = 2*r_earth #km
    L_range = np.linspace(0, 2*r_earth, 30) #km
    
    npoints = 2
    theta_i = np.linspace(-np.pi/2,np.pi/2,50)
    #theta_i = np.linspace(0,np.pi/2,30)

    fig1()
    