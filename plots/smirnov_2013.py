# Plots for Smirnov 2013
from main import *
if __name__ == '__main__':
    E_range = np.linspace(0.1e3,7e3,100)
    sin_34 = [0,0.1,0.2,0.3,0.5]
    th_34 = np.arcsin(np.sqrt(sin_34))/2
    param_dict.update({'dm_21':7.4e-5,
                       'dm_31':2.4e-3,
                       'dm_41':1,
                       'theta_12': 33.36 * np.pi/180,
                       'theta_23': 45 * np.pi/180,
                       'theta_13': 8.66 * np.pi/180,
                       'delta_ij': 300 * np.pi/180,
                       'theta_12': 33.36 * np.pi/180,
                       'theta_24': np.arcsin(np.sqrt(0.04))/2})
    dicts = [dict(param_dict,theta_34=t_34) for t_34 in th_34]
    theta_1 = np.pi - np.arccos(-0.8)
    theta_2 = np.pi - np.arccos(-1)
    x_1,P_1 = P_over_E_parameter('local','m',dicts,E_range,ndim=4,anti=True, theta_i = theta_1)
    x_2,P_2 = P_over_E_parameter('local','m',dicts,E_range,ndim=4,anti=True, theta_i = theta_2)
    fig, ax = plt.subplots(2,1,sharex=True,sharey=True)
    ax= ax.flatten()
    plot_P_E_params(x_1,P_1, ax[0],'m',legend_name='s34', legend_values= sin_34,ylabel=r'$P_{\bar{\mu}\bar{\mu}}$', xlabel='E (TeV)', title=r'$P_{\bar{\mu}\bar{\mu}}\,, \cos{\theta_z} = -0.8$')

    plot_P_E_params(x_2,P_2, ax[1],'m',legend_name='s34', legend_values= sin_34,ylabel=r'$P_{\bar{\mu}\bar{\mu}}$', xlabel='E (TeV)', title=r'$P_{\bar{\mu}\bar{\mu}}\,, \cos{\theta_z} = -1$')
    plt.show()