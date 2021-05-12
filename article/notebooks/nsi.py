import sys,os
sys.path.append('./src')
sys.path.append('./src/data')
sys.path.append('./src/events')
sys.path.append('./src/probability')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from IC.importer import *
from IC.processer import *
from functions import nufit_params_nsi
from plotter import P_over_E_parameter, nsi_oscillogram, nsi_flux_oscillogram,save, savethesis
from IC.event_processing import *
import pickle
#from matplotlib.colors import CenteredNorm
plt.rcParams['figure.figsize'] = [8, 4]
plt.rcParams['figure.dpi'] = 100
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
np.set_printoptions(linewidth=200)

IC_range = np.logspace(np.log10(500),4,50)
DC_range = np.logspace(np.log10(5.6),np.log10(56),50)
#ett_range = [-0.1,0,0.1]#np.linspace(-1e-1,1e-1,3)
emm_range = [-1e-1,0,1e-1]#np.linspace(-1e-1,1e-1,3)
emt_range = [-5e-2,0,5e-2]
params = nufit_params_nsi.copy()
anti=True
s24_range = [0.032]
params.update({'theta_24': np.arcsin(np.sqrt(0.032))/2, 'dm_41':0.93, 'theta_34': np.arcsin(np.sqrt(0))/2})
emm_params = list_of_params_nsi(params, s24_range, emm_range)
emt_params = list_of_params_nsi(params, s24_range, [0],emt_range)
both_params = list_of_params_nsi(params, s24_range, emm_range,emt_range)
zenith = -1

z_range = np.linspace(-1,0,500)
all_range = np.logspace(np.log10(1),5.1,500)
#P_emm_IC = np.array(nsi_oscillogram(IC_range, z_range, emm_params[0]))
P_emt_IC_track= np.array(nsi_flux_oscillogram(all_range, z_range, emt_params[0], only_m=True))
P_emt_IC_cascade= np.array(nsi_flux_oscillogram(all_range, z_range, emt_params[0], only_m=False))
pickle.dump(P_emt_IC_track, open('P_emt_IC_track.p','wb'))
pickle.dump(P_emt_IC_cascade, open('P_emt_IC_cascade.p','wb'))
'''
fig, ax = plt.subplots(1,2, figsize=(6*2,7), sharex=True, squeeze=True)

c1=ax[0].pcolormesh(z_range, all_range,np.log10(P_emt_IC_cascade[2].T),cmap='RdBu',edgecolor=None, shading = 'gouraud', norm = CenteredNorm(0))# vmin=-1,vmax=3)#)
c2=ax[1].pcolormesh(z_range, all_range,np.log10(P_emt_IC_track[2].T),cmap='RdBu',edgecolor=None, shading = 'gouraud',norm = CenteredNorm(0))# vmin=-1,vmax=3)# 

ax[0].set_title(r'$\nu_{e/\tau} + \bar\nu_{e/\tau}$ flux ratio', fontsize=20)
ax[1].set_title(r'$\nu_\mu + \bar\nu_\mu$ flux ratio', fontsize=20)
ax[0].set_xlabel(r'$\cos{(\theta^{true}_z)}$', fontsize=20)
ax[1].set_xlabel(r'$\cos{(\theta^{true}_z)}$', fontsize=20)
ax[0].tick_params(labelsize=16, direction='in', which='both', axis='both')
ax[1].tick_params(labelsize=16, direction='in', which='both', axis='both')
ax[0].set_ylabel(r'$E^{true}\,\,\mathrm{[GeV]}$', fontsize=20)
ax[0].set_yscale('log')
ax[1].set_yscale('log')

cbar2 = fig.colorbar(c2, ax=ax[1])
cbar2.set_label(r'$\log{(\phi_\text{NSI}^\text{det}/\phi_\text{SI}^\text{det})}$', fontsize=20)

ax[0].text(-0.30,2.2,'99\% DC cascades',rotation=0,fontsize=11,alpha = 0.4)
ax[0].axhline(2, color='black', alpha=0.3,ls=':', lw=2) # MC2018_DC().query('pid==0').query('abs(pdg) != 14').query('reco_energy == reco_energy.min()')['true_energy'].quantile(0.01)
ax[0].axhline(119, color='black', alpha=0.3,ls=':', lw=2) # MC2018_DC().query('pid==0').query('abs(pdg) != 14').query('reco_energy == reco_energy.max()')['true_energy'].quantile(1-0.01)

ax[1].text(-0.32,2.5,'99\% DC tracks',rotation=0,fontsize=11,alpha = 0.4)
ax[1].axhline(2.17, color='black', alpha=0.3,ls=':', lw=2) # MC2018_DC().query('pid==1').query('abs(pdg) == 14').query('reco_energy == reco_energy.min()')['true_energy'].quantile(0.01)
ax[1].axhline(149, color='black', alpha=0.3,ls=':', lw=2) # MC2018_DC().query('pid==1').query('abs(pdg) == 14').query('reco_energy == reco_energy.max()')['true_energy'].quantile(1-0.01)


ax[1].text(-0.30,350,'99\% IC tracks',rotation=0, fontsize=11, alpha = 0.4)
ax[1].axhline(310.745, color='black', alpha=0.3, ls='--')#from get_Etrue
ax[1].axhline(95043.804, color='black', alpha=0.3, ls='--') #from get_Etrue
save(fig,'flux_ratio')
'''