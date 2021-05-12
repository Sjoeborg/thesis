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


'''
z_range = np.linspace(-1,0,500)
IC_range = np.logspace(np.log10(500),4,5)
all_range = np.logspace(np.log10(1),5.1,500)
#P_emm_IC = np.array(nsi_oscillogram(IC_range, z_range, emm_params[0]))
#P_emt_IC_track= np.array(nsi_flux_oscillogram(all_range, z_range, emt_params[0], only_m=True))
#P_emt_IC_cascade= np.array(nsi_flux_oscillogram(all_range, z_range, emt_params[0], only_m=False))
P_emt_IC_track = pickle.load(open('./pre_computed/P_emt_IC_track.p','rb'))
P_emt_IC_cascade = pickle.load(open('./pre_computed/P_emt_IC_cascade.p','rb'))
pickle.dump(P_emt_IC_track, open('P_emt_IC_track.p','wb'))
pickle.dump(P_emt_IC_cascade, open('P_emt_IC_cascade.p','wb'))
'''

z_range = np.linspace(-1,0,500)
E_range = np.logspace(0,2,500)

Pex_neg, Pmx_neg, Paeax_neg,Pamax_neg = nsi_oscillogram(E_range=E_range, z_range=z_range, params= emt_params[0])
Pex_pos, Pmx_pos, Paeax_pos,Pamax_pos = nsi_oscillogram(E_range=E_range, z_range=z_range, params= emt_params[2])


pickle.dump(Pmx_neg, open('Pmx_neg.p','wb'))
pickle.dump(Pmx_pos, open('Pmx_pos.p','wb'))
