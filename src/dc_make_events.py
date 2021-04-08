#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys,os
sys.path.append('./src/data')
sys.path.append('./src/events')
sys.path.append('./src/probability')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from importer_gen2 import *
from processer_gen2 import *
from DC.main import get_all_events,get_events
from functions import perform_chisq, dc_params_nsi, dc_params
from gen2.event_processing import list_of_params_nsi, return_precomputed_nsi
from scipy.stats import chi2
import pickle
plt.rcParams['figure.figsize'] = [8, 4]
plt.rcParams['figure.dpi'] = 100
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
np.set_printoptions(linewidth=200)


# In[2]:


ndim = 3
pid = 1
precomputed_events = False
dc_params_nsi['dm_41'] = 0.93
dc_params['dm_41'] = 0.93
emm_range = np.linspace(0,0,1)
s24_range = np.linspace(0,0,1)
emt_range = np.linspace(-1e-2,1e-2,10)
param_list = list_of_params_nsi(dc_params_nsi,s24_range, emm_range,emt_range)

#for p in param_list: # Assert all dicts returned from param_list have precomputed probs.
#    assert is_precomputed_nsi(N=N,ndim=ndim, dict=p,check=False)


'''
param_list = return_precomputed_nsi(pid,ndim,params=param_list)
emm_range = np.sort(np.unique(np.array([p['e_mm'] for p in param_list])))
s24_range = np.sin(2*np.sort(np.unique(np.array([p['theta_24'] for p in param_list]))))**2
emt_range = np.sort(np.unique(np.array([p['e_mt'] for p in param_list])))

print(emm_range)
print(s24_range)
print(emt_range)
'''


# In[ ]:


if not precomputed_events:
    print('Computing events')
    from multiprocessing import Pool
    p = Pool()
    data = [(p, pid,False) for p in param_list]
    H1_events_list = p.starmap(get_all_events, data)
    p.close()
    pickle.dump(H1_events_list,open(f'./pre_computed/H1_DC_{pid}_{len(emm_range)}x{len(emt_range)}_tau_nsi.p','wb'))

    H0_events = get_all_events(params=param_list[0], pid=pid, null=True)
    pickle.dump(H0_events,open(f'./pre_computed/H0_DC_{pid}.p','wb'))

H1_events_list = pickle.load(open(f'./pre_computed/H1_DC_{pid}_{len(emm_range)}x{len(emt_range)}_tau_nsi.p','rb'))
H0_events = pickle.load(open(f'./pre_computed/H0_DC_{pid}.p','wb','rb'))


#H0_normalized, H1_list_normalized = normalize_events(H0_events,H1_events_list,z_bins)


# In[ ]:


'''
sigma_a = 0.25
sigma_b = 0.04
f = 0
deltachi, best_index, H1_chisq,H0_chisq = get_deltachi(H1_list_normalized_z, H0_normalized_z,emt_range,emm_range,delta_T,[sigma_a,sigma_b],f,x0=[1])
s24_cl90, s24_cl99, emm_cl90, emm_cl99 = get_contour(deltachi.T, emt_range,emm_range,2)
best_fit_coords = (param_list[best_index]['e_mt'], param_list[best_index]['e_mm'])
'''

