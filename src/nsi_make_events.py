import sys,os
if __name__ == '__main__':
    #os.chdir('../')
    sys.path.append('./src/data')
    sys.path.append('./src/events')
    sys.path.append('./src/probability')
import numpy as np
import pandas as pd
from importer import *
from processer import *
from event_processing import *
from events.main import sim_events
from probability.functions import ic_params_nsi, ic_params
import pickle


ndim = 4
N = 13
alpha = 0.99
precomputed_events = False
gamma=0
ic_params_nsi['dm_41'] = 0.93
ic_params['dm_41'] = 0.93
emm_range = np.logspace(-1e-2,1e-2,20)
s24_range = np.logspace(-2,np.log10(0.2),20)
emt_range = np.logspace(-1e-2,1e-2,20)
param_list = list_of_params_nsi(ic_params_nsi, emm_range, s24_range, emt_range)
for p in param_list: # Assert all dicts returned from param_list have precomputed probs.
    assert is_precomputed_nsi(N=N,ndim=ndim, dict=p,check=False)

param_list = return_precomputed_nsi(N,ndim,params=param_list)
emm_range = np.sort(np.unique(np.array([p['em_mm'] for p in param_list])))
s24_range = np.sin(2*np.sort(np.unique(np.array([p['theta_24'] for p in param_list]))))**2
emt_range = np.sin(2*np.sort(np.unique(np.array([p['e_mt'] for p in param_list]))))**2
print(emm_range)
print(s24_range)
print(emt_range)


if not precomputed_events:
    print('Computing events')
    from multiprocessing import Pool
    p = Pool()
    data = [(alpha, N,p, False,False, [False, np.median(Ereco), gamma],True, True) for p in param_list]
    H1_events_list = p.starmap(sim_events, data)
    p.close()
    if len(emt_range) > 1:
        pickle.dump(H1_events_list,open(f'./pre_computed/H1_34_N{N}_{len(emm_range)}x{len(s24_range)}x{len(emt_range)}_tau_nsi.p','wb'))
    else:
        pickle.dump(H1_events_list,open(f'./pre_computed/H1_no34_N{N}_{len(emm_range)}x{len(s24_range)}_tau_nsi.p','wb'))


    H0_events = sim_events(alpha=alpha,npoints=N,params=ic_params, null=False, multi=False, spectral_shift=[False, np.median(Ereco), gamma],tau=True, nsi=False)
    pickle.dump(H0_events,open(f'./pre_computed/H0_N{N}_nsi.p','wb'))