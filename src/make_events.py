import sys,os
#os.chdir('../')
sys.path.append('./src/data')
sys.path.append('./src/events')
sys.path.append('./src/probability')
import numpy as np
import pandas as pd
from importer import *
from processer import *
from event_processing import *
from events.main import sim_events, list_of_params, ic_params
import pickle


ndim = 4
N = 13
alpha = 0.99
precomputed_events = False
gamma=0
dm41_range = np.logspace(-1,1,20)
s24_range = np.logspace(-2,np.log10(0.2),20)
#s34_range = np.logspace(-2,0,10)
s34_range = s24_range
param_list = list_of_params(ic_params, dm41_range, s24_range,s24_eq_s34=True)
for p in param_list: # Assert all dicts returned from param_list have precomputed probs.
    assert is_precomputed(N=N,ndim=ndim, dict=p,check=False)

param_list = return_precomputed(N,ndim,params=param_list)
dm41_range = np.sort(np.unique(np.array([p['dm_41'] for p in param_list])))
s24_range = np.sin(2*np.sort(np.unique(np.array([p['theta_24'] for p in param_list]))))**2
s34_range = np.sin(2*np.sort(np.unique(np.array([p['theta_34'] for p in param_list]))))**2
print(dm41_range)
print(s24_range)
print(s34_range)


if not precomputed_events:
    print('Computing events')
    from multiprocessing import Pool
    p = Pool()
    data = [(alpha, N,p, False,False, [False, np.median(Ereco), gamma],True) for p in param_list]
    H1_events_list = p.starmap(sim_events, data)
    p.close()
    if len(s34_range) > 1:
        pickle.dump(H1_events_list,open(f'./pre_computed/H1_34_N{N}_{len(dm41_range)}x{len(s24_range)}x{len(s34_range)}_tau_all.p','wb'))
    else:
        pickle.dump(H1_events_list,open(f'./pre_computed/H1_no34_N{N}_{len(dm41_range)}x{len(s24_range)}_tau_all.p','wb'))


    H0_events = sim_events(alpha=alpha,npoints=N,params=param_list[0], null=True, multi=False, spectral_shift=[False, np.median(Ereco), gamma],tau=True)
    pickle.dump(H0_events,open(f'./pre_computed/H0_N{N}_all.p','wb'))