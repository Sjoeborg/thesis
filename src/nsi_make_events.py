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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s24From', default=0.01, type=float)
parser.add_argument('-s24To', default=1, type=float)
parser.add_argument('-s24N', default=10, type=int)
parser.add_argument('-emm', default=1e-1, type=float)
parser.add_argument('-emmN', default=10, type=int)
parser.add_argument('-emt', default=None, type=float)
parser.add_argument('-emtN', default=10, type=int)
parser.add_argument('-N', default = 13, type=int)
parser.add_argument('-Ndim', default=4, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    ndim = args.Ndim
    N = args.N
    alpha = 0.99
    precomputed_events = False
    gamma=0
    ic_params_nsi['dm_41'] = 0.93
    ic_params['dm_41'] = 0.93
    emm_range = np.linspace(-args.emm,args.emm,args.emmN)
    if ndim == 3:
        s24_range = np.linspace(0.,0.,1)
    else:
        s24_range= np.logspace(np.log10(args.s24From),np.log10(args.s24To),args.s24N)
    emt_range = np.linspace(-args.emt,args.emt,args.emtN)
    param_list = list_of_params_nsi(ic_params_nsi, s24_range, emm_range ,emt_range)
    for p in param_list: # Assert all dicts returned from param_list have precomputed probs.
        assert is_precomputed_nsi(N=N,ndim=ndim, dict=p,check=False)

    param_list = return_precomputed_nsi(N,ndim,params=param_list)
    emm_range = np.sort(np.unique(np.array([p['e_mm'] for p in param_list])))
    s24_range = np.sin(2*np.sort(np.unique(np.array([p['theta_24'] for p in param_list]))))**2
    emt_range = np.sort(np.unique(np.array([p['e_mt'] for p in param_list]))) 
    print(emm_range)
    print(s24_range)
    print(emt_range)



    if not precomputed_events:
        print('Computing events')
        from multiprocessing import Pool
        p = Pool()
        data = [(alpha, N,p, False,False, [False, np.median(Ereco), gamma],True, True,ndim) for p in param_list]
        H1_events_list = p.starmap(sim_events, data)
        p.close()
        if len(emt_range) > 1:
            pickle.dump(H1_events_list,open(f'./pre_computed/H1_N{N}_{len(emm_range)}x{len(s24_range)}x{len(emt_range)}_tau_nsi.p','wb'))
        else:
            pickle.dump(H1_events_list,open(f'./pre_computed/H1_N{N}_{len(emm_range)}x{len(s24_range)}_tau_nsi.p','wb'))


        H0_events = sim_events(alpha=alpha,npoints=N,params=ic_params, null=False, multi=False, spectral_shift=[False, np.median(Ereco), gamma],tau=True, nsi=False, ndim=ndim)
        pickle.dump(H0_events,open(f'./pre_computed/H0_N{N}_nsi.p','wb'))
