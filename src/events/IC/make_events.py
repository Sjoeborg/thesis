import sys,os
if __name__ == '__main__':
    #os.chdir('../')
    sys.path.append('./src/data')
    sys.path.append('./src/events')
    sys.path.append('./src/probability')
import numpy as np
import pandas as pd
from IC.importer import *
from IC.processer import *
from IC.event_processing import *
from DC.event_processing import list_of_params_nsi
from IC.main import sim_events
from functions import nufit_params_nsi, nufit_params_nsi_IO
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-dm31From', default=2.489, type=float)
parser.add_argument('-dm31To', default=2.543, type=float)
parser.add_argument('-dm31N', default=10, type=int)

parser.add_argument('-th23From', default=48*np.pi/180, type=float)
parser.add_argument('-th23To', default=50.1*np.pi/180, type=float)
parser.add_argument('-th23N', default=10, type=int)

parser.add_argument('-emm', default=1e-1, type=float)
parser.add_argument('-emmN', default=10, type=int)
parser.add_argument('-emt', default=1e-2, type=float)
parser.add_argument('-emtN', default=10, type=int)
parser.add_argument('-s', default = 0, type=int)
parser.add_argument('-sT', default = 1, type=int)
parser.add_argument('-nonsi', action='store_false')
parser.add_argument('-IO', action='store_true')
parser.add_argument('-N', default=13, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    alpha = 0.99
    gamma=0
    if args.IO is False:
        dm31_range = np.linspace(args.dm31From*1e-3,args.dm31To*1e-3,args.dm31N-1)
    else: #Nufit gives dm32 for IO, so convert the values and bounds to dm31 explicitly here
        dm31_range = np.linspace(-2.498e-3+7.42e-5 + 0.028e-3 +0.21e-5,-2.498e-3+7.42e-5 - 0.028e-3 -0.2e-5,args.dm31N-1)
    th23_range = np.linspace(args.th23From,args.th23To,args.th23N-1)
    emm_range = np.linspace(-args.emm,args.emm,args.emmN-1)
    emt_range = np.linspace(-args.emt,args.emt,args.emtN-1)

    
    if args.IO:
        nsi_params = nufit_params_nsi_IO.copy()
    else:
        nsi_params = nufit_params_nsi.copy()

    # We want to marg over best-fit points too, so insert them
    if nsi_params['dm_31'] not in dm31_range:
        dm31_range = np.sort(np.append(dm31_range,nsi_params['dm_31']))
    if nsi_params['theta_23'] not in th23_range:
        th23_range = np.sort(np.append(th23_range, nsi_params['theta_23']))
    if nsi_params['e_mm'] not in emm_range:
        emm_range = np.sort(np.append(emm_range, nsi_params['e_mm']))
    if nsi_params['e_mt'] not in emt_range:
        emt_range = np.sort(np.append(emt_range, nsi_params['e_mt']))

    param_list = list_of_params_nsi(nsi_params, dm31_range, th23_range,emm_range, emt_range)

    emm_range = np.sort(np.unique(np.array([p['e_mm'] for p in param_list])))
    emt_range = np.sort(np.unique(np.array([p['e_mt'] for p in param_list]))) 
    if args.IO is False:
        print(f'Precomputing NO DC events for dm31({dm31_range.min()},{dm31_range.max()},{len(dm31_range)}),',
                f'th23({np.round(th23_range.min(),2)},{np.round(th23_range.max(),2)},{len(th23_range)}), emm({emm_range.min()},',
                f'{emm_range.max()},{len(emm_range)}), emt({emt_range.min()},{emt_range.max()},{len(emt_range)})')
    else:
        print('WARNING: Hard-coded values for dm31 under IO! Input from -dm31From/To ignored')
        print(f'Precomputing IO DC events for dm31({dm31_range.min()},{dm31_range.max()},{len(dm31_range)}),',
                f'th23({th23_range.min()},{th23_range.max()},{len(th23_range)}), emm({emm_range.min()},',
                f'{emm_range.max()},{len(emm_range)}), emt({emt_range.min()},{emt_range.max()},{len(emt_range)})')


    from multiprocessing import Pool
    p = Pool()
    data = [(alpha, args.N,p, False,False, [False, np.median(Ereco), gamma],True, True,3) for p in param_list]
    H1_events_list = p.starmap(sim_events, data)
    p.close()
    pickle.dump(H1_events_list,open(f'./pre_computed/H1_IC_N{args.N}_{len(dm31_range)}x{len(th23_range)}x{len(emm_range)}x{len(emt_range)}.p','wb'))


    H0_events = sim_events(alpha=alpha,npoints=args.N,params=ic_params, null=False, multi=False, spectral_shift=[False, np.median(Ereco), gamma],tau=True, nsi=False, ndim=3)
    pickle.dump(H0_events,open(f'./pre_computed/H0_IC_N{args.N}_nsi.p','wb'))