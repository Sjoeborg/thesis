import sys,os
if __name__ == '__main__':
    #os.chdir('../')
    sys.path.append('./src/data')
    sys.path.append('./src/events')
    sys.path.append('./src/probability')
import numpy as np
import pandas as pd
from DC.importer import *
from DC.processer import *
from DC.event_processing import *
from DC.main import get_all_events
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
args = parser.parse_args()

if __name__ == '__main__':
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

    print(dm31_range)
    print(th23_range)
    print(emm_range)
    print(emt_range)

    for pid in [1,0]:
        from multiprocessing import Pool
        p = Pool()
        data = [(p,pid,True) for p in param_list]
        H1_events_list = p.starmap(get_all_events, data)
        p.close()
        pickle.dump(H1_events_list,open(f'./pre_computed/H1_DC_{pid}_{len(dm31_range)}x{len(th23_range)}x{len(emm_range)}x{len(emt_range)}.p','wb'))
