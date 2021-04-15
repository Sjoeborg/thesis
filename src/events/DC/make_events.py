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
from functions import dc_params_nsi, dc_params
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s24From', default=0, type=float)
parser.add_argument('-s24To', default=0, type=float)
parser.add_argument('-s24N', default=1, type=int)
parser.add_argument('-emm', default=0, type=float)
parser.add_argument('-emmN', default=1, type=int)
parser.add_argument('-emt', default=0, type=float)
parser.add_argument('-emtN', default=1, type=int)
parser.add_argument('-Ndim', default=3, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    ndim = args.Ndim
    gamma=0
    dc_params['dm_41'] = 0.93
    dc_params_nsi['dm_41'] = 0.93
    emm_range = np.linspace(-args.emm,args.emm,args.emmN)
    if ndim == 3:
        s24_range = np.linspace(0.,0.,1)
    else:
        s24_range= np.logspace(np.log10(args.s24From),np.log10(args.s24To),args.s24N)
    emt_range = np.linspace(-args.emt,args.emt,args.emtN)
    param_list = list_of_params_nsi(dc_params_nsi, s24_range, emm_range ,emt_range)

    print(emm_range)
    print(s24_range)
    print(emt_range)

    for pid in [1,0]:
        from multiprocessing import Pool
        p = Pool()
        data = [(p,pid,True) for p in param_list]
        H1_events_list = p.starmap(get_all_events, data)
        p.close()
        if len(emt_range) > 1:
            pickle.dump(H1_events_list,open(f'./pre_computed/H1_DC_{pid}_{len(emm_range)}x{len(s24_range)}x{len(emt_range)}_tau_nsi.p','wb'))
        else:
            pickle.dump(H1_events_list,open(f'./pre_computed/H1_DC_{pid}_{len(emm_range)}x{len(s24_range)}_tau_nsi.p','wb'))
