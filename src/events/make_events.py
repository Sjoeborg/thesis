import sys,os
if __name__ == '__main__':
    #os.chdir('../')
    sys.path.append('./src/data')
    sys.path.append('./src/events')
    sys.path.append('./src/probability')
import numpy as np
import pandas as pd
from PINGU.importer import *
from PINGU.processer import *
from DC.event_processing import * # DC.event_processing has all we need. 
from PINGU.main import get_all_events as PINGU_events
from DC.main import get_all_events as DC_events
from IC.main import sim_events as IC_events
from functions import nufit_params_nsi, nufit_params_nsi_IO
import pickle
import argparse
from multiprocessing import Pool
parser = argparse.ArgumentParser()

parser.add_argument('-dm31N', default=10, type=int)
parser.add_argument('-th23N', default=10, type=int)

parser.add_argument('-ett', default=5e-2, type=float)
parser.add_argument('-ettN', default=10, type=int)
parser.add_argument('-emt', default=2e-2, type=float)
parser.add_argument('-emtN', default=10, type=int)

parser.add_argument('-eem', default=2e-1, type=float)
parser.add_argument('-eemN', default=10, type=int)
parser.add_argument('-eet', default=2e-1, type=float)
parser.add_argument('-eetN', default=10, type=int)

parser.add_argument('-s', default = 0, type=int)
parser.add_argument('-sT', default = 1, type=int)
parser.add_argument('-IO', action='store_true')
parser.add_argument('-DC', action='store_true')
parser.add_argument('-PINGU', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    assert args.PINGU or args.DC or args.IC
    dm31_range, th23_range,ett_range, emt_range, eem_range, eet_range = get_param_list(args.dm31N, args.th23N, 
                                                                                       args.ett, args.ettN, 
                                                                                       args.emt, args.emtN, 
                                                                                       args.eem, args.eemN, 
                                                                                       args.eet, args.eetN, 
                                                                                       args.IO)

    print('dm:', dm31_range)
    print('th:', th23_range)
    print('ett:', ett_range)
    print('emt:', emt_range)
    print('eem:', eem_range)
    print('eet:', eet_range)
    param_dict = nufit_params_nsi_IO if args.IO else nufit_params_nsi

    param_list = list_of_params_nsi(param_dict, dm31_range, th23_range, ett_range, emt_range, eem_range, eet_range)
    
    for pid in [1,0]:
        data = [(p,pid,True) for p in param_list]
        if args.PINGU:
            p = Pool()
            H1_events_list = p.starmap(PINGU_events, data)
            p.close()
            pickle.dump(H1_events_list,open(f'./pre_computed/H1_PINGU_{pid}_{len(dm31_range)}x{len(th23_range)}x{len(ett_range)}x{len(emt_range)}x{len(eem_range)}x{len(eet_range)}.p','wb'))
        elif args.DC:
            p = Pool()
            H1_events_list = p.starmap(DC_events, data)
            p.close()
            pickle.dump(H1_events_list,open(f'./pre_computed/H1_DC_{pid}_{len(dm31_range)}x{len(th23_range)}x{len(ett_range)}x{len(emt_range)}x{len(eem_range)}x{len(eet_range)}.p','wb'))
    if args.IC:
        data = [(0.99, 13,p, False,False, [False, 0, 0],True, True,3) for p in param_list]
        p = Pool()
        H1_events_list = p.starmap(IC_events, data)
        p.close()
        pickle.dump(H1_events_list,open(f'./pre_computed/H1_IC_N13_{len(dm31_range)}x{len(th23_range)}x{len(emt_range)}.p','wb'))
    