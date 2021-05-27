import sys,os
if __name__ == '__main__':
    #os.chdir('../')
    sys.path.append('./src/data')
    sys.path.append('./src/events')
    sys.path.append('./src/probability')
import numpy as np
from IC.importer import *
from IC.processer import *
from IC.event_processing import *
from IC.main import sim_events
from functions import ic_params
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-dm41From', default=0.01, type=float)
parser.add_argument('-dm41To', default=10, type=float)
parser.add_argument('-dm41N', default=10, type=int)

parser.add_argument('-s24From', default=0.01, type=float)
parser.add_argument('-s24To', default=1, type=float)
parser.add_argument('-s24N', default=10, type=int)

parser.add_argument('-s', default = 0, type=int)
parser.add_argument('-sT', default = 1, type=int)
parser.add_argument('-N', default=13, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    alpha = 0.99
    gamma=0

    dm41_range = np.logspace(np.log10(args.dm41From), np.log10(args.dm41To),args.dm41N)
    s24_range = np.linspace(args.s24From,args.s24To,args.s24N)

    params = ic_params.copy()


    param_list = list_of_params(params, dm41_range, s24_range)


    print(f'Precomputing IO IC events for dm41({dm41_range.min()},{dm41_range.max()},{len(dm41_range)}),',
            f's24({s24_range.min()},{s24_range.max()},{len(s24_range)})')


    from multiprocessing import Pool
    p = Pool()
    data = [(alpha, args.N,p, False,False, [False, np.median(1), gamma],False, False,4) for p in param_list]
    H1_events_list = p.starmap(sim_events, data)
    p.close()
    pickle.dump(H1_events_list,open(f'./pre_computed/H1_IC_N{args.N}_{len(dm41_range)}x{len(s42_range)}.p','wb'))