import sys,os
sys.path.append('./src/data')
sys.path.append('./src/DC')
sys.path.append('./src/events')
sys.path.append('./src/probability')
import argparse
from functions import dc_params_nsi
from IC.event_processing import list_of_params_nsi
import numpy as np
import time
from PINGU.main import get_events
from multiprocessing import Pool
parser = argparse.ArgumentParser()
parser.add_argument('-s24From', default=0.01, type=float)
parser.add_argument('-s24To', default=1, type=float)
parser.add_argument('-s24N', default=10, type=int)
parser.add_argument('-emm', default=1e-1, type=float)
parser.add_argument('-emmN', default=10, type=int)
parser.add_argument('-emt', default=None, type=float)
parser.add_argument('-emtN', default=10, type=int)
parser.add_argument('-s', default = 0, type=int)
parser.add_argument('-sT', default = 1, type=int)
parser.add_argument('-tracks', action='store_true')
parser.add_argument('-v', action='store_true')
parser.add_argument('-Ndim', default=4, type=int)
args = parser.parse_args()


def precompute_probs(args_tuple, nsi=True):
    i,j,params = args_tuple
    get_events(Ebin=i,zbin=j,params=params,pid=pid,nsi=False)
    get_events(Ebin=i,zbin=j,params=params,pid=pid,nsi=True)


if __name__ == '__main__':
    s24_range = np.logspace(np.log10(args.s24From),np.log10(args.s24To),args.s24N) if args.Ndim > 3 else np.linspace(0.,0.,1)
    pid = 1 if args.tracks else 0
    emm_range = np.linspace(-args.emm,args.emm,args.emmN)
    emt_range = np.linspace(-args.emt,args.emt,args.emtN) if args.emt is not None else None

    nsi_params = dc_params_nsi.copy()
    nsi_params['dm_41'] = 0.93
    param_list = list_of_params_nsi(nsi_params, s24_range,emm_range, emt_range)
    
    if emt_range is not None:
        print(f'Precomputing PINGU {args.Ndim}dim probabilities for dm_41 ={param_list[0]["dm_41"]}, \
                s24({s24_range.min()},{s24_range.max()},{len(s24_range)}), emm({emm_range.min()},{emm_range.max()}, \
                {len(emm_range)}), emt({emt_range.min()},{emt_range.max()},{len(emt_range)}), for pid {pid}. s={args.s+1}/{args.sT}')
    else:
        print(f'Precomputing PINGU {args.Ndim}dim probabilities for dm_41 ={param_list[0]["dm_41"]}, \
                s24({s24_range.min()},{s24_range.max()},{len(s24_range)}), emm({emm_range.min()},{emm_range.max()}, \
                {len(emm_range)}), for pid {pid}. s={args.s+1}/{args.sT}')
    
    bins = [(i,j) for i in range(8) for j in range(8)]
    split_array=  np.array_split(bins,args.sT)[args.s]
    para = [(*b,p) for b in split_array.tolist() for p in param_list]

    start = time.time()
    for i, _ in enumerate(map(precompute_probs, para), 1):
        print(f'{args.s+1}/{args.sT}: ','{0:%}'.format(i/len(para)))
        print(np.round((time.time() - start)/3600,1))
    print(f'Finished part {args.s+1}/{args.sT} in {(np.round((time.time() - start)/3600,1))} h')
    
