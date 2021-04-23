import sys,os
sys.path.append('./src/data')
sys.path.append('./src/DC')
sys.path.append('./src/events')
sys.path.append('./src/probability')
import argparse
from functions import nufit_params_nsi, nufit_params_nsi_IO
from DC.event_processing import list_of_params_nsi, get_param_list
import numpy as np
import time
from PINGU.main import get_events as get_events_PINGU
from DC.main import get_events as get_events_DC
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

parser.add_argument('-nonsi', action='store_false')
args = parser.parse_args()


def precompute_probs(args_tuple, nsi=True):
    i,j,params = args_tuple
    if args.PINGU:
        res_track = get_events_PINGU(Ebin=i,zbin=j,params=params,pid=1,nsi=args.nonsi)
        res_cascade =get_events_PINGU(Ebin=i,zbin=j,params=params,pid=0,nsi=args.nonsi)
    elif args.DC:
        res_track = get_events_DC(Ebin=i,zbin=j,params=params,pid=1,nsi=args.nonsi, no_osc=False)
        res_cascade =get_events_DC(Ebin=i,zbin=j,params=params,pid=0,nsi=args.nonsi, no_osc=False)
    return np.array([res_cascade, res_track])


if __name__ == '__main__':
    experiment = 'PINGU' if args.PINGU else 'DC'
    mass_ordering = 'IO' if args.IO else 'NO'
    dm31_range, th23_range,ett_range, emt_range, eem_range, eet_range = get_param_list(args.dm31N, args.th23N, 
                                                                                       args.ett, args.ettN, 
                                                                                       args.emt, args.emtN, 
                                                                                       args.eem, args.eemN, 
                                                                                       args.eet, args.eetN, 
                                                                                       args.IO)

    param_dict = nufit_params_nsi_IO if args.IO else nufit_params_nsi
    param_list = list_of_params_nsi(param_dict, dm31_range, th23_range, ett_range, emt_range, eem_range, eet_range)

    print(f'Precomputing {experiment} {mass_ordering} probabilities for dm31N {len(dm31_range)},th23N {len(th23_range)},',
                f'ett({ett_range.min()},{ett_range.max()},{len(ett_range)}), ',
                f'emt({emt_range.min()},{emt_range.max()},{len(emt_range)}), ',
                f'eem({eem_range.min()},{eem_range.max()},{len(eem_range)}), ',
                f'eet({eet_range.min()},{eet_range.max()},{len(eet_range)}), ',
                f' s={args.s+1}/{args.sT}')
    
    bins = [(i,j) for i in range(8) for j in range(8)]
    split_array=  np.array_split(bins,args.sT)[args.s]
    para = [(*b,p) for b in split_array.tolist() for p in param_list]
    
    start = time.time()
    #result = []
    
    for i, res in enumerate(map(precompute_probs, para), 1):
        print(f'{args.s+1}/{args.sT}: ','{0:%}'.format(i/len(para)))
        print(np.round((time.time() - start)/3600,1))
        #result.append(res)
    #result = np.array(result).reshape(-1,len(split_array),2)
    #result = np.swapaxes(result, 0, 1)
    #result = np.swapaxes(result, 0, 2) #(pid, params, 1)
    print(f'Finished part {args.s+1}/{args.sT} in {(np.round((time.time() - start)/3600,1))} h')