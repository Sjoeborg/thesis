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
from IC.processer import generate_probabilities, get_Etrue, train_energy_resolution, get_probabilities
from multiprocessing import Pool
parser = argparse.ArgumentParser()

parser.add_argument('-dm31N', default=10, type=int)
parser.add_argument('-th23N', default=10, type=int)

parser.add_argument('-ett', default=5e-2, type=float)
parser.add_argument('-ettN', default=1, type=int)
parser.add_argument('-emt', default=2e-2, type=float)
parser.add_argument('-emtN', default=1, type=int)

parser.add_argument('-eem', default=2e-1, type=float)
parser.add_argument('-eemN', default=1, type=int)
parser.add_argument('-eet', default=2e-1, type=float)
parser.add_argument('-eetN', default=1, type=int)

parser.add_argument('-s', default = 0, type=int)
parser.add_argument('-sT', default = 1, type=int)
parser.add_argument('-IO', action='store_true')
parser.add_argument('-IC', action='store_true')
parser.add_argument('-DC', action='store_true')
parser.add_argument('-PINGU', action='store_true')

parser.add_argument('-nonsi', action='store_false')
args = parser.parse_args()

def probs(E_index, z_index, alpha, npoints, params=nufit_params_nsi, nsi=True, ndim=3):
    z_buckets = np.linspace(-1,0,21)

    zr = np.linspace(z_buckets[z_index], z_buckets[z_index+1], npoints)

    Et, _, _ = get_Etrue(E_index=E_index,npoints=npoints, model=models, left_alpha=alpha, right_alpha=alpha) 
    
    try:
        get_probabilities('m','m',E_index, z_index, params,False,npoints,ndim=ndim)
    except:
        generate_probabilities('m','m',Et,zr,E_index, z_index, params,False,npoints,ndim=ndim, nsi=nsi)
    try:
        get_probabilities('m','m',E_index, z_index, params,True,npoints,ndim=ndim)
    except:
        #print(E_index, z_index, params['dm_41'], params['theta_24'], params['theta_34'])
        generate_probabilities('m','m',Et,zr,E_index, z_index, params,True,npoints,ndim=ndim, nsi=nsi)
    try:
        get_probabilities('e','m',E_index, z_index, params,False,npoints,ndim=ndim)
    except:
        #print(E_index, z_index, params['dm_41'], params['theta_24'], params['theta_34'])
        generate_probabilities('e','m',Et,zr,E_index, z_index, params,False,npoints,ndim=ndim, nsi=nsi)
    try:
        get_probabilities('e','m',E_index, z_index, params,True,npoints,ndim=ndim)
    except:
        #print(E_index, z_index, params['dm_41'], params['theta_24'], params['theta_34'])
        generate_probabilities('e','m',Et,zr,E_index, z_index, params,True,npoints,ndim=ndim, nsi=nsi)
    
    try:
        get_probabilities('m','t',E_index, z_index, params,False,npoints,ndim=ndim)
    except:
        generate_probabilities('m','t',Et,zr,E_index, z_index, params,False,npoints,ndim=ndim, nsi=nsi)
    try:
        get_probabilities('m','t',E_index, z_index, params,True,npoints,ndim=ndim)
    except:
        generate_probabilities('m','t',Et,zr,E_index, z_index, params,True,npoints,ndim=ndim, nsi=nsi)
    
    

def event_wrapper(param_list):
    E_index,z_index, alpha, params, npoints,nsi = param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], param_list[5]
    return probs(E_index=E_index, z_index=z_index, params=params, npoints=npoints, alpha=alpha, nsi=nsi)

def precompute_probs(args_tuple):
    i,j,params = args_tuple
    if args.PINGU:
        res_track = get_events_PINGU(Ebin=i,zbin=j,params=params,pid=1,nsi=args.nonsi)
        res_cascade =get_events_PINGU(Ebin=i,zbin=j,params=params,pid=0,nsi=args.nonsi)
    elif args.DC:
        res_track = get_events_DC(Ebin=i,zbin=j,params=params,pid=1,nsi=args.nonsi, no_osc=False)
        res_cascade =get_events_DC(Ebin=i,zbin=j,params=params,pid=0,nsi=args.nonsi, no_osc=False)
    else:#IC
        event_wrapper([i,j, 0.99, params,13, args.nonsi])
        return 0
    return np.array([res_cascade, res_track])

models= train_energy_resolution()
if __name__ == '__main__':
    if args.PINGU:
        experiment = 'PINGU' 
    elif args.DC:
       experiment = 'DC' 
    elif args.IC:
        experiment = 'IC' 
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
    '''
    bins = [(i,j) for i in range(8) for j in range(8)]
    split_array=  np.array_split(bins,args.sT)[args.s]
    para = [(*b,p) for b in split_array.tolist() for p in param_list]
    '''

    arg_tuples = [(i,j, p) for i in range(8) for j in range(8) for p in param_list]
    rng = np.random.default_rng(12345)
    rng.shuffle(arg_tuples)
    split_arg_tuples=  np.array_split(arg_tuples,args.sT)[args.s]
    start = time.time()

    p=Pool()
    for i, res in enumerate(p.imap(precompute_probs, split_arg_tuples), 1):
        if i % 100 == 1:
            print(f'{args.s+1}/{args.sT}: ','{0:%}'.format(i/len(arg_tuples)))
            print(np.round((time.time() - start)/3600,1))
    print(f'Finished part {args.s+1}/{args.sT} in {(np.round((time.time() - start)/3600,1))} h')
    p.close()