import sys,os
#os.chdir('../')
sys.path.append('./src/data')
sys.path.append('./src/events')
sys.path.append('./src/probability')
import argparse
from probability.functions import ic_params_nsi
from data.processer import generate_probabilities, get_Etrue, train_energy_resolution, get_probabilities
from event_processing import list_of_params_nsi
import numpy as np
import time
import pandas as pd
import pickle
from multiprocessing import Pool
parser = argparse.ArgumentParser()
parser.add_argument('-s24From', default=0.01, type=float)
parser.add_argument('-s24To', default=1, type=float)
parser.add_argument('-s24N', default=10, type=int)
parser.add_argument('-emm', default=1e-1, type=float)
parser.add_argument('-emmN', default=10, type=int)
parser.add_argument('-emt', default=None, type=float)
parser.add_argument('-emtN', default=10, type=int)
parser.add_argument('-N', default = 13, type=int)
parser.add_argument('-s', default = 0, type=int)
parser.add_argument('-sT', default = 1, type=int)
parser.add_argument('-v', action='store_true')
parser.add_argument('-Ndim', default=4, type=int)
args = parser.parse_args()



def probs(E_index, z_index, alpha, npoints, params=ic_params_nsi, nsi=True, ndim=args.Ndim):
    z_buckets = np.linspace(-1,0,21)

    zr = np.linspace(z_buckets[z_index], z_buckets[z_index+1], npoints)

    Et, _, _ = get_Etrue(E_index=E_index,npoints=npoints, model=models, left_alpha=alpha, right_alpha=alpha) 
    
    try:
        get_probabilities('m','m',E_index, z_index, params,False,npoints,ndim=ndim)
    except:
        if args.v:
            print(E_index, z_index, params['e_mm'], params['theta_24'], params['e_mt'])
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

def precompute_probs(args_tuple, nsi=True):
    i,j,params = args_tuple
    event_wrapper([i,j, 0.99, params,args.N, nsi])

models= train_energy_resolution()

if __name__ == '__main__':
    if args.Ndim > 3:
        s24_range = np.logspace(np.log10(args.s24From),np.log10(args.s24To),args.s24N)
    else:
        s24_range=np.linspace(0.,0.,1)
    emm_range = np.linspace(-args.emm,args.emm,args.emmN)
    emt_range = None
    if args.emt is not None:
        emt_range = np.linspace(-args.emt,args.emt,args.emtN)
    nsi_params = ic_params_nsi.copy()
    nsi_params['dm_41'] = 0.93
    param_list = list_of_params_nsi(nsi_params, s24_range,emm_range, emt_range)
    
    if emt_range is not None:
        print(f'Precomputing {args.Ndim}dim probabilities for dm_41 ={param_list[0]["dm_41"]}, s24({s24_range.min()},{s24_range.max()},{len(s24_range)}), emm({emm_range.min()},{emm_range.max()},{len(emm_range)}), emt({emt_range.min()},{emt_range.max()},{len(emt_range)}), for N = {args.N}. s={args.s+1}/{args.sT}')
    else:
        print(f'Precomputing {args.Ndim} probabilities for dm_41 ={param_list[0]["dm_41"]}, s24({s24_range.min()},{s24_range.max()},{len(s24_range)}), emm({emm_range.min()},{emm_range.max()},{len(emm_range)}), for N = {args.N}. s={args.s+1}/{args.sT}')
    
    #split_array=  np.array_split(param_list,args.sT)[args.s]
    bins = [(i,j) for i in range(0,13) for j in range(20)]
    split_array=  np.array_split(bins,args.sT)[args.s]
    para = [(*b,p) for b in split_array.tolist() for p in param_list]

    start = time.time()
    #p = Pool()
    for i, _ in enumerate(map(precompute_probs, para), 1):
        print(f'{args.s+1}/{args.sT}: ','{0:%}'.format(i/len(split_array)))
        print(np.round((time.time() - start)/3600,1))
    #p.close()
    print(f'Finished part {args.s+1}/{args.sT} in {(np.round((time.time() - start)/3600,1))} h')
    
