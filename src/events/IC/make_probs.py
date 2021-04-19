import sys,os
#os.chdir('../')
sys.path.append('./src/data')
sys.path.append('./src/events')
sys.path.append('./src/probability')
import argparse
from functions import nufit_params_nsi, nufit_params_nsi_IO
from IC.processer import generate_probabilities, get_Etrue, train_energy_resolution, get_probabilities
from DC.event_processing import list_of_params_nsi
import numpy as np
import time
import pandas as pd
import pickle
from multiprocessing import Pool
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

def precompute_probs(args_tuple, nsi=True):
    i,j,params = args_tuple
    event_wrapper([i,j, 0.99, params,args.N, nsi])

models= train_energy_resolution()

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
    
    if args.IO is False:
        print(f'Precomputing NO DC probabilities for dm31({dm31_range.min()},{dm31_range.max()},{len(dm31_range)}),',
                f'th23({np.round(th23_range.min(),2)},{np.round(th23_range.max(),2)},{len(th23_range)}), emm({emm_range.min()},',
                f'{emm_range.max()},{len(emm_range)}), emt({emt_range.min()},{emt_range.max()},{len(emt_range)}),',
                f' s={args.s+1}/{args.sT}')
    else:
        print('WARNING: Hard-coded values for dm31 under IO! Input from -dm31From/To ignored')
        print(f'Precomputing IO DC probabilities for dm31({dm31_range.min()},{dm31_range.max()},{len(dm31_range)}),',
                f'th23({th23_range.min()},{th23_range.max()},{len(th23_range)}), emm({emm_range.min()},',
                f'{emm_range.max()},{len(emm_range)}), emt({emt_range.min()},{emt_range.max()},{len(emt_range)}),',
                f' s={args.s+1}/{args.sT}')
                
    bins = [(i,j) for i in range(8) for j in range(8)]
    split_array=  np.array_split(bins,args.sT)[args.s]
    para = [(*b,p) for b in split_array.tolist() for p in param_list]

    start = time.time()
    
    #p = Pool()
    for i, _ in enumerate(map(precompute_probs, para), 1):
        print(f'{args.s+1}/{args.sT}: ','{0:%}'.format(i/len(para)))
        print(np.round((time.time() - start)/3600,1))
    #p.close()
    
    print(f'Finished part {args.s+1}/{args.sT} in {(np.round((time.time() - start)/3600,1))} h')
    
