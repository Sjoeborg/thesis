import argparse
from events import list_of_params, ic_params
from dataProcesser import generate_probabilities, get_Etrue, train_energy_resolution, get_probabilities
import numpy as np
import time
from multiprocessing import Pool
parser = argparse.ArgumentParser()
parser.add_argument('-dmFrom', default=0.1, type=float)
parser.add_argument('-dmTo', default=10, type=float)
parser.add_argument('-dmN', default=5, type=int)
parser.add_argument('-s24From', default=0.01, type=float)
parser.add_argument('-s24To', default=1, type=float)
parser.add_argument('-s24N', default=5, type=int)
parser.add_argument('-s34From', default=0.01, type=float)
parser.add_argument('-s34To', default=1, type=float)
parser.add_argument('-s34N', default=5, type=int)
parser.add_argument('-N', default = 9, type=int)
parser.add_argument('-s', default = 0, type=int)
parser.add_argument('-sT', default = 1, type=int)
parser.add_argument('-s34eqs24', action='store_true')
parser.add_argument('-s34', action='store_true')

args = parser.parse_args()


def probs(E_index, z_index, alpha, npoints, params=ic_params):
    z_buckets = np.linspace(-1,0,21)

    zr = np.linspace(z_buckets[z_index], z_buckets[z_index+1], npoints)

    Et, _, _ = get_Etrue(E_index=E_index,npoints=npoints, model=models, left_alpha=alpha, right_alpha=alpha) 
    
    try:
        get_probabilities('m','m',E_index, z_index, params,False,npoints,ndim=4)
    except:
        generate_probabilities('m','m',Et,zr,E_index, z_index, params,False,npoints,ndim=4)
    try:
        get_probabilities('m','m',E_index, z_index, params,True,npoints,ndim=4)
    except:
        generate_probabilities('m','m',Et,zr,E_index, z_index, params,True,npoints,ndim=4)
    try:
        get_probabilities('e','m',E_index, z_index, params,False,npoints,ndim=4)
    except:
        generate_probabilities('e','m',Et,zr,E_index, z_index, params,False,npoints,ndim=4)
    try:
        get_probabilities('e','m',E_index, z_index, params,True,npoints,ndim=4)
    except:
        generate_probabilities('e','m',Et,zr,E_index, z_index, params,True,npoints,ndim=4)
    
    try:
        get_probabilities('m','t',E_index, z_index, params,False,npoints,ndim=4)
    except:
        generate_probabilities('m','t',Et,zr,E_index, z_index, params,False,npoints,ndim=4)
    try:
        get_probabilities('m','t',E_index, z_index, params,True,npoints,ndim=4)
    except:
        generate_probabilities('e','m',Et,zr,E_index, z_index, params,True,npoints,ndim=4)
    

def event_wrapper(param_list):
    E_index,z_index, alpha, params, npoints = param_list[0], param_list[1], param_list[2], param_list[3], param_list[4]
    return probs(E_index=E_index, z_index=z_index, params=params, npoints=npoints, alpha=alpha)

def precompute_probs(params=ic_params):
    for i in range(3,13):
        for j in range(20):
            event_wrapper([i,j, 0.99, params,args.N])


models= train_energy_resolution()

if __name__ == '__main__':
    dm41_range = np.logspace(np.log10(args.dmFrom),np.log10(args.dmTo),args.dmN)
    s24_range = np.logspace(np.log10(args.s24From),np.log10(args.s24To),args.s24N)
    s34_range = np.logspace(np.log10(args.s34From),np.log10(args.s34To),args.s34N)
    if args.s34:
        param_list = list_of_params(ic_params,dm41_range, s24_range, s34_range=s34_range, s24_eq_s34=args.s34eqs24, short=False)
        print(f'Precomputing probabilities for dm_41({args.dmFrom},{args.dmTo},{args.dmN}), s24({args.sFrom},{args.sTo},{args.sN}), s34({args.s34From},{args.s34To},{args.s34N}), for N = {args.N}. s={args.s+1}/{args.sT}')
    else:
        param_list = list_of_params(ic_params,dm41_range, s24_range, short=False)
        print(f'Precomputing probabilities for dm_41({args.dmFrom},{args.dmTo},{args.dmN}), s24({args.sFrom},{args.sTo},{args.sN}), s34=0, for N = {args.N}. s={args.s+1}/{args.sT}')

    split_array=  np.array_split(param_list,args.sT)[args.s]
    
    start = time.time()
    #p = Pool()
    for i, _ in enumerate(map(precompute_probs, split_array), 1):
        print(f'{args.s+1}/{args.sT}: ','{0:%}'.format(i/len(split_array)))
        print(np.round((time.time() - start)/3600,1))
    #p.close()
    print(f'Finished part {args.s+1}/{args.sT} in {(np.round((time.time() - start)/3600,1))} s')
    
