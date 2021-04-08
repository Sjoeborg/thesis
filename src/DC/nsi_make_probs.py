import sys,os
#os.chdir('../')
sys.path.append('./src/data')
sys.path.append('./src/DC')
sys.path.append('./src/events')
sys.path.append('./src/probability')
import argparse
from functions import dc_params_nsi
from processerDC import generate_probabilitiesDC, get_true, get_probabilitiesDC
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
parser.add_argument('-s', default = 0, type=int)
parser.add_argument('-sT', default = 1, type=int)
parser.add_argument('-tracks', action='store_true')
parser.add_argument('-v', action='store_true')
parser.add_argument('-Ndim', default=4, type=int)
args = parser.parse_args()

df = pd.read_csv('./src/data/files/DC/2018/sample_b/neutrino_mc.csv')

def probs(E_index, z_index, pid,params=dc_params_nsi, nsi=True, ndim=args.Ndim):
    z_buckets = [-1., -0.75, -0.5 , -0.25,  0., 0.25, 0.5, 0.75, 1.]
    E_buckets = [5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413]


    Et_mu, zt_mu = get_true('m',anti=False,pid=pid,E_bin=E_index,z_bin=z_index,df=df)
    Et_mubar, zt_mubar = get_true('m',anti=True,pid=pid,E_bin=E_index,z_bin=z_index,df=df)
    Et_tau, zt_tau = get_true('t',anti=False,pid=pid,E_bin=E_index,z_bin=z_index,df=df)
    Et_taubar, zt_taubar = get_true('t',anti=True,pid=pid,E_bin=E_index,z_bin=z_index,df=df)
    try:
        get_probabilitiesDC('m', 'm', Ebin=E_index, zbin=z_index, param_dict=params,
                            anti=False,pid=pid,ndim=ndim)
    except:
        if args.v:
            print(E_index, z_index, params['e_mm'], params['theta_24'], params['e_mt'])
        generate_probabilitiesDC('m','m',Et_mu,zt_mu,E_index, z_index, params,False,pid,ndim=ndim, nsi=nsi)
    try:
        get_probabilitiesDC('m', 'm', Ebin=E_index, zbin=z_index, param_dict=params,
                            anti=True,pid=pid,ndim=ndim)
    except:
        generate_probabilitiesDC('m','m',Et_mubar,zt_mubar,E_index, z_index, params,True,pid,ndim=ndim, nsi=nsi)
    try:
        get_probabilitiesDC('e', 'm', Ebin=E_index, zbin=z_index, param_dict=params,
                            anti=False,pid=pid,ndim=ndim)
    except:
        generate_probabilitiesDC('e','m',Et_mu,zt_mu,E_index, z_index, params,False,pid,ndim=ndim, nsi=nsi)
    try:
        get_probabilitiesDC('e', 'm', Ebin=E_index, zbin=z_index, param_dict=params,
                            anti=True,pid=pid,ndim=ndim)
    except:
        generate_probabilitiesDC('e','m',Et_mubar,zt_mubar,E_index, z_index, params,True,pid,ndim=ndim, nsi=nsi)
    
    try:
        get_probabilitiesDC('m', 't', Ebin=E_index, zbin=z_index, param_dict=params,
                            anti=False,pid=pid,ndim=ndim)
    except:
        generate_probabilitiesDC('m','t',Et_tau,zt_tau,E_index, z_index, params,False,pid,ndim=ndim, nsi=nsi)
    try:
        get_probabilitiesDC('m', 't', Ebin=E_index, zbin=z_index, param_dict=params,
                            anti=True,pid=pid,ndim=ndim)
    except:
        generate_probabilitiesDC('m','t',Et_taubar,zt_taubar,E_index, z_index, params,True,pid,ndim=ndim, nsi=nsi)
    
    

def event_wrapper(param_list):
    E_index,z_index, params, pid,nsi = param_list[0], param_list[1], param_list[2], param_list[3], param_list[4]
    return probs(E_index=E_index, z_index=z_index, params=params, pid=pid, nsi=nsi)

def precompute_probs(args_tuple, nsi=True):
    i,j,params = args_tuple
    event_wrapper([i,j, params,pid, nsi])


if __name__ == '__main__':
    if args.Ndim > 3:
        s24_range = np.logspace(np.log10(args.s24From),np.log10(args.s24To),args.s24N)
    else:
        s24_range=np.linspace(0.,0.,1)
    if args.tracks:
        pid = 1
    else:
        pid=0
    emm_range = np.linspace(-args.emm,args.emm,args.emmN)
    emt_range = None
    if args.emt is not None:
        emt_range = np.linspace(-args.emt,args.emt,args.emtN)
    nsi_params = dc_params_nsi.copy()
    nsi_params['dm_41'] = 0.93
    param_list = list_of_params_nsi(nsi_params, s24_range,emm_range, emt_range)
    
    if emt_range is not None:
        print(f'Precomputing DC {args.Ndim}dim probabilities for dm_41 ={param_list[0]["dm_41"]}, s24({s24_range.min()},{s24_range.max()},{len(s24_range)}), emm({emm_range.min()},{emm_range.max()},{len(emm_range)}), emt({emt_range.min()},{emt_range.max()},{len(emt_range)}), for pid {pid}. s={args.s+1}/{args.sT}')
    else:
        print(f'Precomputing DC {args.Ndim}dim probabilities for dm_41 ={param_list[0]["dm_41"]}, s24({s24_range.min()},{s24_range.max()},{len(s24_range)}), emm({emm_range.min()},{emm_range.max()},{len(emm_range)}), for pid {pid}. s={args.s+1}/{args.sT}')
    
    bins = [(i,j) for i in range(8) for j in range(4)]
    split_array=  np.array_split(bins,args.sT)[args.s]
    para = [(*b,p) for b in split_array.tolist() for p in param_list]

    start = time.time()
    for i, _ in enumerate(map(precompute_probs, para), 1):
        print(f'{args.s+1}/{args.sT}: ','{0:%}'.format(i/len(para)))
        print(np.round((time.time() - start)/3600,1))
    print(f'Finished part {args.s+1}/{args.sT} in {(np.round((time.time() - start)/3600,1))} h')
    
