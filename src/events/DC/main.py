import sys,os
if __name__ == '__main__':
    sys.path.append('./src/probability')
    sys.path.append('./src/data')
import numpy as np 
import pandas as pd 
from DC.processer import get_true_DC, generate_probabilities_DC, get_probabilities_DC, MC2018_DC
from functions import dc_params_nsi

df = MC2018_DC(track=True, cascade=True)
livetime = 1022*24*3600

def get_events(Ebin,zbin,params,pid,null):
    events = 0
    for anti in [True,False]:
        for flavor_from in ['e','m','t']:
            for flavor_to in ['e','m','t']:
                try:
                    df2 = get_true_DC(flavor=flavor_to,anti=anti,pid=pid,E_bin=Ebin,z_bin=zbin,df=df)
                    Etrue,ztrue,weights = df2.true_energy.values, df2.true_coszen.values, df2.rate_weight.values
                    P = get_probabilities_DC(flavor_from, flavor_to, Ebin,zbin,params,anti,pid,ndim=3)
                except KeyError:
                    P = generate_probabilities_DC(flavor_from,flavor_to,Etrue,ztrue,Ebin, zbin, params,anti,pid,ndim=3, nsi=True)
                if not null:
                    events += np.sum(P*weights)
                else:
                    events += np.sum(weights)
    return events

def get_all_events(params, pid, null):
    result = np.zeros((8,4))
    for Ebin in range(8):
        for zbin in range(4):
            result[Ebin,zbin] = get_events(Ebin,zbin,params,pid,null)
    return result
print(get_events(Ebin=0,zbin=0,params=dc_params_nsi,pid=0,null=True))