import sys,os
sys.path.append('./src/data')
sys.path.append('./src/probability')
import numpy as np 
import pandas as pd 
from PINGU.processer import get_true_PINGU,generate_probabilities_PINGU, get_probabilities_PINGU
from PINGU.importer import PINGU_MC

df = PINGU_MC(track=True, cascade=True)
livetime = 365*24*3600 #arbitary, set to 1 year

def get_events(Ebin,zbin,params,pid,null):
    events = 0
    for anti in [True,False]:
        for flavor_from in ['e','m','t']:
            for flavor_to in ['e','m','t']:
                try:
                    df2 = get_true_PINGU(flavor=flavor_to,anti=anti,pid=pid,E_bin=Ebin,z_bin=zbin,df=df)
                    Etrue,ztrue,weights = df2.true_energy.values, df2.true_coszen.values, df2.rate_weight.values
                    P = get_probabilities_PINGU(flavor_from, flavor_to, Ebin,zbin,params,anti,pid,ndim=3)
                except KeyError:
                    P = generate_probabilities_PINGU(flavor_from,flavor_to,Etrue,ztrue,Ebin, zbin, params,anti,pid,ndim=3, nsi=True)
                if not null:
                    events += np.sum(P*weights)
                else:
                    events += np.sum(weights)
    return events

def get_all_events(params, pid, null):
    result = np.zeros((8,8))
    for Ebin in range(8):
        for zbin in range(8):
            result[Ebin,zbin] = get_events(Ebin,zbin,params,pid,null)
    return result
