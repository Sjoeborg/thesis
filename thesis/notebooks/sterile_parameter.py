import sys,os
sys.path.append('./src/')
sys.path.append('./src/data')
sys.path.append('./src/events')
sys.path.append('./src/probability')
import numpy as np
import pickle
from functions import ic_params
from plotter import flux_oscillogram
from IC.event_processing import list_of_params

if __name__ == '__main__':
    E_range = np.logspace(2,5,1)
    z_range= np.linspace(-1,0,1)
    dm_range = np.logspace(-1,2,1)
    s24_range = np.logspace(-2,0,1)
    params = list_of_params(ic_params, dm_range,s24_range)
    res = flux_oscillogram(E_range, z_range, params, nsi=False)
    pickle.dump(res,open('./pre_computed/sterile_animation.p','wb'))