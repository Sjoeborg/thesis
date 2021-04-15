import sys,os
import h5py
import numpy as np
import pandas as pd
from multiprocessing import Pool
def merge_h5(colab,Ebin,zbin):
    for pid in [1,0]:
        for anti in [True, False]:
            for flavor_from in ['e','m']:
                for flavor_to in ['e','m','t']:
                    if anti:
                        flavor_from = 'a' + flavor_from
                        flavor_to = 'a' + flavor_to
                    try:
                        with h5py.File(f'./pre_computed/new/{colab}/E{Ebin}z{zbin}.hdf5', 'r') as read:
                            with h5py.File(f'./pre_computed/{colab}/E{Ebin}z{zbin}.hdf5', 'a') as write:
                                for param in read[f'{Ndim}gen/P{flavor_from}{flavor_to}/{pid}/'].keys():
                                    path = f'{Ndim}gen/P{flavor_from}{flavor_to}/{pid}/'+param
                                    array = read[path][:]
                                    try:
                                        dset = write.create_dataset(path, data=array)
                                    except RuntimeError:
                                        pass # Don't update
                                    for key in read[path].attrs:
                                        write[path].attrs[key] = read[path].attrs[key]
                    except OSError:
                        pass #File not found
                
p = Pool
data = [('PINGU',Ebin,zbin) for Ebin in range(8) for zbin in range(8)]
p.starmap(merge_h5, data)

old_csv = pd.read_csv('pre_computed/PINGU/hashed_params.csv', header=None)
new_csv = pd.read_csv('pre_computed/new/PINGU/hashed_params.csv', header=None)

new_csv = pd.concat([old_csv,new_csv])
new_csv.to_csv('pre_computed/PINGU/hashed_params.csv', header=False, index=False)