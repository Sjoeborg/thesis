import sys,os
import h5py
import numpy as np
def merge_h5(colab,Ebin,zbin,Ndim,flavor_from,flavor_to,anti,pid):
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
[merge_h5('PINGU',Ebin,zbin,3,f_from,f_to,anti,pid) for Ebin in range(8) for zbin in range(8) for f_to in ['e','m','t'] for f_from in ['e','m'] for anti in [True, False] for pid in [0,1]]