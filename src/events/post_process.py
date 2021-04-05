import os
import argparse
import numpy as np 
import pandas as pd
import pickle
import h5py
from multiprocessing import Process,Event, Pool
parser = argparse.ArgumentParser()
parser.add_argument('-N', default = 13, type=int)
parser.add_argument('-u', action='store_true')
parser.add_argument('-d', action='store_true')
parser.add_argument('-s', default = 0, type=int)
parser.add_argument('-sT', default = 1, type=int)
args = parser.parse_args()
E_range = range(0,13)
z_range = range(0,20)
flavors = ['Pmm', 'Pamam', 'Pem', 'Paeam','Pmt','Pamat']

def gather_precomputed(z_bins,npoints=args.N, update=args.u):
    #print('Inserting precomputed arrays into dfs')
    for flavor in flavors:
        for En in E_range:
            for zn in z_bins:
                filenames=[]
                try:
                    for file in os.listdir(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}/'):
                        if file.endswith('.npy'):
                            filenames.append(file[0:-4])
                    try:
                        df = pickle.load(open(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}.p','rb'))
                    except (FileNotFoundError,EOFError):
                        df = pd.DataFrame(index=[f'E{En}z{zn}'], dtype='object')

                    for file in filenames:
                        array = np.load(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}/{file}.npy')
                        try:
                            df.insert(loc=0,column=file, value=[array])
                        except ValueError: 
                            if update:  # If entry already exists, overwrite/update it
                                df[file][f'E{En}z{zn}'] = array
                    pickle.dump(df,open(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}.p','wb'))
                    if args.d:
                        for file in filenames:
                            os.remove(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}/{file}.npy')
                except (FileNotFoundError,ValueError):
                    print(f'Could not load {flavor} E{En}z{zn}, skipping it')
            print(f'{flavor}, E{En} done')

def merge_precomputed_df(npoints=args.N):
    print('Merging precomputed dfs')
    for flavor in flavors:
        for En in E_range:
            for zn in z_range:
                try:
                    try:
                        old_df = pickle.load(open(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}.p','rb'))
                    except FileNotFoundError:
                        old_df = pd.DataFrame(index=[f'E{En}z{zn}'], dtype='object')

                    probs = pickle.load(open(f'./pre_computed/new/4gen/{flavor}/{npoints}/E{En}z{zn}.p','rb'))
                    old_df.update(probs)
                    pickle.dump(old_df,open(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}.p','wb'))
                    print(f'Updated E{En}z{zn} for {flavor}')
                except FileNotFoundError:
                    pass


def delete_files(npoints=args.N):
    for flavor in flavors:
        for En in E_range:
            for zn in z_range:
                try:
                    for file in os.listdir(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}/'):
                        if file.endswith('.npy'):
                            os.remove(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}/{file}')
                except FileNotFoundError:
                    pass
def df_to_hdf(En,zn, group, npoints, param_dict, N,add_attrs=False):
    for flavor in flavors:
        try:
            df = pickle.load(open(f'./pre_computed/{group}/{flavor}/{npoints}/E{En}z{zn}.p','rb'))
            hashed_list = df.iloc[0].index
            arrays = df.iloc[0].values
            f = h5py.File(f'./pre_computed/IC/E{En}z{zn}.hdf5', 'a')
            #print(f'{group}/{flavor}/{N}/{hashed_list[0]}')
            #print(arrays[0])
            for i in range(len(arrays)): 
                dset = f.create_dataset(f'{group}/{flavor}/{N}/{hashed_list[i]}', data=arrays[i])
            f.close()
            os.remove(f'./pre_computed/{group}/{flavor}/{npoints}/E{En}z{zn}.p')
        except RuntimeError:
            return
                    


if __name__ == '__main__':
    #split_z=  np.array_split(z_range,args.sT)[args.s]
    #gather_precomputed(split_z,args.N,args.u)
    group = '4gen'

    for En in E_range:
        for zn in z_range:
            df_to_hdf(En,zn,group, args.N, {'dm':1, 'th':0.5}, 13,add_attrs=False)
        print(f'{En}{zn} done')
    #delete_files(args.N)
    #merge_precomputed_df()
