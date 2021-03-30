import os
import argparse
import numpy as np 
import pandas as pd
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('-N', default = 13, type=int)
parser.add_argument('-u', action='store_true')
parser.add_argument('-d', action='store_true')
parser.add_argument('-s', default = 0, type=int)
parser.add_argument('-sT', default = 1, type=int)
args = parser.parse_args()
E_range = range(0,13)
z_range = range(0,20)
flavors = ['Pamam', 'Paeam','Pem','Pmm', 'Pmt','Pamat']


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
                except FileNotFoundError:
                    pass
            print(f'{flavor}, E{En}z{zn} done')

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
if __name__ == '__main__':
    split_z=  np.array_split(z_range,args.sT)[args.s]
    gather_precomputed(split_z,args.N,args.u)
    #delete_files(args.N)
    #merge_precomputed_df()
