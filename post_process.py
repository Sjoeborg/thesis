import os
import argparse
import numpy as np 
import pandas as pd
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('-N', default = 13, type=int)
parser.add_argument('-u', action='store_true')
args = parser.parse_args()
E_range = range(3,13)
z_range = range(0,20)
flavors = ['Pamam', 'Paeam','Pem','Pmm', 'Pmt','Pamat']

def gather_precomputed(npoints=args.N, update=args.u):
    for flavor in flavors:
        for En in E_range:
            for zn in z_range:
                filenames=[]
                try:
                    for file in os.listdir(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}/'):
                        if file.endswith('.npy'):
                            filenames.append(file[0:-4])
                    try:
                        df = pickle.load(open(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}.p','rb'))
                    except FileNotFoundError:
                        df = pd.DataFrame(index=[f'E{En}z{zn}'], dtype='object')

                    for file in filenames:
                        array = np.load(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}/{file}.npy')
                        try:
                            df.insert(loc=0,column=file, value=[array])
                        except ValueError: 
                            if update:  # If entry already exists, overwrite/update it
                                df[file][f'E{En}z{zn}'] = array
                    pickle.dump(df,open(f'./pre_computed/4gen/{flavor}/{npoints}/E{En}z{zn}.p','wb'))
                except FileNotFoundError:
                    pass

def delete_files(npoints=args.N):
    for flavor in flavors:
        for En in E_range:
            for zn in z_range:
                try:
                    for file in os.listdir(f'./4gen/{flavor}/{npoints}/E{En}z{zn}/'):
                        if file.endswith('.npy'):
                            os.remove(f'./4gen/{flavor}/{npoints}/E{En}z{zn}/{file}')
                except FileNotFoundError:
                    pass
if __name__ == '__main__':
    gather_precomputed(args.N)
    delete_files(args.N)