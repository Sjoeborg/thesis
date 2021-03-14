import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-N', default = 13, type=int)
args = parser.parse_args()
E_range = range(3,13)
z_range = range(0,20)
flavors = ['Pamam', 'Paeam','Pem','Pmm']

for flavor in flavors:
    for En in E_range:
        for zn in z_range:
            for file in os.listdir(f'./4gen/{flavor}/{args.N}/E{En}z{zn}/'):
                if file.endswith('.npy'):
                    os.remove(f'./4gen/{flavor}/{args.N}/E{En}z{zn}/{file}')