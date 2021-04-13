import sys,os
if __name__ == '__main__':
    os.chdir('../../')
import numpy as np
import pandas as pd
import warnings
import pickle
from DC.processer import get_flux, interpolate_flux_DC #DC flux can be used


def MC_PINGU(track, cascade):
    df = pd.read_csv(f'./src/data/files/PINGU/neutrino_mc.csv', dtype=np.float64)
    return df


if __name__ == '__main__':
    pass