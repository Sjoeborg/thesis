import sys, os

if __name__ == "__main__":
    os.chdir("../../")
import numpy as np
import pandas as pd
import warnings
import pickle
from DC.processer import get_flux, interpolate_flux_DC  # DC flux can be used


def MC_PINGU():
    df = pd.read_csv(f"../../src/data/files/PINGU/neutrino_mc.csv", dtype=np.float64)
    df["reco_coszen"] = np.cos(df["reco_zenith"])
    df["true_coszen"] = np.cos(df["true_zenith"])
    return df


if __name__ == "__main__":
    pass
