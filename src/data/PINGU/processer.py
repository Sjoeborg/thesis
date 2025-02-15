import sys, os
import numpy as np
import pandas as pd
from src.probability.functions import mass_dict, nufit_params_nsi
from dict_hash import sha256
import pandas as pd
from src.probability.numerical import P_num
import h5py
from scipy.stats import lognorm
import pickle
import time
overwrite = False

pdg_dict = {"e": 12, "m": 14, "t": 16}
Ebins_2018 = [
    5.623413,
    7.498942,
    10.0,
    13.335215,
    17.782795,
    23.713737,
    31.622776,
    42.16965,
    56.23413,
]
zbins_2018 = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]


def get_probabilities_PINGU(
    flavor_from, flavor_to, Ebin, zbin, param_dict, anti, pid, ndim, nsi
):
    hashed_param_name = sha256(param_dict)
    if anti:
        flavor_from = "a" + flavor_from
        flavor_to = "a" + flavor_to

    try:
        f = h5py.File(f"../../pre_computed/PINGU/E{Ebin}z{zbin}.hdf5", "r")
    except OSError:
        raise KeyError(f"E{Ebin}z{zbin}.hdf5 doesnt exist in ../../pre_computed/PINGU/")
    try:
        fh = f[f"{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name}"]
    except KeyError:
        f.close()
        # print(f'{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name} doesnt exist in E{Ebin}z{zbin}.hdf5')
        raise KeyError(
            f"{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name} doesnt exist in E{Ebin}z{zbin}.hdf5"
        )
    res = fh[()]
    f.close()
    return res


def generate_probabilities_PINGU(
    flavor_from,
    flavor_to,
    E_range,
    z_range,
    E_bin,
    z_bin,
    param_dict,
    anti,
    pid,
    ndim=4,
    nsi=True,
    save=True,
):
    prob = np.array(
        [
            P_num(
                flavor_from=flavor_from,
                E=E_range[i],
                ndim=ndim,
                anti=anti,
                params=param_dict,
                zenith=z,
                nsi=nsi,
            )[mass_dict[flavor_to], -1]
            for i, z in enumerate(z_range)
        ]
    ).reshape(-1, 1)
    if save:
        hashed_param_name = sha256(param_dict)
        if anti:
            flavor_from = "a" + flavor_from
            flavor_to = "a" + flavor_to

        f = h5py.File(f"../../pre_computed/PINGU/E{E_bin}z{z_bin}.hdf5", "a")

        try:
            dset = f.create_dataset(
                f"{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name}",
                data=prob,
                chunks=True,
            )
            for key in param_dict.keys():
                dset.attrs[key] = param_dict[key]
            f.close()
        except RuntimeError:
            if overwrite:
                dset = f[
                    f"{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name}"
                ]
                dset[...] = prob
            else:
                print(
                    f"{ndim}gen/P{flavor_from}{flavor_to}/{pid}/{hashed_param_name} already exists, skipping"
                )
            f.close()
            return prob
        if (
            E_bin == 5
            and z_bin == 5
            and flavor_from == "am"
            and flavor_to == "am"
            and pid == 1
        ):
            with open(f"../../pre_computed/PINGU/hashed_params.csv", "a") as fd:
                fd.write(f"{param_dict};{hashed_param_name}\n")

    return prob


def get_binned_PINGU(pid, E_bin, z_bin, df):
    df1 = (
        df.query(f"pid=={pid}")
        .query(f"reco_energy<{Ebins_2018[E_bin+1]}")
        .query(f"reco_energy>{Ebins_2018[E_bin]}")
        .query(f"reco_coszen<{zbins_2018[z_bin+1]}")
        .query(f"reco_coszen>{zbins_2018[z_bin]}")
    )

    return df1


if __name__ == "__main__":
    pass
