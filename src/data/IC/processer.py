import sys, os

if __name__ == "__main__":
    sys.path.append("./../../src/probability")
    sys.path.append("./../../src/data")
import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator as CT
from src.probability.functions import mass_dict
from src.data.IC.importer import get_flux_df, get_aeff_df
from dict_hash import sha256
import pandas as pd
from src.probability.numerical import wrapper
from scipy.stats import lognorm
import pickle
import os

import h5py
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.exceptions import InconsistentVersionWarning
import warnings
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)



def get_flux(flavor, E, z, df):
    """
    Returns flux for a set of flavor, energy [GeV], and z=cos(theta_z).
    """
    try:
        flux_avg = df[f"{flavor}_flux"][0](E, z)
    except KeyError:
        raise KeyError("NYI for tau flux")
    return np.abs(flux_avg)


def interpolate_aeff(recompute):
    if recompute:
        df = get_aeff_df()
        E = df.E_avg
        z_avg = df.z_avg
        aeff = np.array(df.Aeff)

        points_avg = np.array([E, z_avg]).T
        f_avg = CT(points_avg, aeff, rescale=True)
        pickle.dump(f_avg, open("../../pre_computed/aeff_interpolator.p", "wb"))
    else:
        try:
            f_avg = pickle.load(open("../../pre_computed/aeff_interpolator.p", "rb"))
        except:
            raise FileNotFoundError(
                "File ´aeff_interpolator.p´ not present in ´../../pre_computed/´. Rerun with recompute = True to generate it."
            )

    return f_avg


def interpolate_flux(recompute=False):
    """
    Returns a df of the interpolated fluxes.
    """
    colnames = ["m_flux", "mbar_flux", "e_flux", "ebar_flux"]
    if not recompute:
        try:
            inter_df = pickle.load(open("../../pre_computed/flux_interpolator.p", "rb"))
        except:
            print(os.getcwd())
            raise FileNotFoundError(
                "File ´flux_interpolator.p´ not present in ´../../pre_computed/´. Rerun with recompute = True to generate it."
            )
    else:
        df = get_flux_df()
        E = df.GeV
        z_avg = (df.z_min + df.z_max) / 2

        points_avg = np.array([E, z_avg]).T

        interp_list = []
        for flavor in colnames:
            phi = df[flavor]
            values = np.array(phi)

            f_avg = CT(
                points_avg, values, rescale=True
            )  # Rescale seems to have no effect, but is good according to doc
            interp_list.append([f_avg])

        inter_df = pd.DataFrame(np.transpose(interp_list), columns=colnames)
        pickle.dump(inter_df, open("../../pre_computed/flux_interpolator.p", "wb"))
    return inter_df


def interpolate_flux_DC(recompute=False):
    """
    Returns a df of the interpolated fluxes.
    """
    colnames = ["m_flux", "mbar_flux", "e_flux", "ebar_flux"]
    if not recompute:
        try:
            inter_df = pickle.load(
                open("../../pre_computed/flux_interpolator_DC.p", "rb")
            )
        except:
            raise FileNotFoundError(
                "File ´flux_interpolator_DC.p´ not present in ´../../pre_computed/´. Rerun with recompute = True to generate it."
            )
    else:
        df = get_flux_df_DC()
        E = df.GeV
        z_avg = (df.z_min + df.z_max) / 2

        points_avg = np.array([E, z_avg]).T

        interp_list = []
        for flavor in colnames:
            phi = df[flavor]
            values = np.array(phi)

            f_avg = CT(
                points_avg, values, rescale=True
            )  # Rescale seems to have no effect, but is good according to doc
            interp_list.append([f_avg])

        inter_df = pd.DataFrame(np.transpose(interp_list), columns=colnames)
        pickle.dump(inter_df, open("../../pre_computed/flux_interpolator_DC.p", "wb"))
    return inter_df


def bin_flux_factors(E_df, z_df):
    E_bins = 500 * 10 ** (np.linspace(0.0, 1.3, 14))
    z_bins = np.linspace(-1.0, 0.0, 21)
    E_res = []
    z_res = []
    for i in range(13):
        mean_per_bin = E_df[E_df.E.between(E_bins[i], E_bins[i + 1])].factor.mean()
        E_res.append(mean_per_bin)
    for i in range(20):
        mean_per_bin = z_df[z_df.E.between(z_bins[i], z_bins[i + 1])].factor.mean()
        z_res.append(mean_per_bin)
    return np.array(E_res), np.array(z_res)


def bin_flux_factors_DC(E_df, z_df):
    z_buckets = np.linspace(-1, 1, 9)
    E_buckets = np.logspace(0.75, 1.75, 9)
    E_res = []
    z_res = []
    for i in range(9):
        mean_per_bin = E_df[
            E_df.E.between(E_buckets[i], E_buckets[i + 1])
        ].factor.mean()
        E_res.append(mean_per_bin)
    for i in range(9):
        mean_per_bin = z_df[
            z_df.E.between(z_buckets[i], z_buckets[i + 1])
        ].factor.mean()
        z_res.append(mean_per_bin)
    return np.array(E_res), np.array(z_res)


def interpolate_aeff_dc(recompute=False):
    if not recompute:
        try:
            inter = pickle.load(open("../../pre_computed/aeff_dc_interpolator.p", "rb"))
        except:
            raise FileNotFoundError(
                "File aeff_dc_interpolator.p´ not present in ´../../pre_computed/´. Run ´interpolate_aeff_dc()´ with recompute = True to generate it."
            )
    else:
        aeff_df = get_aeff_df_dc()
        from scipy.interpolate import interp1d

        inter = interp1d(aeff_df.logE, aeff_df.Aeff)
        pickle.dump(inter, open("../../pre_computed/aeff_dc_interpolator.p", "wb"))
    return inter


def get_probabilities(flavor_from, flavor_to, Ebin, zbin, param_dict, anti, N, ndim):
    hashed_param_name = sha256(param_dict)
    if anti:
        flavor_from = "a" + flavor_from
        flavor_to = "a" + flavor_to
    try:
        f = h5py.File(f"../../pre_computed/IC/E{Ebin}z{zbin}.hdf5", "r")
    except OSError:
        raise KeyError(f"E{Ebin}z{zbin}.hdf5 doesnt exist in ../../pre_computed/IC/")
    try:
        fh = f[f"{ndim}gen/P{flavor_from}{flavor_to}/{N}/{hashed_param_name}"]
    except KeyError:
        f.close()
        raise KeyError(
            f"{ndim}gen/P{flavor_from}{flavor_to}/{N}/{hashed_param_name} doesnt exist in E{Ebin}z{zbin}.hdf5"
        )
    res = fh[()]
    f.close()
    return res


def generate_probabilities(
    flavor_from,
    flavor_to,
    E_range,
    z_range,
    E_bin,
    z_bin,
    params,
    anti,
    N,
    ndim=4,
    nsi=False,
    save=True,
):
    prob = np.array(
        [
            wrapper([flavor_from, E_range, z, anti, params, ndim, nsi])[
                mass_dict[flavor_to]
            ]
            for z in z_range
        ]
    )
    if save:
        hashed_param_name = sha256(params)
        if anti:
            flavor_from = "a" + flavor_from
            flavor_to = "a" + flavor_to
        f = h5py.File(f"../../pre_computed/IC/E{E_bin}z{z_bin}.hdf5", "a")
        try:
            dset = f.create_dataset(
                f"{ndim}gen/P{flavor_from}{flavor_to}/{N}/{hashed_param_name}",
                data=prob,
                chunks=True,
            )
            for key in params.keys():
                dset.attrs[key] = params[key]
            f.close()
        except (RuntimeError, ValueError):
            print(
                f"{ndim}gen/P{flavor_from}{flavor_to}/{N}/{hashed_param_name} already exists, skipping"
            )
            f.close()
            return prob
        if E_bin == 5 and z_bin == 5 and flavor_from == "am" and flavor_to == "am":
            with open(f"../../pre_computed/IC/hashed_params.csv", "a") as fd:
                fd.write(f"{params};{hashed_param_name}\n")
    return prob


def train_energy_resolution(recompute=False):
    # TODO: Finetune this
    if not recompute:
        try:
            gpr = pickle.load(
                open("../../pre_computed/energy_resolution_models.p", "rb")
            )
        except:
            raise FileNotFoundError(
                "File energy_resolution_models.p´ not present in ´.../../pre_computed/´. Run ´train_energy_resolution()´ with recompute=True to generate it."
            )
    else:
        filename = "../NuFSGenMC_nominal.dat"
        df = pd.read_csv(
            filename,
            delimiter=" ",
            names=[
                "pdg",
                "Ereco",
                "zreco",
                "Etrue",
                "ztrue",
                "mcweight",
                "flux_pion",
                "flux_kaon",
            ],
            skiprows=12,
        )
        df.Ereco = np.round(df.Ereco, 0)
        df = df.groupby("Ereco").median().reset_index()
        df["Ebin"] = pd.cut(df.Ereco, bins=500 * 10 ** np.linspace(0.0, 1.3, 14))
        if len(df) > 5000:
            df_subsetted = df.sample(5000, random_state=0)
        X = np.array(np.log(df_subsetted.Ereco)).reshape(-1, 1)
        y = np.log(df_subsetted.Etrue)
        kernel2 = 1.0 * RBF() + WhiteKernel(noise_level=3)
        gpr = GaussianProcessRegressor(kernel=kernel2, random_state=0).fit(X, y)
        pickle.dump(gpr, open("../../pre_computed/energy_resolution_models.p", "wb"))
    return gpr


def get_Etrue(model, npoints, left_alpha, right_alpha, E_index=None, Ereco=False):
    E_buckets = 500 * 10 ** (np.linspace(0.0, 1.3, 14))
    if not E_index is None:
        Ereco = np.logspace(
            np.log10(E_buckets[E_index]), np.log10(E_buckets[E_index + 1]), npoints
        )
    mu_base_e, std_base_e = model.predict(np.log(Ereco).reshape(-1, 1), return_std=True)

    Etrue = np.logspace(
        np.log10(
            lognorm.ppf(1 - left_alpha, s=std_base_e[0], scale=np.exp(mu_base_e[0]))
        ),
        np.log10(
            lognorm.ppf(right_alpha, s=std_base_e[-1], scale=np.exp(mu_base_e[-1]))
        ),
        npoints,
    )
    return Etrue, mu_base_e, std_base_e


def get_interpolators(
    recompute_flux=False, recompute_aeff=False, recompute_energy_res=False
):
    interp_flux = interpolate_flux(recompute_flux)

    interp_aeff = interpolate_aeff(recompute_aeff)
    gpr_models = train_energy_resolution(recompute_energy_res)

    return interp_flux, interp_aeff, gpr_models


def get_interpolators_dc(recompute_flux=False, recompute_aeff=False):
    interp_flux = interpolate_flux_DC(recompute_flux)
    interp_aeff = interpolate_aeff_dc(recompute_aeff)

    return interp_flux, interp_aeff


if __name__ == "__main__":
    pass
