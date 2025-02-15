import sys

if __name__ == "__main__":
    sys.path.append("./../../src/data")
    sys.path.append("./../../src/events")
    sys.path.append("./../../src/probability")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.data.IC.importer import *
from src.data.IC.processer import *
from src.probability.functions import perform_chisq
from scipy.stats import chi2

IC_observed_full = np.array(get_IC_data().T)
E_rate, z_rate = get_flux_factor()
flux_E_factors_full, flux_z_factors_full = bin_flux_factors(E_rate, z_rate)
EFrom, ETo, zFrom, zTo = 0, 13, 0, 20
z_bins = np.arange(zFrom, zTo)
E_bins, z_bins_T = np.arange(EFrom, ETo), np.arange(zFrom, zTo)[:, None]
n_Ebins, n_zbins = len(E_bins), len(z_bins)
flux_E_factors = flux_E_factors_full[E_bins]
flux_z_factors = flux_z_factors_full[z_bins]
flux_factors = np.outer(flux_E_factors, flux_z_factors)
IC_observed = IC_observed_full[E_bins, z_bins_T].T

E_ratios_full = pd.read_csv(
    "../../src/data/files/E_ratios.csv", header=None, names=["Ereco", "Eratio"]
).Eratio.values
z_ratios_full = pd.read_csv(
    "../../src/data/files/z_ratios.csv", header=None, names=["zreco", "zratio"]
).zratio.values
IC_per_z_full = np.array(np.sum(IC_observed_full, axis=0))
IC_per_E_full = np.array(np.sum(IC_observed_full, axis=1))
MC_per_E_full = IC_per_E_full / E_ratios_full
MC_per_z_full = IC_per_z_full / z_ratios_full

MC_ratios_full = np.outer(E_ratios_full, z_ratios_full)
IC_MC_full = IC_observed_full / MC_ratios_full
IC_MC = IC_MC_full[EFrom : ETo + 1, z_bins]

Ereco_full = 500 * 10 ** np.linspace(0.0, 1.3, 14)
Ereco_full_midpoints = Ereco_full[0:-1] + np.diff(Ereco_full) / 2  # For scatter plot

Ereco = Ereco_full[EFrom : ETo + 1]
Ereco_midpoints = Ereco_full_midpoints[EFrom : ETo + 1]

zreco_full = np.linspace(-1, 0, 21)
zreco_full_midpoints = zreco_full[0:-1] + np.diff(zreco_full) / 2  # For scatter plot

zreco = zreco_full[zFrom : zTo + 1]
zreco_midpoints = zreco_full_midpoints[z_bins]


def to_hist(arr):
    return arr.tolist() + [arr[-1]]


def count_plots(H1, H0):
    IC_per_z = np.sum(IC_observed, axis=0)
    IC_per_E = np.sum(IC_observed, axis=1)

    IC_rate_z = IC_per_z / np.sum(H0, axis=0)
    IC_rate_E = IC_per_E / np.sum(H0, axis=1)

    H1_per_z_hist = to_hist(np.sum(H1, axis=0))
    H1_per_E_hist = to_hist(np.sum(H1, axis=1))
    H0_per_z_hist = to_hist(np.sum(H0, axis=0))
    H0_per_E_hist = to_hist(np.sum(H0, axis=1))

    H1_rate_z_hist = to_hist(np.sum(H1, axis=0) / np.sum(H0, axis=0))
    H1_rate_E_hist = to_hist(np.sum(H1, axis=1) / np.sum(H0, axis=1))
    H0_rate_z_hist = to_hist(np.sum(H0, axis=0) / np.sum(H0, axis=0))
    H0_rate_E_hist = to_hist(np.sum(H0, axis=1) / np.sum(H0, axis=1))

    fig, ax = plt.subplots(
        2,
        2,
        sharex="col",
        squeeze=True,
        gridspec_kw={"width_ratios": [3, 3], "height_ratios": [3, 1]},
        figsize=(12, 8),
    )
    ax = ax.flatten()

    label_size = 15
    tick_size = 15
    legend_size = 13

    ax[0].scatter(
        Ereco_midpoints, IC_per_E, label="IC data", s=10, color="black", zorder=10
    )
    ax[0].step(Ereco, H1_per_E_hist, label="Sterile", lw=3, where="post", color="blue")
    ax[0].step(Ereco, H0_per_E_hist, label="Null", lw=1.5, where="post", color="red")

    ax[1].scatter(
        zreco_midpoints, IC_per_z, label="IC data", s=10, color="black", zorder=10
    )
    ax[1].step(zreco, H1_per_z_hist, label="Sterile", lw=3, where="post", color="blue")
    ax[1].step(zreco, H0_per_z_hist, label="Null", lw=1.5, where="post", color="red")

    ax[2].scatter(
        Ereco_midpoints, IC_rate_E, label="IC data", s=10, color="black", zorder=10
    )
    ax[2].step(Ereco, H1_rate_E_hist, label="Sterile", lw=3, where="post", color="blue")
    ax[2].step(Ereco, H0_rate_E_hist, label="Null", lw=1.5, where="post", color="red")

    ax[3].scatter(
        zreco_midpoints, IC_rate_z, label="IC data", s=10, color="black", zorder=10
    )
    ax[3].step(zreco, H1_rate_z_hist, label="Sterile", lw=3, where="post", color="blue")
    ax[3].step(zreco, H0_rate_z_hist, label="Null", lw=1.5, where="post", color="red")

    ax[0].set_xlim((Ereco.min(), Ereco.max()))
    ax[0].set_ylabel("Counts", fontsize=label_size)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")

    ax[2].set_xscale("log")
    ax[2].set_xlabel(r"$E^{reco}$ [GeV]", fontsize=label_size)
    ax[2].set_ylabel("Ratio to Null", fontsize=label_size)
    ax[2].grid(True, which="both", axis="both", alpha=0.3)

    ax[3].set_xlim((zreco.min(), zreco.max()))
    ax[3].set_ylim(ax[2].get_ylim())
    ax[3].grid(True, which="both", axis="both", alpha=0.3)
    ax[3].set_xlabel(r"$\cos{(\theta^{reco}_z)}$", fontsize=label_size)

    ax[0].tick_params(axis="both", direction="in", which="both")
    ax[1].tick_params(axis="both", direction="in", which="both")
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    leg = ax[0].legend(fontsize=legend_size)
    leg.get_frame().set_edgecolor("k")
    plt.subplots_adjust(hspace=0.05)

    return fig


def get_boundary(arr):
    returned = []
    for i in range(arr.shape[1]):
        try:
            returned.append((np.max(np.nonzero(arr[:, i] == True)) + 1))
        except ValueError:
            returned.append(0)
    returned = np.array(returned)
    max_val = arr.shape[0]
    returned[returned >= max_val] = (
        max_val - 1
    )  # If a column has all true, set cotour at last row
    return np.array(returned)


def norm_plot(simulated_events):
    normalization = IC_observed / simulated_events
    n_zbins, n_Ebins = normalization.shape
    fig, ax = plt.subplots()

    fig.set_size_inches(18.5, 10.5)
    im = ax.imshow(
        normalization, cmap="GnBu", origin="lower", extent=(0, n_zbins, 0, n_Ebins)
    )
    cbar = ax.figure.colorbar(im, ax=ax)


def normalize_bin_by_bin(simulated_events, MC=True, MC_old=False, correct_flux=False):
    if MC_old:
        IC_events = IC_MC_2017
    elif MC:
        IC_events = IC_MC

    if correct_flux:
        simulated_events = flux_E_factors * simulated_events

    normalization = IC_events / simulated_events

    return np.array(normalization)


def normalize_events(H0_events, H1_events_list, z_bins):
    norm_factors = normalize_bin_by_bin(H0_events[:, z_bins], MC=True)
    H0_normalized = norm_factors * H0_events[:, z_bins]
    H1_list_normalized = [norm_factors * H1[:, z_bins] for H1 in H1_events_list]

    return H0_normalized, H1_list_normalized


def get_deltachi(H1_list_normalized, delta_T, sigma=[0.25, 0.15], f=0.09, x0=[1, 0, 0]):
    sigma_a = sigma[0]
    sigma_b = sigma[1]
    sigma_g = delta_T
    if np.ndim(H1_list_normalized) != 3:
        data = np.sum(IC_observed, axis=0)
    elif H1_list_normalized.shape[2] == 13:
        data = np.sum(IC_observed, axis=1)
    else:
        data = IC_observed
    sigma_syst = f * data
    chisq_H1_list = np.array(
        [
            perform_chisq(
                H1_norm,
                data,
                sigma_syst=sigma_syst,
                z=zreco,
                sigma_a=sigma_a,
                sigma_b=sigma_b,
                sigma_g=sigma_g,
                x0=x0,
            )[0]
            for H1_norm in H1_list_normalized
        ]
    )
    delta_chi = chisq_H1_list - np.min(chisq_H1_list)  # chisq_H1_list - chisq_H0

    best_fit_index = np.argmin(delta_chi)

    return chisq_H1_list, best_fit_index


def get_contour(deltachi, y_range, x_range, df):
    cl_99_bool = np.where(deltachi < chi2.ppf(q=0.99, df=df), True, False)
    cl_90_bool = np.where(deltachi < chi2.ppf(q=0.90, df=df), True, False)

    x_cl90_index = get_boundary(cl_90_bool)
    y_cl90_index = np.linspace(0, len(x_cl90_index) - 1, len(x_cl90_index)).astype(
        "int"
    )
    x_cl99_index = get_boundary(cl_99_bool)
    y_cl99_index = np.linspace(0, len(x_cl99_index) - 1, len(x_cl99_index)).astype(
        "int"
    )

    return (
        x_range[x_cl90_index],
        x_range[x_cl99_index],
        y_range[y_cl90_index],
        y_range[y_cl99_index],
    )


def list_of_params_nsi(dicta, s24_range, emm_range, emt_range=None, eem_range=None):
    def update_dict(dict, p):
        dict2 = dicta.copy()
        dict2.update(p)
        return dict2

    if emt_range is None and eem_range is None:
        dict_list = [
            update_dict(dicta, {"e_mm": mm, "theta_24": np.arcsin(np.sqrt(s24)) / 2})
            for mm in emm_range
            for s24 in s24_range
        ]
    elif emt_range is not None and eem_range is None:
        dict_list = [
            update_dict(
                dicta, {"e_mm": mm, "e_mt": mt, "theta_24": np.arcsin(np.sqrt(s24)) / 2}
            )
            for mt in emt_range
            for mm in emm_range
            for s24 in s24_range
        ]
    elif emt_range is None and eem_range is not None:
        dict_list = [
            update_dict(
                dicta, {"e_mm": mm, "e_em": em, "theta_24": np.arcsin(np.sqrt(s24)) / 2}
            )
            for em in eem_range
            for mm in emm_range
            for s24 in s24_range
        ]
    else:  # both emt_range and eem_range are provided
        dict_list = [
            update_dict(
                dicta, 
                {
                    "e_mm": mm, 
                    "e_mt": mt, 
                    "e_em": em, 
                    "theta_24": np.arcsin(np.sqrt(s24)) / 2
                }
            )
            for mt in emt_range
            for em in eem_range
            for mm in emm_range
            for s24 in s24_range
        ]
    return dict_list


def list_of_params(dict, dm_range, s24_range, s34_range=None, s24_eq_s34=False):
    def update_dict(dict, p):
        dict2 = dict.copy()
        dict2.update(p)
        return dict2

    if s24_eq_s34:
        dict_list = [
            update_dict(
                dict,
                {
                    "dm_41": v,
                    "theta_24": np.arcsin(np.sqrt(k)) / 2,
                    "theta_34": np.arcsin(np.sqrt(k)) / 2,
                },
            )
            for k in s24_range
            for v in dm_range
        ]
    elif s34_range is not None:
        dict_list = [
            update_dict(
                dict,
                {
                    "dm_41": v,
                    "theta_24": np.arcsin(np.sqrt(k)) / 2,
                    "theta_34": np.arcsin(np.sqrt(j)) / 2,
                },
            )
            for j in s34_range
            for k in s24_range
            for v in dm_range
        ]
    else:
        dict_list = [
            update_dict(dict, {"dm_41": v, "theta_24": np.arcsin(np.sqrt(k)) / 2})
            for k in s24_range
            for v in dm_range
        ]
    return dict_list


if __name__ == "__main__":
    pass
