import numpy as np
import matplotlib.pyplot as plt
from src.probability.numerical import P_num, P_num_over_E
from src.probability.analytical import P_an
from src.probability.functions import ic_params, r_earth, mass_dict
import matplotlib
from multiprocessing import Pool
from src.data.DC.processer import get_interpolators_DC, get_flux

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rc("text", usetex=True)
matplotlib.rc("text.latex", preamble=r"\usepackage{amsmath}")
np.set_printoptions(precision=3)


def an_plots(
    flavor_from_list=["e", "m", "t"],
    flavor_to_list=["e", "m", "t"],
    param="L",
    material="vac",
    E=None,
    L=None,
    param_min=None,
    param_max=None,
    earth_start=0,
    ndim=3,
    anti=False,
):
    styles = ["solid", "dotted", "dashed"]
    colors = ["blue", "red", "green"]
    ncols = len(flavor_from_list)
    nrows = len(flavor_to_list)
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for col, flavor_from in enumerate(flavor_from_list):
        for row, flavor_to in enumerate(flavor_to_list):
            x_an, y_an = P_an(
                flavor_from,
                flavor_to,
                param,
                E,
                L,
                param_min,
                param_max,
                material,
                earth_start,
                ndim,
            )
            ax[col, row].plot(x_an, y_an, linestyle=styles[0])
            ax[col, row].set_title(f"{flavor_from} to {flavor_to}")
    if param == "E":
        plt.suptitle(
            rf"Vacuum oscillations in Earth core, $E = \in [{param_min},{param_max}]$ GeV, $L = {L}$ km"
        )
    elif param == "L":
        plt.suptitle(
            rf"Vacuum oscillations in Earth core, $E = {E}$ GeV, $L \in [{param_min},{param_max}]$ km"
        )
    plt.tight_layout()
    plt.show()


def compare_an_nu(
    flavor_from_list,
    flavor_to_list,
    param,
    material,
    E=None,
    L=None,
    param_min=None,
    param_max=None,
    earth_start=0,
    ndim=3,
    anti=False,
):
    styles = ["solid", "dotted", "dashed"]
    colors = ["blue", "red", "green"]
    ncols = len(flavor_from_list)
    nrows = len(flavor_to_list)
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for col, flavor_from in enumerate(flavor_from_list):
        for row, flavor_to in enumerate(flavor_to_list):
            x_nu, y_nu = P_num(
                earth_start=earth_start,
                E=E,
                param_min=param_min,
                param_max=param_max,
                ndim=ndim,
                flavor_from=flavor_from,
                flavor_to=flavor_to,
            )
            x_an, y_an = P_an(
                flavor_from,
                flavor_to,
                param,
                E,
                L,
                param_min,
                param_max,
                material,
                earth_start,
                ndim,
            )
            ax[col, row].plot(x_an, y_an, linestyle=styles[0])
            ax[col, row].plot(x_nu, y_nu[row], linestyle=styles[1], c="black")
            ax[col, row].set_title(f"{flavor_from} to {flavor_to}")
    if param == "E":
        plt.suptitle(
            rf"Vacuum oscillations in Earth core, $E = \in [{param_min},{param_max}]$ GeV, $L = {L}$ km"
        )
    elif param == "L":
        plt.suptitle(
            rf"Vacuum oscillations in Earth core, $E = {E}$ GeV, $L \in [{param_min},{param_max}]$ km"
        )
    plt.tight_layout()
    plt.show()


def P_over_E_parameter(
    flavor_from,
    param_dict_list,
    E,
    zenith=-1,
    ndim=3,
    anti=False,
    nsi=False,
    tols=(1e-4, 1e-7),
):
    """
    Returns the range of energies and the list of all flavour oscillation probabilities.
    """

    args = [
        (
            flavor_from,
            E,
            None,
            2 * r_earth,
            zenith,
            ndim,
            False,
            None,
            anti,
            p,
            "earth",
            nsi,
            tols,
        )
        for p in param_dict_list
    ]
    p = Pool()
    # res = []
    # for p in param_dict_list:
    #    res.append(P_num_over_E_wrapper(p))
    res = p.starmap(P_num_over_E, args)

    return np.array(res)


def plot_P_E_params(
    x,
    P,
    ax,
    flavor_to="m",
    colors=None,
    legend_name="",
    legend_values="",
    ylabel="",
    xlabel="",
    title="",
):
    if colors is None:
        colors = ["black", "blue", "red", "green", "brown"]
    beta = mass_dict[flavor_to]
    for i in range(len(legend_values)):
        ax.plot(
            x / 1e3,
            P[i][beta],
            label=f"{legend_name} = {np.round(legend_values[i],2)}",
            color=colors[i],
        )
        ax.tick_params(axis="y", which="major", length=5, direction="in", right=True)
        ax.tick_params(axis="y", which="minor", length=2, direction="in", right=True)
    ax.set_xlabel(f"{xlabel}")
    ax.set_ylabel(f"{ylabel}")
    ax.set_xlim((np.min(x[0] / 1e3), np.max(x[0] / 1e3)))
    ax.set_xscale("log")
    ax.set_ylim((0, 1.1))
    ax.set_title(f"{title}")
    ax.legend()


def wrap(flavor_from, E, zenith, ndim, anti, params, nsi):
    return P_num_over_E(
        flavor_from=flavor_from,
        E=E,
        zenith=zenith,
        ndim=ndim,
        anti=anti,
        params=params,
        nsi=nsi,
    )


def wrap_param(p_list):
    flavor_from, E, zenith, ndim, anti, params, nsi = p_list
    return P_over_E_parameter(
        flavor_from, params, E, zenith=zenith, ndim=ndim, anti=anti, nsi=nsi
    )


def _oscillogram_no_pool(p_list):
    """
    p = [flavor_from, E, zenith, ndim, anti, params]
    """
    res = list(map(wrap_param, p_list))
    Pxm = np.array(res)[:, :, 1, :]
    return np.swapaxes(Pxm, 1, 2)


def _oscillogram(p_list):
    """
    p = [flavor_from, E, zenith, ndim, anti, params]
    """
    p = Pool()
    res = p.starmap(wrap, p_list)
    p.close()
    Pxm = np.array(res)[:, 1, :]
    Pxe = np.array(res)[:, 0, :]
    Pxt = np.array(res)[:, 2, :]
    return Pxe, Pxm, Pxt


def oscillogram(E_range, z_range, params):
    """
    Returns (Pex, Pmx, Paeax, Pamax)
    """
    lista_mbar = [("m", E_range, z, 3, True, params, False) for z in z_range]
    lista_ebar = [("e", E_range, z, 3, True, params, False) for z in z_range]
    lista_m = [("m", E_range, z, 3, False, params, False) for z in z_range]
    lista_e = [("e", E_range, z, 3, False, params, False) for z in z_range]

    Pmx = _oscillogram(lista_m)
    Pex = _oscillogram(lista_e)
    Pamax = _oscillogram(lista_mbar)
    Paeax = _oscillogram(lista_ebar)

    return Pex, Pmx, Paeax, Pamax


def nsi_oscillogram(E_range, z_range, params):
    """
    Returns (Pex_nsi, Pmx_nsi, Paeax_nsi, Pamax_nsi)
    """
    lista_mbar_nsi = [("m", E_range, z, 3, True, params, True) for z in z_range]
    lista_ebar_nsi = [("e", E_range, z, 3, True, params, True) for z in z_range]
    lista_m_nsi = [("m", E_range, z, 3, False, params, True) for z in z_range]
    lista_e_nsi = [("e", E_range, z, 3, False, params, True) for z in z_range]

    Pmx_nsi = _oscillogram(lista_m_nsi)
    Pex_nsi = _oscillogram(lista_e_nsi)
    Pamax_nsi = _oscillogram(lista_mbar_nsi)
    Paeax_nsi = _oscillogram(lista_ebar_nsi)

    return Pex_nsi, Pmx_nsi, Paeax_nsi, Pamax_nsi


def flux_oscillogram(E_range, z_range, params, nsi=False):
    """
    Returns 1-flux_final/flux_initial for Pxm
    """
    lista_mbar = [("m", E_range, z, 4, True, params, nsi) for z in z_range]
    lista_ebar = [("e", E_range, z, 4, True, params, nsi) for z in z_range]
    lista_m = [("m", E_range, z, 4, False, params, nsi) for z in z_range]
    lista_e = [("e", E_range, z, 4, False, params, nsi) for z in z_range]
    Pmm = _oscillogram_no_pool(lista_m)
    Pem = _oscillogram_no_pool(lista_e)
    Pamam = _oscillogram_no_pool(lista_mbar)
    Paeam = _oscillogram_no_pool(lista_ebar)

    interp_flux, _ = get_interpolators_DC()
    E_mesh, z_mesh = np.meshgrid(E_range, z_range)
    flux_m = get_flux("m", E_mesh, z_mesh, interp_flux).reshape(
        len(E_range), len(z_range), 1
    )
    flux_e = get_flux("e", E_mesh, z_mesh, interp_flux).reshape(
        len(E_range), len(z_range), 1
    )
    flux_mbar = get_flux("mbar", E_mesh, z_mesh, interp_flux).reshape(
        len(E_range), len(z_range), 1
    )
    flux_ebar = get_flux("ebar", E_mesh, z_mesh, interp_flux).reshape(
        len(E_range), len(z_range), 1
    )

    flux_initial = flux_mbar + flux_ebar + flux_m + flux_e
    flux_final = flux_mbar * Pamam + flux_ebar * Paeam + flux_m * Pmm + flux_e * Pem

    return 1 - flux_final / flux_initial


def nsi_flux_oscillogram(E_range, z_range, params):
    """
    Returns (flux_final/flux_initial, flux_bar_final/flux_bar_initial, flux_both_final/flux_both_initial)
    for either only Pxm flux (only_m is True), or for only Pxe+Pxt flux (only_m is False)
    """
    lista_mbar_nsi = [("m", E_range, z, 3, True, params, True) for z in z_range]
    lista_ebar_nsi = [("e", E_range, z, 3, True, params, True) for z in z_range]
    lista_m_nsi = [("m", E_range, z, 3, False, params, True) for z in z_range]
    lista_e_nsi = [("e", E_range, z, 3, False, params, True) for z in z_range]

    lista_mbar = [("m", E_range, z, 3, True, params, False) for z in z_range]
    lista_ebar = [("e", E_range, z, 3, True, params, False) for z in z_range]
    lista_m = [("m", E_range, z, 3, False, params, False) for z in z_range]
    lista_e = [("e", E_range, z, 3, False, params, False) for z in z_range]

    Pmx_nsi = _oscillogram(lista_m_nsi)
    Pex_nsi = _oscillogram(lista_e_nsi)
    Pamax_nsi = _oscillogram(lista_mbar_nsi)
    Paeax_nsi = _oscillogram(lista_ebar_nsi)

    Pmx = _oscillogram(lista_m)
    Pex = _oscillogram(lista_e)
    Pamax = _oscillogram(lista_mbar)
    Paeax = _oscillogram(lista_ebar)

    interp_flux, _ = get_interpolators_DC()
    E_mesh, z_mesh = np.meshgrid(E_range, z_range)
    flux_m = get_flux("m", E_mesh, z_mesh, interp_flux)
    flux_e = get_flux("e", E_mesh, z_mesh, interp_flux)
    flux_mbar = get_flux("mbar", E_mesh, z_mesh, interp_flux)
    flux_ebar = get_flux("ebar", E_mesh, z_mesh, interp_flux)

    # maybe clean up
    flux_final_m = flux_m * Pmx[1] + flux_e * Pex[1]
    flux_final_m_nsi = flux_m * Pmx_nsi[1] + flux_e * Pex_nsi[1]

    flux_final_m_bar = flux_mbar * Pamax[1] + flux_ebar * Paeax[1]
    flux_final_m_nsi_bar = flux_mbar * Pamax_nsi[1] + flux_ebar * Paeax_nsi[1]

    flux_final_e = flux_m * (Pmx[0]) + flux_e * (Pex[0])
    flux_final_e_nsi = flux_m * (Pmx_nsi[0]) + flux_e * (Pex_nsi[0])

    flux_final_e_bar = flux_mbar * (Pamax[0]) + flux_ebar * (Paeax[0])
    flux_final_e_nsi_bar = flux_mbar * (Pamax_nsi[0]) + flux_ebar * (Paeax_nsi[0])

    flux_final_t = flux_m * (Pmx[2]) + flux_e * (Pex[2])
    flux_final_t_nsi = flux_m * (Pmx_nsi[2]) + flux_e * (Pex_nsi[2])

    flux_final_t_bar = flux_mbar * (Pamax[2]) + flux_ebar * (Paeax[2])
    flux_final_t_nsi_bar = flux_mbar * (Pamax_nsi[2]) + flux_ebar * (Paeax_nsi[2])

    return (
        (flux_final_e_nsi + flux_final_e_nsi_bar) / (flux_final_e + flux_final_e_bar),
        (flux_final_m_nsi + flux_final_m_nsi_bar) / (flux_final_m + flux_final_m_bar),
        (flux_final_t_nsi + flux_final_t_nsi_bar) / (flux_final_t + flux_final_t_bar),
    )


def save(fig, name):
    fig.savefig(f"article/figures/{name}.pdf")
    from subprocess import call

    _ = call("pdfcrop article/figures/{name}.pdf article/figures/{name}.pdf")


def savethesis(fig, name):
    fig.savefig(f"thesis/tex/figures/{name}.pdf")
    from subprocess import call

    _ = call("pdfcrop thesis/tex/figures/{name}.pdf thesis/tex/figures/{name}.pdf")


if __name__ == "__main__":
    an_plots()
