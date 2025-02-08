import sys

if __name__ == "__main__":
    sys.path.append("./../src/probability")
    sys.path.append("./../src/data")
import numpy as np
from src.data.IC.processer import (
    get_flux,
    generate_probabilities,
    get_Etrue,
    get_interpolators,
)
from src.data.IC.importer import *
from multiprocessing import Pool
from src.probability.functions import ic_params, integrate
from scipy.stats import lognorm


def get_events(
    E_index,
    z_index,
    alpha,
    npoints,
    params=ic_params,
    spectral_shift_parameters=[False, 2e3, 0.02],
    null=False,
    tau=False,
    nsi=False,
    ndim=4,
    gen2=False,
):
    """
    Gets the number of events (N_ij) in IceCube for E_index and z_index.
    alpha: the percentile from which to sample true points from. 0.99 is recommended. Higher alpha needs flux for lower and higher energies.
    npoints: the number of points in which to integrate over. Recommended values between 13 and 19.
    params: dictionary with oscillation parameters
    spectral_shift_parameters: list with 3 elements. If first is True, use spectral shifting. Second element is pivot point, third is delta gamma (shift).
    null: If True, take P_amam == 1 (no sterile hypothesis)
    tau: if True, use decayed tauons (17.39% BR) and include in final probability.
    nsi: If True, construct Hamiltonian assuming NSI as specified in params.
    ndim: Number of neutrino generations (active AND sterile)
    gen2: If True, multiply Aeff by 5 to simulate Icecube Gen2
    """

    E_buckets = 500 * 10 ** (
        np.linspace(0.0, 1.3, 14)
    )  # log-binned in steps of 0.1 from 500 GeV to 500*1e1.3 GeV
    z_buckets = np.linspace(-1, 0, 21)

    Er = np.logspace(
        np.log10(E_buckets[E_index]), np.log10(E_buckets[E_index + 1]), npoints
    )
    zr = np.linspace(z_buckets[z_index], z_buckets[z_index + 1], npoints)
    # Assume zr == zt, and thus the zenith resolution function is 1.

    Et, mu, sigma = get_Etrue(
        E_index=E_index,
        npoints=npoints,
        left_alpha=alpha,
        right_alpha=alpha,
        model=energy_resolution_models,
    )
    resolution_gaussian = lognorm.pdf(
        Et, s=sigma, scale=np.exp(mu)
    )  # Check doc of scipy.stats.lognorm for explanation of variables

    _, Et_mesh = np.meshgrid(Er, Et)  # meshgrid has shape (Er,zr,Et)

    zr_mesh, _, Et_mesh = np.meshgrid(zr, Er, Et)

    aeff_m = interp_aeff(Et_mesh, zr_mesh)
    aeff_mbar = interp_aeff(Et_mesh, zr_mesh)

    if gen2:  # gen2 has 5 times aeff 1-100 TeV
        aeff_m *= 5
        aeff_mbar *= 5

    if spectral_shift_parameters[0]:
        E_pivot = spectral_shift_parameters[1]
        delta_gamma = spectral_shift_parameters[2]
        factor = spectral_shift_factor(
            E=Et_mesh, E_pivot=E_pivot, delta_gamma=delta_gamma
        )
        flux_m = factor * get_flux("m", Et_mesh, zr_mesh, interp_flux)
        flux_mbar = factor * get_flux("mbar", Et_mesh, zr_mesh, interp_flux)
    else:
        flux_m = get_flux("m", Et_mesh, zr_mesh, interp_flux)
        flux_mbar = get_flux("mbar", Et_mesh, zr_mesh, interp_flux)

    if not null:
        Pmm = generate_probabilities(
            "m",
            "m",
            Et,
            zr,
            E_index,
            z_index,
            params,
            False,
            npoints,
            ndim=ndim,
            nsi=nsi,
            save=False,
        )
        P_amam = generate_probabilities(
            "m",
            "m",
            Et,
            zr,
            E_index,
            z_index,
            params,
            True,
            npoints,
            ndim=ndim,
            nsi=nsi,
            save=False,
        )
        Pem = generate_probabilities(
            "e",
            "m",
            Et,
            zr,
            E_index,
            z_index,
            params,
            False,
            npoints,
            ndim=ndim,
            nsi=nsi,
            save=False,
        )
        P_aeam = generate_probabilities(
            "e",
            "m",
            Et,
            zr,
            E_index,
            z_index,
            params,
            True,
            npoints,
            ndim=ndim,
            nsi=nsi,
            save=False,
        )
        if tau:
            Pmt = generate_probabilities(
                "m",
                "t",
                Et,
                zr,
                E_index,
                z_index,
                params,
                False,
                npoints,
                ndim=ndim,
                nsi=nsi,
                save=False,
            )
            P_amat = generate_probabilities(
                "m",
                "t",
                Et,
                zr,
                E_index,
                z_index,
                params,
                True,
                npoints,
                ndim=ndim,
                nsi=nsi,
                save=False,
            )
            Pmm = Pmm + 0.1739 * Pmt
            P_amam = P_amam + 0.1739 * P_amat

        flux_e = get_flux("e", Et_mesh, zr_mesh, interp_flux)
        flux_ebar = get_flux("ebar", Et_mesh, zr_mesh, interp_flux)
        integrand = (
            aeff_m * flux_m * Pmm
            + aeff_mbar * flux_mbar * P_amam
            + aeff_m * flux_e * Pem
            + aeff_mbar * flux_ebar * P_aeam
        )
    else:
        integrand = aeff_m * flux_m + aeff_mbar * flux_mbar
    integrand *= resolution_gaussian * 2 * np.pi * 240747841

    return integrate(integrand, "simps", Et, zr, Er)


def event_wrapper(param_list):
    E_index, z_index, alpha, params, npoints, null, spectral, tau, ndim = (
        param_list[0],
        param_list[1],
        param_list[2],
        param_list[3],
        param_list[4],
        param_list[5],
        param_list[6],
        param_list[7],
        param_list[8],
    )
    return get_events(
        E_index=E_index,
        z_index=z_index,
        params=params,
        npoints=npoints,
        alpha=alpha,
        null=null,
        spectral_shift_parameters=spectral,
        tau=tau,
        ndim=ndim,
    )


def sim_events(
    alpha,
    npoints,
    params=ic_params,
    null=False,
    multi=True,
    spectral_shift=[False, 2e3, 0.02],
    tau=False,
    nsi=False,
    ndim=4,
):
    """
    if nsi:
        E_offset = 0 #For nsi, include all bins
    else:
        E_offset = 3 # For non-nsi, exclude bottom 3 bins
    """
    E_offset = 0
    res = np.empty((13 - E_offset, 20))
    E_z_combinations = []
    for E_bin in range(E_offset, 13):
        for z_bin in range(20):
            if multi:
                E_z_combinations.append(
                    [E_bin, z_bin, alpha, params, npoints, null, spectral_shift, tau]
                )
            if not multi:
                res[E_bin - E_offset][z_bin] = event_wrapper(
                    [
                        E_bin,
                        z_bin,
                        alpha,
                        params,
                        npoints,
                        null,
                        spectral_shift,
                        tau,
                        ndim,
                    ]
                )
    if multi:
        p = Pool()
        res = p.map(event_wrapper, E_z_combinations)

    res_array = np.array(res).reshape(-1, 20)
    return res_array


def wrap(p):
    return sim_events(0.99, 25, multi=True, params=p)


def spectral_shift_factor(E, E_pivot=2e3, delta_gamma=0.02):
    return (E / E_pivot) ** -delta_gamma


interp_flux, interp_aeff, energy_resolution_models = get_interpolators()
if __name__ == "__main__":
    print(get_events(5, 11, 0.99, 13))
