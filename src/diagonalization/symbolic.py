# Eigenvector substitution (reduction to 2gen)
import sympy as sp
from functions import *
import numpy as np
from sympy.matrices import matrix_multiply_elementwise
import matplotlib.pyplot as plt

# sp.init_printing(use_latex="mathjax")
import matplotlib
from numerical import P_num_over_E_single as P_num
import pickle

matplotlib.rc("text", usetex=False)
matplotlib.rc("text.latex", preamble=r"\usepackage{amsmath}")

m_21, m_31, m_41 = sp.symbols("dm_21 dm_31 dm_41")
th_12, th_13, th_23, th_34, th_23, th_14, th_24 = sp.symbols(
    "theta_12 theta_13 theta_23 theta_34 theta_23 theta_14 theta_24"
)
E = sp.symbols("E")
d_ij = sp.symbols("delta_ij")
(
    V_cc,
    V_nc,
) = sp.symbols("V_cc,V_nc")
(
    A_cc,
    A_nc,
) = sp.symbols("A_cc,A_nc")
param_dict_num = param_dict  # Regular dict with parameter values
param_dict_sym = {
    "theta_12": th_12,  # Symbolic dict with sympy symbols
    "theta_13": th_13,
    "theta_23": th_23,
    "theta_34": th_34,
    "theta_24": th_24,
    "theta_14": th_14,
    "delta_ij": d_ij,
    "dm_21": m_21,
    "dm_31": m_31,
    "dm_41": m_41,
}


def V_ijab(i, j, a, b, A=0, params=param_dict_sym):  # Blennow 78:807 delta_ij = 0
    if a == b:
        if a == i or a == j:
            return sp.cos(theta(i, j, A, params=params))
        else:
            return 1
    else:
        if a == i and b == j:
            return sp.sin(-theta(i, j, A, params=params))
        elif a == j and b == i:
            return -sp.sin(-theta(i, j, A, params=params))
        else:
            return 0


def V_matrix(i, j, A=0, n=0, params=param_dict_sym):  # Blennow 78:807
    result = sp.zeros(3 + n, 3 + n)
    for a in range(1, 4 + n):
        elem1 = [V_ijab(i, j, a, b, A, params=params) for b in range(1, 3 + n + 1)]
        result[a - 1] = elem1
    return result


def U_nu(ndim, A=0, params=param_dict_sym):
    if ndim == 4:
        return (
            V_matrix(3, 4, A, 1, params=params)
            @ V_matrix(2, 4, A, 1, params=params)
            @ V_matrix(1, 4, A, 1, params=params)
            @ V_matrix(2, 3, A, 1, params=params)
            @ V_matrix(1, 3, A, 1, params=params)
            @ V_matrix(1, 2, A, 1, params=params)
        )
    elif ndim == 3:
        return (
            V_matrix(2, 3, A, 0, params=params)
            @ V_matrix(1, 3, A, 0, params=params)
            @ V_matrix(1, 2, A, 0, params=params)
        )
    elif ndim == 2:  # sin blir -sin jmf med U i constants
        return V_matrix(1, 2, A, -1, params=params)


def get_H_(En, params, mtype, anti):
    """
    En in [GeV]
    """
    if mtype == "full":
        U = U_nu(4, params=params)
    elif mtype == "single":
        U = (
            V_matrix(3, 4, 0, 1, params=params)
            @ V_matrix(2, 4, 0, 1, params=params)
            @ V_matrix(1, 4, 0, 1, params=params)
        )
    if anti:
        # A = -sp.diag(2*E*V_cc, 0 , 0, 2*E*V_nc)
        M = sp.diag(0, params["dm_21"], params["dm_31"], -params["dm_41"])
    else:
        M = sp.diag(0, params["dm_21"], params["dm_31"], params["dm_41"])
    A = sp.diag(2 * E * V_cc, 0, 0, 2 * E * V_nc)
    H = 1 / (2 * E * 1e9) * (U @ M @ U.T + A * 1e18)
    H = H.subs(E, En)
    return H


def diagonalize_H_(H, anti):
    # TODO: clean up
    UM_unsorted, HM_unsorted = H.diagonalize(
        reals_only=False, sort=True, normalize=True
    )
    HM_unsorted = np.diag(np.asarray(HM_unsorted))
    UM_unsorted = np.asarray(UM_unsorted)
    if anti:
        UM_sorted = np.array(
            [UM_unsorted[:, 0], UM_unsorted[:, 2], UM_unsorted[:, 1], UM_unsorted[:, 3]]
        )
        HM_sorted = np.array(
            [HM_unsorted[0], HM_unsorted[1], HM_unsorted[2], HM_unsorted[3]]
        )
    else:
        UM_sorted = np.array(
            [UM_unsorted[:, 1], UM_unsorted[:, 2], UM_unsorted[:, 3], UM_unsorted[:, 0]]
        )
        HM_sorted = np.array(
            [HM_unsorted[1], HM_unsorted[2], HM_unsorted[3], HM_unsorted[0]]
        )
    return UM_sorted, HM_sorted


def get_dmM_(HM, En):
    """
    En in [GeV]
    """
    eigenvals = HM * np.array([1, 1, 1, 1])
    dmM_res = 2 * En * 1e9 * eigenvals
    return dmM_res


def get_Usquared_(UM):
    UM_squared = UM * UM
    UMe = UM_squared[:, 0]
    UMm = UM_squared[:, 1]
    UMt = UM_squared[:, 2]
    UMs = UM_squared[:, 3]
    return UMe, UMm, UMt, UMs


def get_dmM(En, params, mtype, anti):
    """
    En in [GeV]
    """
    H = get_H_(En, params=params, mtype=mtype, anti=anti)
    UM, HM = diagonalize_H_(H, anti)
    dmM = get_dmM_(HM, En)
    return dmM


def get_UMsq(En, params, mtype, anti):
    """
    En in [GeV]
    """
    H = get_H_(En, params=params, mtype=mtype, anti=anti)
    UM, HM = diagonalize_H_(H, anti)
    UMsq = get_Usquared_(UM)
    return UMsq


def get_UM(En, params, mtype, anti):
    """
    En in [GeV]
    """
    H = get_H_(En, params=params, mtype=mtype, anti=anti)
    UM, HM = diagonalize_H_(H, anti)
    return UM


def P_an_single(alpha, beta, En, L, params, anti):
    dmM = np.asarray(get_dmM(En, params, "single", anti=anti)).astype(np.float64)
    UM_single = np.asarray(get_UM(En, params, "single", anti=anti)).astype(np.float64)
    Ufrom = UM_single[alpha]
    Uto = UM_single[beta]
    params_with_dmM = params.copy()
    params_with_dmM.update({"dm_21": dmM[1], "dm_31": dmM[2], "dm_41": dmM[3]})
    P_list = []
    for k in range(0, 4):
        for j in range(0, k):  # 1605.08607 eq3, giunti 7.38 and 7.51
            m = dm(k + 1, j + 1, params=params_with_dmM)
            # if not np.isclose(m,0.):
            #    print(f'dmM{k+1}{j+1}:', m)
            U_product = Ufrom[k] * Uto[k] * Ufrom[j] * Uto[j]
            real_part = U_product * np.sin(GeV2tokm1 * m * L / (4 * En)) ** 2
            P_list.append(real_part)
    if (Ufrom == Uto).all():
        return 1 - 4 * np.sum(P_list)
    else:
        return -4 * np.sum(P_list)
    # Define numeric parameters


full_params = {
    "dm_21": param_dict_num["dm_21"],
    "dm_31": param_dict_num["dm_31"],
    "dm_41": param_dict_num["dm_41"],
    "delta_ij": 0,
    "theta_12": param_dict_num["theta_12"],
    "theta_13": param_dict_num["theta_13"],
    "theta_23": param_dict_num["theta_23"],
    "theta_14": np.arcsin(np.sqrt(0)),
    "theta_24": np.arcsin(np.sqrt(0.04)) / 2,
    "theta_34": np.arcsin(np.sqrt(0)),
}
single_params = {
    "dm_21": 0,
    "dm_31": 0,
    "dm_41": param_dict_num["dm_41"],
    "delta_ij": 0,
    "theta_12": param_dict_num["theta_12"],
    "theta_13": param_dict_num["theta_13"],
    "theta_23": param_dict_num["theta_23"],
    "theta_14": np.arcsin(np.sqrt(0)),
    "theta_24": np.arcsin(np.sqrt(0.04)) / 2,
    "theta_34": np.arcsin(np.sqrt(0)),
}


# V_cc = np.sqrt(2) * GF * 0.5 * N_A * 8.44 * (1/GeVtocm1)**3 #GeV
# V_nc = V_cc / 2 #GeV
def to_num(symb):
    try:
        return symb.subs(m_41, single_params["dm_41"]).subs(
            th_24, single_params["theta_24"].subs(E, En)
        )
    except AttributeError:
        return symb.subs(m_41, single_params["dm_41"]).subs(
            th_24, single_params["theta_24"]
        )


symb_params = param_dict_sym.copy()
symb_params.update(
    {
        "theta_12": param_dict_num["theta_12"],
        "theta_13": param_dict_num["theta_13"],
        "theta_23": param_dict_num["theta_23"],
        "theta_14": 0,
        "theta_24": param_dict_sym["theta_24"],
        "theta_34": param_dict_sym["theta_34"],
        "dm_21": 0,
        "dm_31": 0,
        "dm_41": param_dict_sym["dm_41"],
    }
)


anti = True
L = 2 * r_earth


def get_H_A(En, params, mtype, anti):
    """
    En in [GeV]
    """
    if mtype == "full":
        U = U_nu(4, params=params)
    elif mtype == "single":
        U = (
            V_matrix(3, 4, 0, 1, params=params)
            @ V_matrix(2, 4, 0, 1, params=params)
            @ V_matrix(1, 4, 0, 1, params=params)
        )
    if anti:
        # A = -sp.diag(2*E*V_cc, 0 , 0, 2*E*V_nc)
        M = sp.diag(0, params["dm_21"], params["dm_31"], -params["dm_41"])
    else:
        M = sp.diag(0, params["dm_21"], params["dm_31"], params["dm_41"])
    A = sp.diag(
        2 * np.sqrt(2) * GF * 0.5 * N_A * 8.44 * (1 / GeVtocm1) ** 3 * En,
        0,
        0,
        2 * np.sqrt(2) * GF * 0.5 * N_A * 8.44 * (1 / GeVtocm1) ** 3 / 2 * En,
    )
    H = 1 / (2 * E * 1e9) * (U @ M @ U.T + A * 1e18)
    H = H.subs(E, En)
    return H


H_block = get_H_A(E, symb_params, "single", True)[1:4, 1:4]

a = H_block.diagonalize(reals_only=False, sort=False, normalize=True)
pickle.dump(a, open("diagonalized_H.npy", "wb"))
