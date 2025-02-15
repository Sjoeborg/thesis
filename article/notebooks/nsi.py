import sys, os

sys.path.append("./src")
sys.path.append("./../../src/data")
sys.path.append("./../../src/events")
sys.path.append("./../../src/probability")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src.data.IC.importer import *
from src.data.IC.processer import *
from src.probability.functions import nufit_params_nsi
from src.plotter import (
    P_over_E_parameter,
    nsi_oscillogram,
    nsi_flux_oscillogram,
    save,
    savethesis,
)
from src.events.IC.event_processing import *
import pickle

# from matplotlib.colors import CenteredNorm
plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams["figure.dpi"] = 100
matplotlib.rc("text", usetex=True)
matplotlib.rc("text.latex", preamble=r"\usepackage{amsmath}")
np.set_printoptions(linewidth=200)

IC_range = np.logspace(np.log10(500), 4, 50)
DC_range = np.logspace(np.log10(5.6), np.log10(56), 50)
# ett_range = [-0.1,0,0.1]#np.linspace(-1e-1,1e-1,3)
emm_range = [-1e-1, 0, 1e-1]  # np.linspace(-1e-1,1e-1,3)
emt_strong_range = [-5e-2, 0, 5e-2]
params = nufit_params_nsi.copy()
anti = True
s24_range = [0.032]
params.update(
    {
        "theta_24": np.arcsin(np.sqrt(0.032)) / 2,
        "dm_41": 0.93,
        "theta_34": np.arcsin(np.sqrt(0)) / 2,
    }
)
emt_strong_params = list_of_params_nsi(params, s24_range, [0], emt_strong_range)

emt_weak_range = [-1e-2, 0, 1e-2]
emt_weak_params = list_of_params_nsi(params, s24_range, [0], emt_weak_range)
zenith = -1


z_range = np.linspace(-1, 0, 500)
IC_range = np.logspace(
    np.log10(284), np.log10(103821), 500
)  # get_Etrue(model, 3,0.995,0.995,E_index=0)[0][0] #get_Etrue(model, 3,0.995,0.995,E_index=12)[0][2]
DC_range = np.logspace(
    np.log10(1.9), np.log10(210), 500
)  # MC2018_DC().query('pid==1').query('abs(pdg) == 14').query('reco_energy == reco_energy.min()')['true_energy'].quantile(0.005)

flux_emt_IC_track = np.array(
    nsi_flux_oscillogram(IC_range, z_range, emt_strong_params[0])
)
flux_emt_DC_track = np.array(
    nsi_flux_oscillogram(DC_range, z_range, emt_strong_params[0])
)

pickle.dump(flux_emt_IC_track, open("flux_emt_IC_track.p", "wb"))
pickle.dump(flux_emt_DC_track, open("flux_emt_DC_track.p", "wb"))


z_range = np.linspace(-1, 0, 500)
E_range = np.logspace(0, np.log10(60), 500)

Pex_neg, Pmx_neg, Paeax_neg, Pamax_neg = nsi_oscillogram(
    E_range=E_range, z_range=z_range, params=emt_weak_params[0]
)
Pex_pos, Pmx_pos, Paeax_pos, Pamax_pos = nsi_oscillogram(
    E_range=E_range, z_range=z_range, params=emt_weak_params[2]
)


pickle.dump(Pmx_neg, open("Pmx_neg.p", "wb"))
pickle.dump(Pmx_pos, open("Pmx_pos.p", "wb"))
