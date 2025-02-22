import sys, pickle, argparse
from tqdm.contrib.concurrent import process_map
import numpy as np
sys.path.append("../..")
from src.events.DC.event_processing import get_param_list, list_of_params_nsi
from src.events.PINGU.main import get_events as PINGU_events
from src.events.DC.main import get_events as DC_events
from src.events.IC.main import get_events as IC_events
from src.probability.functions import nufit_params_nsi, nufit_params_nsi_IO
from sklearn.exceptions import InconsistentVersionWarning
import warnings
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

# H1_NO_DC_5x5x9x1x1x1
# H1_{ordering}_DC_{args.pid}_{len(dm31_range)}x{len(th23_range)}x{len(ett_range)}x{len(emt_range)}x{len(eem_range)}x{len(eet_range)}
parser = argparse.ArgumentParser()
parser.add_argument("-dm31N", default=1, type=int)
parser.add_argument("-th23N", default=1, type=int)

parser.add_argument("-ettFrom", default=-7e-2, type=float)
parser.add_argument("-ettTo", default=7e-2, type=float)
parser.add_argument("-ettN", default=1, type=int)
parser.add_argument("-emtFrom", default=-3e-2, type=float)
parser.add_argument("-emtTo", default=3e-2, type=float)
parser.add_argument("-emtN", default=1, type=int)

parser.add_argument("-eemFrom", default=-3e-1, type=float)
parser.add_argument("-eemTo", default=3e-1, type=float)
parser.add_argument("-eemN", default=1, type=int)
parser.add_argument("-eetFrom", default=-3e-1, type=float)
parser.add_argument("-eetTo", default=3e-1, type=float)
parser.add_argument("-eetN", default=1, type=int)

parser.add_argument("-s", default=0, type=int)
parser.add_argument("-sT", default=1, type=int)
parser.add_argument("-pid", type=int)
parser.add_argument("-IO", action="store_true")
parser.add_argument("-IC", action="store_true")
parser.add_argument("-DC", action="store_true")
parser.add_argument("-PINGU", action="store_true")
args = parser.parse_args()


def precompute_probs(args_tuple):
    if args.PINGU:
        i, j, params, pid = args_tuple
        res = PINGU_events(Ebin=i, zbin=j, params=params, pid=pid, nsi=True, save=True)
    elif args.DC:
        i, j, params, pid = args_tuple
        res = DC_events(
            Ebin=i, zbin=j, params=params, pid=pid, nsi=True, no_osc=False, save=True
        )
    elif args.IC:
        i, j, a, N, p, spectral, null, tau, nsi, ndim = args_tuple
        res = IC_events(
            E_index=i,
            z_index=j,
            alpha=a,
            npoints=N,
            params=p,
            spectral_shift_parameters=spectral,
            null=null,
            tau=tau,
            nsi=nsi,
            ndim=ndim,
        )
    return np.array(res)


if __name__ == "__main__":
    assert args.PINGU or args.DC or args.IC
    dm31_range, th23_range, ett_range, emt_range, eem_range, eet_range = get_param_list(
        args.dm31N,
        args.th23N,
        (args.ettFrom, args.ettTo, args.ettN),
        (args.emtFrom, args.emtTo, args.emtN),
        (args.eemFrom, args.eemTo, args.eemN),
        (args.eetFrom, args.eetTo, args.eetN),
        args.IO,
    )
    dm31_range = np.array_split(dm31_range, args.sT)[args.s]
    print("dm:", dm31_range)
    print("th:", th23_range)
    print("ett:", ett_range)
    print("emt:", emt_range)
    print("eem:", eem_range)
    print("eet:", eet_range)
    param_dict = nufit_params_nsi_IO if args.IO else nufit_params_nsi
    ordering = "IO" if args.IO else "NO"
    param_list = list_of_params_nsi(
        param_dict, dm31_range, th23_range, ett_range, emt_range, eem_range, eet_range
    )
    param_tuple = [
        (i, j, p, args.pid) for p in param_list for i in range(8) for j in range(8)
    ]

    if args.PINGU:
        H1_events_list = process_map(precompute_probs, param_tuple, chunksize=4)
        H1 = np.array(H1_events_list)
        H1 = H1.reshape(len(param_list), 1, 8, 8)  # second axis is pid
        pickle.dump(
            H1,
            open(
                f"../../pre_computed/H1_{ordering}_PINGU_{args.pid}_{args.s}x{len(th23_range)}x{len(ett_range)}x{len(emt_range)}x{len(eem_range)}x{len(eet_range)}.p",
                "wb",
            ),
        )
    elif args.DC:
        H1_events_list = process_map(precompute_probs, param_tuple, chunksize=4)
        H1 = np.array(H1_events_list)
        H1 = H1.reshape(len(param_list), 1, 8, 8)  # second axis is pid
        pickle.dump(
            H1,
            open(
                f"../../pre_computed/H1_{ordering}_DC_{args.pid}_{len(dm31_range)}x{len(th23_range)}x{len(ett_range)}x{len(emt_range)}x{len(eem_range)}x{len(eet_range)}.p",
                "wb",
            ),
        )
    if args.IC:
        data = [
            (i, j, 0.99, 13, p, [False, 0, 0], False, True, True, 3)
            for p in param_list
            for i in range(13)
            for j in range(20)
        ]
        H1_events_list = process_map(precompute_probs, data, chunksize=20)
        H1_events_list = np.array(H1_events_list).reshape(len(param_list), 13, 20)
        pickle.dump(
            H1_events_list,
            open(
                f"../../pre_computed/H1_{ordering}_IC_N13_{len(dm31_range)}x{len(th23_range)}x{len(emt_range)}.p",
                "wb",
            ),
        )
