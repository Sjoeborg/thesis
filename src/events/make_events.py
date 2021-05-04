import sys, pickle, argparse
from tqdm.contrib.concurrent import process_map
import numpy as np
if __name__ == '__main__':
    sys.path.append('./src/data')
    sys.path.append('./src/events')
    sys.path.append('./src/probability')
from DC.event_processing import get_param_list, list_of_params_nsi
from PINGU.main import get_events as PINGU_events
from DC.main import get_events as DC_events
from IC.main import sim_events as IC_events
from functions import nufit_params_nsi, nufit_params_nsi_IO

parser = argparse.ArgumentParser()
parser.add_argument('-dm31N', default=1, type=int)
parser.add_argument('-th23N', default=1, type=int)

parser.add_argument('-ett', default=5e-2, type=float)
parser.add_argument('-ettN', default=1, type=int)
parser.add_argument('-emt', default=2e-2, type=float)
parser.add_argument('-emtN', default=1, type=int)

parser.add_argument('-eem', default=2e-1, type=float)
parser.add_argument('-eemN', default=1, type=int)
parser.add_argument('-eet', default=2e-1, type=float)
parser.add_argument('-eetN', default=1, type=int)

parser.add_argument('-s', default = 0, type=int)
parser.add_argument('-sT', default = 1, type=int)
parser.add_argument('-IO', action='store_true')
parser.add_argument('-IC', action='store_true')
parser.add_argument('-DC', action='store_true')
parser.add_argument('-PINGU', action='store_true')
args = parser.parse_args()


def precompute_probs(args_tuple):
    i,j,params = args_tuple
    if args.PINGU:
        res_track = PINGU_events(Ebin=i,zbin=j,params=params,pid=1,nsi=True, save=False)
        res_cascade =PINGU_events(Ebin=i,zbin=j,params=params,pid=0,nsi=True, save=False)
    elif args.DC:
        res_track = DC_events(Ebin=i,zbin=j,params=params,pid=1,nsi=True, no_osc=False, save=False)
        res_cascade =DC_events(Ebin=i,zbin=j,params=params,pid=0,nsi=True, no_osc=False, save=False)
    return np.array([res_cascade, res_track])

if __name__ == '__main__':
    assert args.PINGU or args.DC or args.IC
    dm31_range, th23_range,ett_range, emt_range, eem_range, eet_range = get_param_list(args.dm31N, args.th23N, 
                                                                                       args.ett, args.ettN, 
                                                                                       args.emt, args.emtN, 
                                                                                       args.eem, args.eemN, 
                                                                                       args.eet, args.eetN, 
                                                                                       args.IO)

    print('dm:', dm31_range)
    print('th:', th23_range)
    print('ett:', ett_range)
    print('emt:', emt_range)
    print('eem:', eem_range)
    print('eet:', eet_range)
    param_dict = nufit_params_nsi_IO if args.IO else nufit_params_nsi
    ordering = 'IO' if args.IO else 'NO'
    param_list = list_of_params_nsi(param_dict, dm31_range, th23_range, ett_range, emt_range, eem_range, eet_range)
    param_tuple = [(i,j,p) for p in param_list for i in range(8) for j in range(8)]
    
    if args.PINGU:
        H1_events_list = process_map(precompute_probs, param_tuple)
        H1 = np.array(H1_events_list)
        H1 = H1.reshape(len(param_list),8,8,2)
        H1 = np.swapaxes(H1,1,3) #Put pid on second index
        H1 = np.swapaxes(H1,2,3) #Swap e and z bins
        pickle.dump(H1,open(f'./pre_computed/H1_{ordering}_PINGU_{len(dm31_range)}x{len(th23_range)}x{len(ett_range)}x{len(emt_range)}x{len(eem_range)}x{len(eet_range)}.p','wb'))
    elif args.DC:
        H1_events_list = process_map(precompute_probs, param_tuple)
        H1 = np.array(H1_events_list)
        H1 = H1.reshape(len(param_list),8,8,2)
        H1 = np.swapaxes(H1,1,3) #Put pid on second index
        H1 = np.swapaxes(H1,2,3) #Swap e and z bins
        pickle.dump(H1,open(f'./pre_computed/H1_{ordering}_DC_{len(dm31_range)}x{len(th23_range)}x{len(ett_range)}x{len(emt_range)}x{len(eem_range)}x{len(eet_range)}.p','wb'))
    if args.IC:
        data = [(0.99, 13,p, False,False, [False, 0, 0],True, True,3) for p in param_list]
        H1_events_list = process_map(IC_events, data)
        pickle.dump(H1_events_list,open(f'./pre_computed/H1_{ordering}_IC_N13_{len(dm31_range)}x{len(th23_range)}x{len(emt_range)}.p','wb'))