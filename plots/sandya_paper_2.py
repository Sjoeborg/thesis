#File for 0709.1937
from main import *
from constants import param_dict,V,theta,dm,GeV2tokm1
from analytical import P_an
from numerical import P_num_over_E_single
from multiprocessing import Pool,Process
import csv
import time
def fig4(type, npoints):
    L = 2*r_earth
    L_range = np.linspace(0,L,500)
    
    param_dict.update({'dm_31': 2.5e-3,
                       'dm_21': 8e-5,
                       'theta_23': np.arcsin(np.sqrt(0.5)),
                       'theta_13': 0,  
                       'theta_24': np.arcsin(np.sqrt(0.04)),
                       'theta_12': np.arcsin(np.sqrt(0.3)),
                       'delta_ij': 0,
                       'dm_41': -1,
                       'theta_14': 0,
                       'theta_34': np.arcsin(np.sqrt(0.04))
                        }) 
    #plt.title(rf'$E = {E}, dm = {param_dict["dm_41"]}, theta_24 = {param_dict["theta_24"]}$')
    E_range= np.linspace(0.1e3,10e3, npoints)
    x,P = P_over_E_parameter(type=type,flavor_from='m', flavor_to='m',E_range=E_range, L=L, earth_start = 0, ndim = 4,vacuum=False,param_dict_list=[param_dict],eval_at=L)

    x,P_vac = P_over_E_parameter(type=type,flavor_from='m', flavor_to='m', E_range=E_range, L=L,  earth_start = 0, ndim = 4,vacuum=True,param_dict_list=[param_dict],eval_at=L)

    plt.subplot(221)
    plt.tight_layout()
    plt.xscale('log')
    plt.title(r'$P_{\mu\mu}$')
    plt.plot(x[0],P[0][1], color='black')
    plt.plot(x[0],P_vac[0][1], linestyle='dotted', color='darkviolet')
    plt.ylim((0,1))
    plt.xlim((0.1e3,10e3))
    plt.subplot(222)
    plt.title(r'$P_{\mu\tau}$')
    plt.xscale('log')
    plt.tight_layout()
    plt.plot(x[0],P[0][2], color='black')
    plt.plot(x[0],P_vac[0][2], linestyle='dotted', color='darkviolet')
    plt.xlim((0.1e3,10e3))
    plt.ylim((0,1))
    plt.subplot(224)
    plt.title(r'$P_{\mu s}$')
    plt.plot(x[0],P[0][3], color='black', label='neutrinos')
    plt.plot(x[0],P_vac[0][3], linestyle='dotted', color='darkviolet', label='vacuum')
    plt.xlim((0.1e3,10e3))
    plt.ylim((0,1))
    plt.suptitle(rf'(3+1) matter oscillations using PREM, $dm41 = {param_dict["dm_41"]} eV^2, L = 2R_e$')
    plt.xscale('log')
    plt.xlabel('E (GeV)')
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig('plots/fig4.png')

    with open("plots/out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(len([param_dict])):
            writer.writerows([x[i],P[i][0],P[i][1],P[i][2],P[i][3]])
    with open('plots/out.csv') as f, open('plots/fig4.csv', 'w') as fw: #transpose csv
        csv.writer(fw, delimiter=',').writerow(['GeV','Pmm','Pmt','Pms'])
        csv.writer(fw, delimiter=',').writerows(zip(*csv.reader(f, delimiter=',')))
def fig6(type, npoints):
    L = 2*r_earth
    
    param_dict.update({'dm_31': 2.5e-3,
                       'dm_21': 8e-5,
                       'theta_23': np.arcsin(np.sqrt(0.5)),
                       'theta_13': 0,  
                       'theta_24': np.arcsin(np.sqrt(0.04)),
                       'theta_12': np.arcsin(np.sqrt(0.3)),
                       'delta_ij': 0,
                       'dm_41': -1,
                       'theta_14': 0,
                       'theta_34': np.arcsin(np.sqrt(0.04))
                        }) 
    th_24 = [np.arcsin(np.sqrt(0.04)), np.arcsin(np.sqrt(0.02)), np.arcsin(np.sqrt(0.04)),np.arcsin(np.sqrt(0.02))]
    th_34 = [np.arcsin(np.sqrt(0.04)), np.arcsin(np.sqrt(0.02)), np.arcsin(np.sqrt(0.00)),np.arcsin(np.sqrt(0.00))]

    dicts=[]
    for i in range(4):
        dicts.append(dict(param_dict, theta_24=th_24[i], theta_34=th_34[i]))
    E_range= np.linspace(0.1e3,10e3, npoints)
    x,P = P_over_E_parameter(type=type,flavor_from='m', flavor_to='m', E_range=E_range,L=L,earth_start = 0, ndim = 4,vacuum=False,param_dict_list=dicts)


    '''
    plt.subplot(221)
    plt.tight_layout()
    plt.title(r'$P_{\mu\mu}$')
    plt.plot(x[0]/1e3,P[0][1], color='blue', linestyle='dashed')
    plt.plot(x[1]/1e3,P[1][1], linestyle='dashdot', color='red')
    plt.plot(x[2]/1e3,P[2][1], linestyle='dotted', color='green')
    plt.plot(x[3]/1e3,P[3][1], linestyle='solid', color='black')
    plt.xlim((0,10))
    plt.ylim((0,1))
    plt.subplot(222)
    plt.title(r'$P_{\mu\tau}$')
    plt.tight_layout()
    plt.plot(x[0]/1e3,P[0][2], color='blue', linestyle='dashed')
    plt.plot(x[1]/1e3,P[1][2], linestyle='dashdot', color='red')
    plt.plot(x[2]/1e3,P[2][2], linestyle='dotted', color='green')
    plt.plot(x[3]/1e3,P[3][2], linestyle='solid', color='black')
    plt.xlim((0,10))
    plt.ylim((0,1))
    plt.subplot(224)
    plt.title(r'$P_{\mu s}$')
    plt.plot(x[0]/1e3,P[0][3], color='blue', linestyle='dashed', label= 's24 = 0.04, s34 = 0.04')
    plt.plot(x[1]/1e3,P[1][3], linestyle='dashdot', color='red', label= 's24 = 0.02, s34 = 0.02')
    plt.plot(x[2]/1e3,P[2][3], linestyle='dotted', color='green', label= 's24 = 0.04, s34 = 0.00')
    plt.plot(x[3]/1e3,P[3][3], linestyle='solid', color='black', label= 's24 = 0.02, s34 = 0.00')
    plt.ylim((0,1))
    plt.xlim((0,10))
    plt.suptitle(rf'Fig 6: (3+1) matter oscillations using PREM, dm41 = ${param_dict["dm_41"]} eV^2, L = 2R_e$')
    plt.xlabel('E (TeV)')
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig('plots/fig6.png')

    assert(np.all([x[0],x[1],x[2],x[3]]))
    with open("plots/out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([
            x[0],
            P[0][1],P[0][2],P[0][3],
            P[1][1],P[1][2],P[1][3],
            P[2][1],P[2][2],P[2][3],
            P[3][1],P[3][2],P[3][3]])
    
    with open('plots/out.csv') as f, open('plots/fig6.csv', 'w') as fw: #transpose csv
        csv.writer(fw, delimiter=',').writerow(['GeV',
        'Pmm_s24004_s34004','Pmt_s24004_s34004','PmsP_s24004_s34004'
        'Pmm_s24002_s34002','Pmt_s24002_s34002','PmsP_s24002_s34002',
        'Pmm_s24004_s34000','Pmt_s24004_s34000','PmsP_s24004_s34000',
        'Pmm_s24002_s34000','Pmt_s24002_s34000','PmsP_s24002_s34000'
        ])
        csv.writer(fw, delimiter=',').writerows(zip(*csv.reader(f, delimiter=',')))
    '''
def fig6_mt(type,E_range, s_34):
    th_34 = np.arcsin(np.sqrt(s_34))

    param_dict.update({'dm_31': 2.5e-3,
                       'dm_21': 8e-5,
                       'theta_23': np.arcsin(np.sqrt(0.5)),
                       'theta_13': 0,  
                       'theta_24': np.arcsin(np.sqrt(0.04)),
                       'theta_12': np.arcsin(np.sqrt(0.3)),
                       'delta_ij': 0,
                       'dm_41': -1,
                       'theta_14': 0,
                       'theta_34': np.arcsin(np.sqrt(0.04))
                        }) 
    dicts = [dict(param_dict, theta_34=t_34) for t_34 in th_34]

    x, y= P_over_E_parameter(type=type, flavor_from='m', param_dict_list=dicts, E_range=E_range,ndim = 4)

    for i in range(len(th_34)):
        plt.plot(x[i],y[i][2], label = f'$\\theta = {np.round(s_34[i],3)}$')
    plt.xlim((np.min(E_range), np.max(E_range)))
    plt.ylim((0,1))
    plt.legend()
    plt.show()
def fig7(type, E_range, s_range):
    th_34 = np.arcsin(np.sqrt(s_34))

    param_dict.update({'dm_31': 2.5e-3,
                       'dm_21': 8e-5,
                       'theta_23': np.arcsin(np.sqrt(0.5)),
                       'theta_13': 0,  
                       'theta_24': np.arcsin(np.sqrt(0.04)),
                       'theta_12': np.arcsin(np.sqrt(0.3)),
                       'delta_ij': 0,
                       'dm_41': -1,
                       'theta_14': 0,
                       'theta_34': np.arcsin(np.sqrt(0.04))
                        }) 
    dicts = [dict(param_dict, theta_34=t_34) for t_34 in th_34]
    th=[]
    for p in dicts:
        th.append([theta(3,4,2 * E * np.sqrt(2) * GF * Y * N_A * 8.44 * (1/GeVtocm1)**3, p) for E in E_range])
    
    for i in range(len(s_34)):
        plt.plot(E_range,np.sin(th[i])**2, label = f'$s34 = {np.round(s_34[i],3)}$')
    plt.xlim((np.min(E_range), np.max(E_range)))
    #plt.ylim((0,1))
    plt.legend()
    plt.show()
    

def fig8(type, npoints):
    L= 2*r_earth
    param_dict.update({'dm_31': 2.5e-3,
                       'dm_21': 8e-5,
                       'theta_23': np.arcsin(np.sqrt(0.5)),
                       'theta_13': np.arcsin(np.sqrt(0.01)),  
                       'theta_24': np.arcsin(np.sqrt(0.034)),
                       'theta_12': np.arcsin(np.sqrt(0.3)),
                       'delta_ij': 0,
                       'dm_41': -0.87,
                       'dm_51': -1.91,
                       'theta_14': np.arcsin(np.sqrt(0.014)),
                       'theta_15': np.arcsin(np.sqrt(0.012)),
                       'theta_25': np.arcsin(np.sqrt(0.008))
                        })
    th_34 = [np.arcsin(np.sqrt(0.01)), np.arcsin(np.sqrt(0.00))]
    th_35 = [np.arcsin(np.sqrt(0.01)), np.arcsin(np.sqrt(0.00))]
    th_45 = [np.arcsin(np.sqrt(0.01)), np.arcsin(np.sqrt(0.00))]
    dicts=[]
    for i in range(2):
        dicts.append(dict(param_dict, theta_34=th_34[i], theta_35=th_35[i], theta_45=th_45[i]))

    E_range= np.linspace(0.1e3,10e3, npoints)
    x,P = P_over_E_parameter(type=type,flavor_from='m', flavor_to='m', E_range=E_range,L=L,earth_start = 0, ndim = 5,vacuum=False,param_dict_list=dicts)


    #fig, ax = plt.subplots(2,2)

    plt.plot(x[0],P[0][3])
    plt.plot(x[1],P[1][3])
    plt.plot(x[0],P[0][4])
    plt.plot(x[1],P[1][4])
    plt.xscale('log')
    #plt.show()
    plt.savefig('plots/fig7.png')
    assert(np.all([x[0],x[1]])) #check if we can use x[0] as the E for all params below
    with open("plots/out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([
            x[0],
            P[0][3],P[1][3],P[0][4], P[1][4]])
    with open('plots/out.csv') as f, open('plots/fig7.csv', 'w') as fw: #transpose csv
        csv.writer(fw, delimiter=',').writerow(['GeV','Pms1_s001','Pms1_s000','Pms2_s001','Pms2_s000'])
        csv.writer(fw, delimiter=',').writerows(zip(*csv.reader(f, delimiter=',')))

if __name__ == '__main__':
    
    E_range = np.linspace(0.1e3,10e3,20)
    s_34 = np.linspace(0.01,0.04,3)
    '''
    start_time = time.time()
    fig7('local',E_range,s_34)
    print("--- %s seconds ---" % (time.time() - start_time))
    '''
    th_24 = np.arcsin(np.sqrt(0.04))
    th_34 = np.arcsin(np.sqrt(0.04))
    E = 10e3
    A = 2 * E * np.sqrt(2) * GF * Y * N_A * 8.44 * (1/GeVtocm1)**3 * 1e18
    param_dict.update({'dm_41':-1,
    'theta_24': np.arcsin(np.sqrt(0.04)), 'theta_34': np.arcsin(np.sqrt(0.04))})

    x, P= P_over_E_parameter(type='local', flavor_from='m', param_dict_list=[param_dict], E_range=E_range,ndim = 4)

    _,Pe= P_over_E_parameter(type='local', flavor_from='e', param_dict_list=[param_dict], E_range=E_range,ndim = 4)
    
    P_mt = P[0][2]
    P_ms = P[0][3]
    P_me = P[0][0]
    P_ee = Pe[0][0]

    s_34 = np.sin(np.arctan(np.sqrt(P_mt/P_ms)))**2
    s_24 = P_me/(1-P_ee)
    plt.plot(x[0],s_34)
    plt.plot(x[0],s_24)
    plt.show()

    
    #fig7('local', E_range,s_34)




    #### Speed tests ######
    #n1-highcpu 8x2 fig7('cloud',100) 34 s
    #c2 8x1 fig7('cloud',100) 43 s
    #c2 4x2 fig7('cloud',100) 49 s
    #c2 4x2 fig6('cloud',100) 37 s

    #n1-highcpu 4x4 fig6('cloud',100) 59 s
    #n1-highcpu 4x4 fig7('cloud',100) 1 m 17 s

    #n2-highcpu 8x1 fig7('cloud',100) 55 s
    #n2-highcpu 8x1 fig6('cloud',100) 15 s
    #n2-highcpu 8x1 fig6('cloud',5000) 12m 30s
    #n2-highcpu 4x2 fig7('cloud',100) 1m 3s
    #n2-highcpu 4x2 fig6('cloud',100) 45 s
