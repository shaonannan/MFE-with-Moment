import numpy as np
import itertools
import time
import utilities as util
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt

from internalpopulation import RecurrentPopulation
from externalpopulation import ExternalPopulation
from simulation import Simulation
from connection import Connection as Connection


"""
02/02/2018 version
edited by SYX

initialization and configuration
vflag for dawson function 
fin for \rho_{Eq}
"""


# intuitive network structure
Net_settings = {'hyp_num': 2,
                'xhyp':2,
                'yhyp':1,
                'xn':1,
                'yn':1, # 25 subpopulation each 20 cells 20 * 25 = 500 cells per hyp!
                # or 9 subpopulations and each 50 cells 9 * 50 = 450 cells per hyp!
                'nmax': 0,
                'Final_time':1024,
                'dt':0.1,
                'dv':1e-3}
Net_settings['nmax'] = Net_settings['hyp_num'] * Net_settings['xn'] * Net_settings['yn']
# here we use orientation and phase
Fun_map_settings = {'ori_num':1,
                   'phase_num':1}
# cell numbers within identical orientation hypercolumn
Cell_type_num = {'e':32,
                'i':24}
print(Net_settings['hyp_num'])

def create_network_population(Structure_Net,Functional_Map):
    #calculate total number of CG patches
    nmax    = Structure_Net['nmax']
    CGindex = np.arange(nmax)
    hypind  = np.arange(Structure_Net['hyp_num'])
    CGinx   = np.arange(Structure_Net['xn'])
    CGiny   = np.arange(Structure_Net['yn'])
    # Create populations:
    background_population_dict = {}
    internal_population_dict = {}
    for layer, celltype in itertools.product(CGindex, ['e', 'i']):    
        background_population_dict[layer, celltype] = 0
        internal_population_dict[layer, celltype] = 0

    return (Structure_Net,background_population_dict,internal_population_dict)


# Simulation settings:
t0 = 0.0
dt = Net_settings['dt']
tf = Net_settings['Final_time']
tfinal = tf
dv = 1e-3
Net_settings['dv'] = dv
verbose = True

update_method = 'approx'
approx_order = 1
tol  = 1e-14

def input_signal_stream(t_properties,Net_settings):
    (dt, tfinal) = t_properties

    ntt = int(tfinal/dt) # 20 to escape from error
    # Create base-visual stimuli
    External_stimuli_dict = np.ones(ntt)*0.150

    # Create populations:
    background_population_dict = {}
    internal_population_dict = {}
    CG_range = np.arange(Net_settings['nmax'])

    # base activities
    # show order
    id_order = 0
    for layer, celltype in itertools.product(CG_range, ['e', 'i']):  
        if celltype == 'e':
            External_stimuli_dict = np.ones(ntt)*0.45
        else:
            External_stimuli_dict = np.ones(ntt)*0.45
            
        background_population_dict[layer,celltype] = ExternalPopulation(External_stimuli_dict,dt, record=False)
        internal_population_dict[layer, celltype]  = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method,
                                approx_order=approx_order, tol=tol, hyp_idx = 0,ei_pop = celltype,NumCell = Cell_type_num[celltype])
        print('order:',id_order,' layer:',layer,' type:',celltype)
        id_order +=1
        
    population_list = list(background_population_dict.values()) + list(internal_population_dict.values())
    return (background_population_dict,internal_population_dict,population_list)

background_population_dict,internal_population_dict,population_list = input_signal_stream([dt,tfinal],Net_settings)

NE_source = Cell_type_num['e']
NI_source = Cell_type_num['i']

NPATCH = Net_settings['nmax']
DEE = np.zeros((NPATCH,NPATCH))
DEI = np.zeros((NPATCH,NPATCH))
DII = np.zeros((NPATCH,NPATCH))
DIE = np.zeros((NPATCH,NPATCH))

DEE[0,0] = 0.50
DEE[1,1] = 0.50
DIE[0,0] = 0.40
DIE[1,1] = 0.40

DEI[0,0] = 0.45
DEI[1,1] = 0.45
DII[0,0] = 0.35
DII[1,1] = 0.35

DEE /= NE_source
DIE /= NE_source
DEI /= NI_source
DII /= NI_source

    
def cortical_to_cortical_connection(background_population_dict, internal_population_dict, DEE,DIE,DEI,DII, delay):
    connection_list = []
    """
    """
    # feedforward connections
    for ifftarget in range(Net_settings['nmax']):
        ''' excitatory '''
        source_population = background_population_dict[ifftarget,'e']
        target_population = internal_population_dict[ifftarget,'e']
        curr_connection = Connection(source_population,target_population,nsyn = 1.0,nsyn_post = Cell_type_num['e'], 
                                             weights = 0.25,probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
        connection_list.append(curr_connection)
        
        ''' inhibitory '''
        source_population = background_population_dict[ifftarget,'i']
        target_population = internal_population_dict[ifftarget,'i']
        curr_connection = Connection(source_population,target_population,nsyn = 1.0,nsyn_post = Cell_type_num['e'], 
                                             weights = 0.35,probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
        connection_list.append(curr_connection)
    # short range connection
    for isource in range(Net_settings['nmax']):
        for itarget in range(isource,Net_settings['nmax']):
            if isource == itarget: # self-connection
                # seeself
                source_population = internal_population_dict[isource,'e']
                target_population = internal_population_dict[itarget,'e']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['e'], weights = DEE[itarget,isource],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                # seiself
                source_population = internal_population_dict[isource,'i']
                target_population = internal_population_dict[itarget,'e']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['e'], weights = -DEI[itarget,isource],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                
                # sieself
                source_population = internal_population_dict[isource,'e']
                target_population = internal_population_dict[itarget,'i']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['i'], weights = DIE[itarget,isource],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                # siiself
                source_population = internal_population_dict[isource,'i']
                target_population = internal_population_dict[itarget,'i']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['i'], weights = -DII[itarget,isource],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
            else:
                # seeself
                source_population = internal_population_dict[isource,'e']
                target_population = internal_population_dict[itarget,'e']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['e'], weights = DEE[itarget,isource],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                
                source_population = internal_population_dict[itarget,'e']
                target_population = internal_population_dict[isource,'e']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['e'], weights = DEE[isource,itarget],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                # seiself
                source_population = internal_population_dict[isource,'i']
                target_population = internal_population_dict[itarget,'e']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['e'], weights = -DEI[itarget,isource],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                
                source_population = internal_population_dict[itarget,'i']
                target_population = internal_population_dict[isource,'e']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['e'], weights = -DEI[isource,itarget],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                
                # sieself
                source_population = internal_population_dict[isource,'e']
                target_population = internal_population_dict[itarget,'i']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['i'], weights = DIE[itarget,isource],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                
                source_population = internal_population_dict[itarget,'e']
                target_population = internal_population_dict[isource,'i']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['i'], weights = DIE[isource,itarget],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                # siiself
                source_population = internal_population_dict[isource,'i']
                target_population = internal_population_dict[itarget,'i']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['i'], weights = -DII[itarget,isource],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                
                source_population = internal_population_dict[itarget,'i']
                target_population = internal_population_dict[isource,'i']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['i'], weights = -DII[isource,itarget],
                                             probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)

    return connection_list


# Net_settings,Hypm,Pham,Orim,index_2_loc = create_functional_columns(Net_settings,Fun_map_settings,ori)
(dxx,dyy) = (500.0/Net_settings['xn'],500.0/Net_settings['yn'])
dxx = dxx**2
dyy = dyy**2

connection_list = cortical_to_cortical_connection(background_population_dict,internal_population_dict,DEE,DIE,DEI,DII,0.0)


"""
"""
simulation = Simulation(population_list, connection_list,Net_settings,Cell_type_num,verbose=True)
mEbin_ra,mIbin_ra,xEbin_ra,xIbin_ra,VEavg_ra,VIavg_ra,VEstd_ra,VIstd_ra,rEbin_ra,rIbin_ra,P_MFEbin_ra = simulation.update(t0=t0, dt=dt, tf=tf)
'''
if plot_flag == 1 and dtbin_record_flag == 1
means that figures should be plotted and time-bin recording rather than tiny
time interval is used as minimum recording time step
'''
# for figure 2
Vedges = util.get_v_edges(-1,1.0,dv)
Vbins  = 0.5 * (Vedges[:-1]+Vedges[1:])

plt.figure()
for idx_pop in range(NPATCH):
    # POPULATION ipop
    plt.subplot(2,1,idx_pop+1)
    rEpop,rIpop = np.squeeze(rEbin_ra[idx_pop,:,:]),np.squeeze(rIbin_ra[idx_pop,:,:])
    mEpop,mIpop = np.squeeze(mEbin_ra[:,idx_pop]),np.squeeze(mIbin_ra[:,idx_pop])
    xEpop,xIpop = np.squeeze(xEbin_ra[:,idx_pop]),np.squeeze(xIbin_ra[:,idx_pop])
    VEavgpop,VIavgpop = np.squeeze(VEavg_ra[:,idx_pop]),np.squeeze(VIavg_ra[:,idx_pop])
    VEstdpop,VIstdpop = np.squeeze(VEstd_ra[:,idx_pop]),np.squeeze(VIstd_ra[:,idx_pop])
#    #start plot
#    rEtmp,rItmp = np.mean(rEpop,axis = 1),np.mean(rIpop,axis =1)
#    plt.subplot(1,2,idx_pop+1)
#    plt.plot(Vbins,rEtmp,'r')
#    plt.plot(Vbins,rItmp,'b')
#    plt.plot(VEavgpop,'r')
#    plt.plot(VEavgpop+VEstdpop,'m')
#    plt.plot(VEavgpop-VEstdpop,'m')
#    
#    plt.plot(VIavgpop,'b')
#    plt.plot(VIavgpop+VEstdpop,'g')
#    plt.plot(VIavgpop-VEstdpop,'g')
    
    plt.subplot(2,1,idx_pop+1)
    c,bins = np.histogram(mEbin_ra[:,idx_pop],np.arange(49))
    plt.plot(np.log2(np.arange(1,49)),np.log2(1+c),'r')
    c,bins = np.histogram(mIbin_ra[:,idx_pop],np.arange(37))
    plt.plot(np.log2(np.arange(1,37)),np.log2(1+c),'b')
    
    
## transform into intuitive description
#ne = Net_settings['nmax']
#ni = Net_settings['nmax']
#m_exc = np.zeros((ne,int(tf/dt)))
#m_inh = np.zeros_like(m_exc)
#v_exc = np.zeros_like(m_exc)
#v_inh = np.zeros_like(m_exc)
#tt = np.arange(0,int(tf/dt)) * dt
## plt.figure(2)
#for i in range(1,ne+1):
#    m_exc[i-1,:] = m_record[2*i-1,:int(tf/dt)]
#    v_exc[i-1,:] = v_record[2*i-1,:int(tf/dt)]
#    # plt.subplot(2,1,1)
#    # plt.plot(tt,m_exc[i-1,:])
#    m_inh[i-1,:] = m_record[2*i,:int(tf/dt)]
#    v_inh[i-1,:] = v_record[2*i,:int(tf/dt)]
#    # plt.subplot(2,1,2)
#    # plt.plot(tt,m_inh[i-1,:])
##plt.show()
#    
#np.save("me.npy", m_exc)
#np.save("mi.npy",m_inh)
#np.save("ve.npy", m_exc)
#np.save("vi.npy",m_inh)



