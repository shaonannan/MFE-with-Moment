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

import scipy.io as scio 

"""
06/28/2018 version
edited by SYX

Exact balance between External Inhibition and Internal Inhibition
"""


# intuitive network structure
Net_settings = {'hyp_num': 2,
                'xhyp':2,
                'yhyp':1,
                'xn':1,
                'yn':1, # 25 subpopulation each 20 cells 20 * 25 = 500 cells per hyp!
                # or 9 subpopulations and each 50 cells 9 * 50 = 450 cells per hyp!
                'nmax': 0,
                'Final_time':20000,
                'dt':0.1,
                'dv':1e-3}
Net_settings['nmax'] = Net_settings['hyp_num'] * Net_settings['xn'] * Net_settings['yn']
# here we use orientation and phase
Fun_map_settings = {'ori_num':1,
                   'phase_num':1}
# cell numbers within identical orientation hypercolumn
''' till 0703 '''
#Cell_type_num = {'e':100,
#                'i':100}
''' from 0703 '''
Cell_type_num = {'e':100,
                'i':100}
print('Here, Network Settings')
print('Number of Hyper-columns: ',Net_settings['hyp_num'])
print('Time resolution:',Net_settings['dt'],' Final time:',Net_settings['Final_time'])
print('Voltage resolution:',Net_settings['dv'])

def create_network_population(Structure_Net,Functional_Map):
    #calculate total number of CG patches
    nmax    = Structure_Net['nmax']
    CGindex = np.arange(nmax)

    # Create populations:
    background_population_dict = {}
    internal_population_dict = {}
    for layer, celltype in itertools.product(CGindex, ['e', 'i']):    
        background_population_dict[layer, celltype] = 0
        internal_population_dict[layer, celltype] = 0

    return (Structure_Net,background_population_dict,internal_population_dict)


# Simulation settings:
t0 = 0.0
# extract effective network parameters
dt = Net_settings['dt']
tf = Net_settings['Final_time']
tfinal = tf
dv = Net_settings['dv']
# parameters may not be used
verbose = True
update_method = 'approx'
approx_order = 1
tol  = 1e-14

NPATCH = Net_settings['nmax']
mEY,mIY,fE,fI = np.zeros(NPATCH),np.zeros(NPATCH),np.zeros(NPATCH),np.zeros(NPATCH)
''' 2018/06/28 '''
## version 1.0
#for i in range(NPATCH):
#    mEY[i] = 1.586*1
#    mIY[i] = 1.500*1#1.504
#    fE[i],fI[i] = 0.028/1.0,0.028/1.0

## version 2.0
#for i in range(NPATCH):
#    mEY[i] = 1.60*1
#    mIY[i] = 1.519*1#1.504
#    fE[i],fI[i] = 0.028/1.0,0.028/1.0

# version 3.0
for i in range(NPATCH):
    mEY[i] = 3.116*1.0
    mIY[i] = 2.462*1.0#81*1#1.504
    fE[i],fI[i] = 0.01316/1.0,0.01314/1.0


# background_population_dict,internal_population_dict,population_list = input_signal_stream([dt,tfinal],Net_settings,mEY,mIY)
def input_signal_stream(t_properties,Net_settings,mEY,mIY):
    (dt, tfinal) = t_properties
    ntt = int(tfinal/dt) # 20 to escape from error
    # Create base-visual stimuli

    # Create populations:
    background_population_dict = {}
    internal_population_dict   = {}
    CG_range = np.arange(Net_settings['nmax'])

    # base activities
    # show order
    id_order = 0
    print('Showing Input Streams in Order')
    for layer, celltype in itertools.product(CG_range, ['e', 'i']):  
        if celltype == 'e':
            External_stimuli_dict = np.ones(ntt)*mEY[layer] * 1
            if layer==0:
                External_stimuli_dict[6000:6005] = 3.116*2.50
        else:
            External_stimuli_dict = np.ones(ntt)*mIY[layer] * 1
            
        background_population_dict[layer,celltype] = ExternalPopulation(External_stimuli_dict,dt, record=False)
        internal_population_dict[layer, celltype]  = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method,
                                approx_order=approx_order, tol=tol, hyp_idx = 0,ei_pop = celltype,NumCell = Cell_type_num[celltype])
        print('order:',id_order,' layer:',layer,' type:',celltype)
        id_order +=1
        
    population_list = list(background_population_dict.values()) + list(internal_population_dict.values())
    return (background_population_dict,internal_population_dict,population_list)

''' Generate Populations, External Populations as well as Internal Populations '''
background_population_dict,internal_population_dict,population_list = input_signal_stream([dt,tfinal],Net_settings,mEY,mIY)

NE_source = Cell_type_num['e']
NI_source = Cell_type_num['i']

''' Generate Recurrent Connectivity Matrix '''
DEE = np.zeros((NPATCH,NPATCH))
DEI = np.zeros((NPATCH,NPATCH))
DII = np.zeros((NPATCH,NPATCH))
DIE = np.zeros((NPATCH,NPATCH))


#''' 2018/07/01 '''
## version 1.0
DEE[0,0] = 0.308
DEE[1,1] = 0.308
DIE[0,0] = 0.308
DIE[1,1] = 0.308
DEI[0,0] = 0.0363
DEI[1,1] = 0.0363
DII[0,0] = 0.0363
DII[1,1] = 0.0363

'''
'''
# correct and best Fast long-range connections V1.0
DEE[0,1] = 0.03
DEE[1,0] = 0.03
DIE[1,0] = 0.03
DIE[0,1] = 0.03
DEI[0,1] = 0.125#25
DEI[1,0] = 0.125#25
DII[0,1] = 0.125#25
DII[1,0] = 0.125#25



''' end  '''
LEE  = np.zeros((NPATCH,NPATCH))
LIE  = np.zeros((NPATCH,NPATCH))

# version 3.0
LEE[0,1] = 0.01020154*3.090
LEE[1,0] = 0.01020154*3.090# original 0z.012015
LIE[0,1] = 0.06120002*3.095
LIE[1,0] = 0.06120002*3.095# original 0.0614

''' recording all parameters '''
'''
!!!!!!!!!!! All have long float
'''
DEER = DEE*1000000
DEIR = DEI*1000000
DIER = DIE*1000000
DIIR = DII*1000000

mEYR = mEY*1000000
mIYR = mIY*1000000

LEER = LEE*1000000
LIER = LIE*1000000
ISOTIMEFORMAT='%Y%m%d%H%M%S'
fileparamname=str(time.strftime(ISOTIMEFORMAT)) + '_params.mat'
scio.savemat(fileparamname, {'DEE':DEER,'DEI':DEIR,'DIE':DIER,'DII':DIIR,'LEE':LEER,'LIE':LIER,'mEY':mEYR,'mIY':mIYR,'fE':fE,'fI':fI})

DEE /= NE_source
DIE /= NE_source
DEI /= NI_source
DII /= NI_source

LEE /= NE_source
LIE /= NE_source
       
# connection_list = cortical_to_cortical_connection(background_population_dict,internal_population_dict,DEE,DIE,DEI,DII,lambdaE,lambdaI,fE,fI,0.0)    
def cortical_to_cortical_connection(background_population_dict, internal_population_dict, DEE,DIE,DEI,DII,LEE,LIE,fE,fI,delay):
    connection_list = []
    """
    """
    # feedforward connections
    for ifftarget in range(Net_settings['nmax']):
        ''' excitatory '''
        source_population = background_population_dict[ifftarget,'e']
        target_population = internal_population_dict[ifftarget,'e']
        curr_connection = Connection(source_population,target_population,nsyn = 1.0,nsyn_post = Cell_type_num['e'], 
                                             weights = fE[ifftarget],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
        connection_list.append(curr_connection)
        
        ''' inhibitory '''
        source_population = background_population_dict[ifftarget,'i']
        target_population = internal_population_dict[ifftarget,'i']
        curr_connection = Connection(source_population,target_population,nsyn = 1.0,nsyn_post = Cell_type_num['i'], 
                                             weights = fI[ifftarget],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)
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
                
                
                # seeself
                source_population = internal_population_dict[isource,'e']
                target_population = internal_population_dict[itarget,'e']
                
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['e'], weights = LEE[itarget,isource],
                                             probs = 1.0,conn_type = 'LongRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                
                source_population = internal_population_dict[itarget,'e']
                target_population = internal_population_dict[isource,'e']
                
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['e'], weights = LEE[isource,itarget],
                                             probs = 1.0,conn_type = 'LongRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)

                
                # sieself
                source_population = internal_population_dict[isource,'e']
                target_population = internal_population_dict[itarget,'i']
                
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['i'], weights = LIE[itarget,isource],
                                             probs = 1.0,conn_type = 'LongRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                
                source_population = internal_population_dict[itarget,'e']
                target_population = internal_population_dict[isource,'i']
                curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['i'], weights = LIE[isource,itarget],
                                             probs = 1.0,conn_type = 'LongRange',v_min = -1.0,v_max = 1.0,dv = dv)
                connection_list.append(curr_connection)
                print('LEE;',LEE[1,0])


    return connection_list


# Net_settings,Hypm,Pham,Orim,index_2_loc = create_functional_columns(Net_settings,Fun_map_settings,ori)
(dxx,dyy) = (500.0/Net_settings['xn'],500.0/Net_settings['yn'])
dxx = dxx**2
dyy = dyy**2

connection_list = cortical_to_cortical_connection(background_population_dict,internal_population_dict,DEE,DIE,DEI,DII,LEE,LIE,fE,fI,0.0)


"""
"""
simulation = Simulation(population_list, connection_list,Net_settings,Cell_type_num,DEE,DIE,DEI,DII,LEE,LIE,verbose=True)
(mEbin_ra,mIbin_ra,xEbin_ra,xIbin_ra,rEbin_ra,rIbin_ra,P_MFEbin_ra,NMDAEbin_ra,NMDAIbin_ra,HNMDAEbin_ra,HNMDAIbin_ra,VEavgbin_ra,VIavgbin_ra,VEstdbin_ra,VIstdbin_ra) = simulation.update(t0=t0, dt=dt, tf=tf)

'''
if plot_flag == 1 and dtbin_record_flag == 1
means that figures should be plotted and time-bin recording rather than tiny
time interval is used as minimum recording time step
'''
# for figure 2
Vedges = util.get_v_edges(-1,1.0,dv)
Vbins  = 0.5 * (Vedges[:-1]+Vedges[1:])
NE = Cell_type_num['e']
NI = Cell_type_num['i']
plt.figure()
for idx_pop in range(NPATCH):
    # POPULATION ipop
    plt.subplot(2,2,idx_pop+1)
    rEpop,rIpop = np.squeeze(rEbin_ra[idx_pop,:,:]),np.squeeze(rIbin_ra[idx_pop,:,:])
    rEpop,rIpop = np.mean(rEpop,axis =1),np.mean(rIpop,axis= 1)
    v_axis = 1.0/(np.arange(len(rEpop))+1)
    plt.plot(rEpop,'r')
    plt.plot(rIpop,'b')
    plt.xlim([1000,2000])
    plt.ylim([0,3.0])
    
    
    plt.subplot(2,2,2+idx_pop+1)
    lenNE = int(np.ceil(1.5*NE))
    c,bins = np.histogram(mEbin_ra[:,idx_pop],np.arange(1,lenNE+1))
    plt.plot(np.log2(np.arange(1,lenNE)),np.log2(1+c),'r')
    lenNI = int(np.ceil(1.5*NI))
    c,bins = np.histogram(mIbin_ra[:,idx_pop],np.arange(1,lenNI+1))
    plt.plot(np.log2(np.arange(1,lenNI)),np.log2(1+c),'b')

    
plt.figure()
for idx_pop in range(NPATCH):
    # POPULATION ipop
    plt.subplot(5,1,idx_pop+1)
    xEpop,xIpop = np.squeeze(xEbin_ra[:,idx_pop]),np.squeeze(xIbin_ra[:,idx_pop])
    
    plt.plot(xEpop,'r')
#    plt.plot(xIpop,'b')
    plt.ylim([0,20])
    
for idx_pop in range(NPATCH):
    # POPULATION ipop
    plt.subplot(5,1,2+idx_pop+1)
    xEpop,xIpop = np.squeeze(xEbin_ra[:,idx_pop]),np.squeeze(xIbin_ra[:,idx_pop])
    
#    plt.plot(xEpop,'r')
    plt.plot(xIpop,'b')
    plt.ylim([0,20])
plt.subplot(5,1,5)
plt.plot(P_MFEbin_ra[:,0],'y')
plt.ylim([0,0.2])


filename=str(time.strftime(ISOTIMEFORMAT)) + '.mat'
scio.savemat(filename,{'mEbin_ra':mEbin_ra,'mIbin_ra':mIbin_ra,'xEbin_ra':xEbin_ra,'xIbin_ra':xIbin_ra,\
                       'VEavgbin_ra':VEavgbin_ra,'VIavgbin_ra':VIavgbin_ra,'VEstdbin_ra':VEstdbin_ra,'VIstdbin_ra':VIstdbin_ra,\
                       'rEbin_ra':rEbin_ra,'rIbin_ra':rIbin_ra,'P_MFEbin_ra':P_MFEbin_ra}) 
filelrname=str(time.strftime(ISOTIMEFORMAT)) + '_LR.mat'
scio.savemat(filelrname, {'NMDAEbin_ra':NMDAEbin_ra,'NMDAIbin_ra':NMDAIbin_ra,'HNMDAEbin_ra':HNMDAEbin_ra,'HNMDAIbin_ra':HNMDAIbin_ra})



