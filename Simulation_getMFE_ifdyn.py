from connectiondistributioncollection import ConnectionDistributionCollection
import time
import numpy as np
import utilities as util
import matplotlib.pyplot as plt



class Simulation(object):
    """
    Parameters:
    list :
        All sub-population (cluster)
        All connection (cluster)
        [type of both is 'List', which is changable variable, and could be changed]
        
    generate after initiate(by hand)
        connection_distribution
        connection_distribution_list
        [the differences between connection, connection_distribution and connection_distribution_list are
        connection ---> the component of 'connection_list', record all information and related information and object,like source and others
        connection_distribution --> this variable is a preparation variable for further processing, each 'connection' could generate a 
        class 'connecton_distribution' and then, using weight,syn,prob, could calculate flux_matrix and threshold
        each 'connection_distribution' item is defined by 'weight''syn ''prob', items with identical symbol will be classified to the same
        distribution
        connection_distribution_list --> this is a 'basket', store all unique connections(definition of unique: unique symbol
        'weight','syn','prob' no matter the target/source population)
    """
    def __init__(self,population_list,connection_list,Net_settings,Cell_type_num,verbose=True):
        
        self.verbose = verbose
        self.population_list = population_list
        self.connection_list = [c for c in connection_list if c.nsyn!=0.0]
        self.Net_settings    = Net_settings
        tfinal = Net_settings['Final_time']
        dt     = Net_settings['dt']
        self.ntt = int(tfinal/dt)
        self.m_record = None
        ''' all for MFE '''
        self.VE,self.VI = None,None
        self.DEE,self.DIE,self.DEI,self.DII = None,None,None,None
        self.Vedges,self.Vbins = None,None
        self.NE,self.NI = Cell_type_num['e'],Cell_type_num['i']
        self.MFE_num  = 0
        self.MFE_flag = 0

    
    def initialize(self,t0=0.0):
        """
        initialize by hand, first put all sub-population and connection-pair
        !!! put them on the same platform!!! simulationBridge
        """
        ''' initialize P_MFE '''
        ''' at first, we only use NHYP as NPATCH '''
        self.iteration_max = self.ntt+100
        iteration_max = self.iteration_max
        self.tbin_tmp = 0 # initial
        self.tbinsize = 1.0
        dtperbin = int(self.tbinsize/self.dt)
        self.dtperbin = dtperbin
        iteration_bin = int(iteration_max/dtperbin)
        NPATCH = self.Net_settings['hyp_num']
        NE,NI  = self.NE,self.NI
        self.VE,self.VI = np.zeros((NE,NPATCH)),np.zeros((NI,NPATCH))
        self.DEE,self.DIE  = np.zeros((NPATCH,NPATCH)),np.zeros((NPATCH,NPATCH))
        self.DEI,self.DII  = np.zeros((NPATCH,NPATCH)),np.zeros((NPATCH,NPATCH))
        # DTBIN_RECORD_FLAG
        self.tbin_ra = np.zeros((iteration_max,1))
        self.mE_ra   = np.zeros((iteration_max,NPATCH))
        self.mI_ra   = np.zeros((iteration_max,NPATCH))
        self.mEbin_ra = np.zeros((iteration_bin,NPATCH))
        self.mIbin_ra = np.zeros_like(self.mEbin_ra)
        self.xEbin_ra = np.zeros_like(self.mEbin_ra)
        self.xIbin_ra = np.zeros_like(self.xEbin_ra)
        self.P_MFEbin_ra = np.zeros_like(self.xIbin_ra)
        self.P_MFE_ra = np.zeros((iteration_max,1))
        self.rEbin_ra = np.zeros((NPATCH,2000,iteration_bin))
        self.rIbin_ra = np.zeros_like(self.rEbin_ra)

        self.VEavgbin_ra = np.zeros_like(self.P_MFEbin_ra)
        self.VIavgbin_ra = np.zeros_like(self.VEavgbin_ra)
        self.VEstdbin_ra = np.zeros_like(self.VIavgbin_ra)
        self.VIstdbin_ra = np.zeros_like(self.VEstdbin_ra)
        
        self.VEavg_ra = np.zeros((iteration_max,NPATCH))
        self.VIavg_ra = np.zeros_like(self.VEavg_ra)
        self.VEstd_ra = np.zeros_like(self.VIavg_ra)
        self.VIstd_ra = np.zeros_like(self.VEstd_ra)
        self.rE,self.rI  = None,None
        self.NPATCH = NPATCH
        
        self.LE_ra = np.zeros((iteration_max,NPATCH))
        self.LI_ra = np.zeros_like(self.LE_ra)
        ''' IDX '''
        # get v 
        ''' COMMON PARAMETERS '''
        SEE = np.zeros((NPATCH,NPATCH))
        SEI = np.zeros((NPATCH,NPATCH))
        SII = np.zeros((NPATCH,NPATCH))
        SIE = np.zeros((NPATCH,NPATCH))
        
        SEE[0,0] = 0.50
        SEE[1,1] = 0.50
        SIE[0,0] = 0.25
        SIE[1,1] = 0.25
        
        SEI[0,0] = 0.45
        SEI[1,1] = 0.45
        SII[0,0] = 0.35
        SII[1,1] = 0.35
        
        SEE,SIE = SEE/NE,SIE/NE
        SII,SEI = SII/NI,SEI/NI
        self.DEE = SEE.copy()
        self.DEI = SEI.copy()
        self.DIE = SIE.copy()
        self.DII = SII.copy()
        vT = 1.0
        dv = self.Net_settings['dv']
        self.Vedges = util.get_v_edges(-1.0,1.0,dv)
        ''' bins = edges - 1'''
        # in internal , rhov length len(self.Vbins), len(Vedges)-1
        self.Vbins = 0.5*(self.Vedges[0:-1] + self.Vedges[1:]) 
        Vedges = self.Vedges 
        Vbins  = self.Vbins
        idx_vT = len(Vedges)-1 #len(Vbins)
        idx_kickE,idx_kickI = np.zeros((NPATCH,NPATCH),dtype=int),np.zeros((NPATCH,NPATCH),dtype=int)
        for it in range(self.NPATCH):
            for js in range(self.NPATCH):
                value_kickE = vT - SEE[it,js]
                value_kickI = vT - SIE[it,js]
                Ind_k1  = np.where(Vedges>value_kickE)
                IndI_k1 = np.where(Vedges>value_kickI) 
                if np.shape(Ind_k1)[1]>0:
                    idx_kickE[it,js]  = Ind_k1[0][0]
                else:
                    idx_kickE[it,js]  = idx_vT
                if np.shape(IndI_k1)[1]>0:
                    idx_kickI[it,js]  = IndI_k1[0][0]
                else:
                    idx_kickI[it,js]  = idx_vT
        
        self.idx_kickE,self.idx_kickI = idx_kickE,idx_kickI
        self.idx_vT   = idx_vT
        self.MFE_pevent = np.zeros(self.NPATCH)
        self.p_single = np.zeros(self.NPATCH)
        self.rE = np.zeros((len(self.Vbins),self.NPATCH))
        self.rI = np.zeros_like(self.rE)


        
        # An connection_distribution_list (store unique connection(defined by weight,syn,prob))
        self.connection_distribution_collection = ConnectionDistributionCollection() # this is 
        self.t = t0

        # Matrix to record 
        numCGPatch = self.Net_settings['nmax'] * 2 # excitatory and inhibitory
        # 2 * numCGPatch = External Population and Recurrent Population
        # set Matrix to record only Internal Population
        self.m_record = np.zeros((numCGPatch+1, self.ntt + 10))
        self.v_record = np.zeros_like(self.m_record)
        
        # put all subpopulation and all connections into the same platform
        for subpop in self.population_list:
            subpop.simulation = self    # .simulation = self(self is what we called 'simulation')
        for connpair in self.connection_list:
            connpair.simulation = self
            
        # initialize population_list, calculate         
        for p in self.population_list:
            p.initialize()      # 2   
        
        for c in self.connection_list:
            #print 'initialize population'
            c.initialize()      # 1
    '''
    def sim_getsamples(Vedges,rho,Nsamples):
        rho = np.squeeze(rho)
        rho = [0] + rho
        rho = rho/np.sum(rho)
        L   = len(rho)
        F   = np.cumsum(rho)
        [Fv,Fi,Fj]  = np.unique(F,return_index = True,return_inverse = True)
        print('F_indx:',np.shape(F[Fi]),'inverse_indx:',np.shape(Vedges[Fi]))
        output = np.interp(np.random.random(Nsamples),F[Fi],Vedges[Fi])
        
        return output  
    '''      
    def update(self,t0 = 0.0,dt = 1e-1,tf = 200.0):
        self.dt = dt
        self.tf = tf   
        # initialize:
        start_time = time.time()
        self.initialize(t0)
        self.initialize_time_period = time.time()-start_time
        
        # start_running
        start_time = time.time()
        counter = 0
        numCGPatch = self.Net_settings['nmax'] * 2
        print('nET:',self.Net_settings['nmax']*2)

        '''
        at first 
        '''
        Vbins,Vedges,NPATCH = self.Vbins,self.Vedges,self.NPATCH
        while self.t < self.tf:
            self.t+=self.dt
            self.tbin_tmp = int(np.floor(self.t/self.tbinsize*self.dt))
            ind_rec = 0
            #if self.verbose: print ('time: %s' % self.t)
            idxE,idxI = 0,0   # start to accumulate index of hypercolumn
            for p in self.population_list:
                p.USUALorMFE = 1
                p.update()
                ind_rec += 1
                if(ind_rec>numCGPatch):
                    # print('num: ',numCGPatch,np.shape(self.v_record))
                    # print('ind_rec:',ind_rec,'numCG:',numCGPatch)
                    self.v_record[ind_rec-numCGPatch,counter] = p.local_pevent#curr_firing_rate#
                    self.m_record[ind_rec-numCGPatch,counter] = p.curr_firing_rate#
                if(counter>0):
                    if(ind_rec<=numCGPatch):
                        print('')
                    else:   # pnly for interneuron, the MFE make sense
                        if p.ei_pop == 'e': 
                            print('excite : %.5f'%p.local_pevent)
                        else:
                            print('ind : %.5f'%p.local_pevent)
            ind_rec = 0                
            for p in self.population_list:
                ind_rec += 1
                if(counter>0):
                    if(ind_rec<=numCGPatch):
                        print('')
                    else:   # pnly for interneuron, the MFE make sense
                        if p.ei_pop == 'e': 
                            self.rE[:,idxE] = p.curr_rhov
                            self.rEbin_ra[idxE,:,self.tbin_tmp] += p.curr_rhov * self.dt
#                            print('pre: ',p.firing_rate)
#                            p.firing_rate = 0.0
#                            print('pos: ',p.curr_firing_rate)
                            self.mE_ra[counter,idxE] = p.curr_firing_rate
                            self.p_single[idxE] = p.curr_firing_rate * self.dt * p.NumCell

                            self.VEavg_ra[counter,idxE] = p.v1
                            self.VEstd_ra[counter,idxE] = np.sqrt(p.v2-p.v1**2)
                            self.VEavgbin_ra[self.tbin_tmp,idxE] += p.v1*dt
                            self.VEstdbin_ra[self.tbin_tmp,idxE] += np.sqrt(p.v2-p.v1**2)*dt
                            idxE += 1
                        else:
                            self.mI_ra[counter,idxI] = p.curr_firing_rate
                            self.rI[:,idxI] = p.curr_rhov
                            self.rIbin_ra[idxI,:,self.tbin_tmp] += p.curr_rhov * self.dt
                            self.VIavg_ra[counter,idxI] = p.v1
                            self.VIstd_ra[counter,idxI] = np.sqrt(p.v2-p.v1**2)
                            self.VIavgbin_ra[self.tbin_tmp,idxI] += p.v1*dt
                            self.VIstdbin_ra[self.tbin_tmp,idxI] += np.sqrt(p.v2-p.v1**2)*dt
                            idxI += 1
                
            
            ''' start to calculate P_MFE for each time step '''  
            ''' vedges is new for calculate MFE '''
            NE,NI = self.NE,self.NI
            NPATCH = self.NPATCH
            h = self.Vedges[1]- self.Vedges[0]
            local_pevent = 1.0
            
            for isource in range(self.NPATCH):
                #print('sim:')
                local_pevent = 1.0
                for jt in range(self.NPATCH):
                    #print(self.idx_kickE[jt,isource])
                    #print(self.idx_kickI[jt,isource])
                    kickE,kickI = self.idx_kickE[jt,isource],self.idx_kickI[jt,isource]
                    idx_vT   = self.idx_vT
                    ''' Excitatory '''
                    trhov    = np.squeeze(self.rE[:,jt])
                    if isource!=jt:
                        Nup = NE 
                    else:
                        Nup = NE
                    
                    prob_event = (1.0 - np.sum(np.squeeze(trhov[kickE:idx_vT])) * h) ** Nup 
                    local_pevent *= prob_event    

                    ''' Inhibitory '''
                    trhov    = np.squeeze(self.rI[:,jt])
                    Nup = NI
                    prob_event = (1.0 - np.sum(np.squeeze(trhov[kickI:idx_vT])) * h) ** Nup 
                    local_pevent *= prob_event
                self.MFE_pevent[isource] = self.p_single[isource] * (1-local_pevent)  
                print('check MFE:%.5f'% self.MFE_pevent[isource])
                
                
                
            ''' then prepare for MFE '''
            MFE_pevent_max = max(self.MFE_pevent)
            self.P_MFE_ra[counter,0] = MFE_pevent_max
            #print('wrong:',self.tbin_tmp)
            self.P_MFEbin_ra[self.tbin_tmp] +=  MFE_pevent_max * dt
            idx_pevent_max = np.argmax(self.MFE_pevent)
            
            self.MFE_flag = 0
            local_pevent_d = np.random.random()
            if local_pevent_d < MFE_pevent_max:
                self.MFE_flag = 1
                self.MFE_num += 1
                VE = np.zeros((self.NE,self.NPATCH))
                VI = np.zeros((self.NI,self.NPATCH))
                ''' distribution sampling '''
                for i in range(self.NPATCH):
                    V_sample = util.getsamples(np.squeeze(self.Vedges[1:]),np.squeeze(self.rE[:,i]),self.NE)
                    VE[:,i]  = np.squeeze(V_sample)
                    V_sample = util.getsamples(np.squeeze(self.Vedges[1:]),np.squeeze(self.rI[:,i]),self.NI)
                    VI[:,i]  = np.squeeze(V_sample)
                ''' trigger neuron '''
                VE[0:2,idx_pevent_max] = 1.0
                DEEu = self.DEE.copy()
                DIEu = self.DIE.copy()
                DEIu = self.DEI.copy()
                DIIu = self.DII.copy()
                
                pop_idx_E = np.zeros((self.NE,self.NPATCH),dtype=int)
                pop_idx_I = np.zeros((self.NI,self.NPATCH),dtype=int)
                for i in range(self.NPATCH):
                    pop_idx_E[:,i] = i
                    pop_idx_I[:,i] = i
                    
                (E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos) = util.getMFE_ifdyn(0,VE,VI,DEEu,DEIu,DIEu,DIIu,0,0,pop_idx_E,pop_idx_I)
                E_fired_num,I_fired_num = np.zeros((NE*NPATCH,1)),np.zeros((NI*NPATCH,1))
                E_fired_num[E_fired] = 1
                I_fired_num[I_fired] = 1
                E_fired_num = np.reshape(E_fired_num,(NE,NPATCH))
                I_fired_num = np.reshape(I_fired_num,(NI,NPATCH))
                E_ind_num = np.sum(E_fired_num,axis = 0)
                I_ind_num = np.sum(I_fired_num,axis = 0) 
                for i in range(NPATCH):
                    self.LE_ra[self.MFE_num,i] = E_ind_num[i]
                    self.LI_ra[self.MFE_num,i] = I_ind_num[i]
                VEpos,VIpos,VEpre,VIpre = VE_pos,VI_pos,VE_pre,VI_pre
                Vedges = self.Vedges
                Vbins  = self.Vbins
                h = Vbins[2]-Vbins[1]
                rE,rI = np.zeros((len(Vbins),NPATCH)),np.zeros((len(Vbins),NPATCH))
                for i in range(NPATCH):
                    VEposu = np.squeeze(VEpos[:,i])
                    rE_tmp,Vedge = np.histogram(VEposu, Vedges)
                    rE[:,i] = rE_tmp/(np.sum(rE_tmp)*h)
                    #print('sum',np.sum(rE_tmp))
                    #plt.plot(rE[:,i])
                    VIposu = np.squeeze(VIpos[:,i])
                    rI_tmp,Vedge = np.histogram(VIposu, Vedges)
                    rI[:,i] = rI_tmp/(np.sum(rI_tmp)*h)
                V1 = Vbins
                V2 = V1*Vbins
                V3 = V2*Vbins
                V4 = V3*Vbins   
                h = Vbins[2] - Vbins[1]
                vbarE = np.zeros(NPATCH)
                wbarE,vbar3E,vbar4E = np.zeros_like(vbarE),np.zeros_like(vbarE),np.zeros_like(vbarE)
                vbarI = np.zeros(NPATCH)
                wbarI,vbar3I,vbar4I = np.zeros_like(vbarI),np.zeros_like(vbarI),np.zeros_like(vbarI)
                for i in range(NPATCH):
                    rE_tmp = np.squeeze(rE[:,i])
                    rI_tmp = np.squeeze(rI[:,i])
                    vbarE[i] = np.sum(V1*rE_tmp ) * h
                    wbarE[i] = np.sum(V2*rE_tmp ) * h
                    vbar3E[i] = np.sum(V3*rE_tmp ) * h
                    vbar4E[i] = np.sum(V4*rE_tmp ) * h
                    
                    vbarI[i] = np.sum(V1*rI_tmp ) * h
                    wbarI[i] = np.sum(V2*rI_tmp ) * h
                    vbar3I[i] = np.sum(V3*rI_tmp ) * h
                    vbar4I[i] = np.sum(V4*rI_tmp ) * h
                print('SHAPE',np.shape(V1),np.shape(rE_tmp))      
                ''' refresh data in simulation '''
                idxE,idxI = 0,0
                ind_rec   = 0
                for p in self.population_list:
                    ind_rec +=1
                    if(counter>0):
                        if(ind_rec>numCGPatch):
                            if p.ei_pop == 'e':
                                ''' 
                                print previous informatio, before MFE
                                '''
#                                print('before MFE, Exc cell: ')
#                                print('firing rate: ',p.curr_firing_rate)
#                                print('moment: v1 %.5f'%(p.v1))
#                                print('Lag-parame: ',p.La0)
                                ''' me[i]=0'''
                                p.firing_rate = 0.0
                                ''' v1,v2,v3,v4 '''
                                p.v1,p.v2,p.v3,p.v4 = vbarE[idxE],wbarE[idxE],vbar3E[idxE],vbar4E[idxE]
                                ''' la0/1 '''
                                md_La = np.zeros_like(p.La0) 
                                p.La0,p.La1 = md_La,md_La
#                                print('after MFE, Exc cell: ')
#                                print('firing rate: ',p.curr_firing_rate)
#                                print('moment: v1 %.5f'%(p.v1))
#                                print('Lag-parame: ',p.La0)
                                idxE += 1
                            else:
#                                print('before MFE, Inh cell: ')
#                                print('firing rate: ',p.curr_firing_rate)
#                                print('moment: v1 %.5f'%(p.v1))
#                                print('Lag-parame: ',p.La0)
                                p.firing_rate = 0.0
                                p.v1,p.v2,p.v3,p.v4 = vbarI[idxI],wbarI[idxI],vbar3I[idxI],vbar4I[idxI]
                                md_La = np.zeros_like(p.La0) 
                                p.La0,p.La1 = md_La,md_La
#                                print('after MFE, Inh cell: ')
#                                print('firing rate: ',p.curr_firing_rate)
#                                print('moment: v1 %.5f'%(p.v1))
#                                print('Lag-parame: ',p.La0)
                                idxI += 1
                    
                ''' after reset, calculate again'''
                ind_rec   = 0
                for p in self.population_list:
                    ind_rec +=1
                    if(counter>0):
                        if(ind_rec>numCGPatch):
                            p.USUALorMFE = 0
                            p.update()
#                            if p.ei_pop == 'e':
#                                print('after RECALCULATE, Exc cell: ')
#                            else:
#                                print('after RECALCULATE, Inh cell: ')
#                            print('firing rate: ',p.curr_firing_rate)
#                            print('moment: v1 %.5f'%(p.v1))
#                            print('Lag-parame: ',p.La0)

            for c in self.connection_list:
                c.update()
            counter +=1
            
            ''' recording ! '''
                    # DTBIN_RECORD_FLAG
            self.tbin_ra[counter] = np.floor(self.t/self.tbinsize)
            tbin = int(np.floor(self.t/self.tbinsize))
            ind_rec,idxE,idxI   = 0,0,0
            for p in self.population_list:
                ind_rec +=1
                if(counter>0):
                    if(ind_rec>numCGPatch):
                        if p.ei_pop == 'e':
                            self.mEbin_ra[tbin,idxE] += (1-self.MFE_flag) * p.curr_firing_rate * NE * dt + self.MFE_flag * self.LE_ra[self.MFE_num,idxE]
                            idxE += 1
                        else:
                            self.mIbin_ra[tbin,idxI] += (1-self.MFE_flag) * p.curr_firing_rate * NE * dt + self.MFE_flag * self.LI_ra[self.MFE_num,idxI]
                            idxI += 1
                            
                            
        return self.mEbin_ra,self.mIbin_ra,self.VEavg_ra,self.VIavg_ra,self.VEstd_ra,self.VIstd_ra