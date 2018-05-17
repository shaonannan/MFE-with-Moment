import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio 
import scipy.misc as sm
VE = np.reshape(np.arange(12),(6,2))
VE = VE/16.0
VE[:,0] = VE[:,1]-0.02

VI = np.reshape(np.arange(12),(6,2))
VI = VI/16.0
VI[:,0] = VI[:,1]-0.02

DEE = np.zeros((2,2))
DEI = np.zeros((2,2))
DII = np.zeros((2,2))
DIE = np.zeros((2,2))

DEE[0,0] = 0.14
DEE[1,1] = 0.22
DIE[0,0] = 0.6
DIE[1,1] = 0.15

DEI[0,0] = 0.05
DEI[1,1] = 0.05
DII[0,0] = 0.07
DII[1,1] = 0.02

pop_idx_E = np.zeros((6,2),dtype=int)
pop_idx_E[:,0] = 0
pop_idx_E[:,1] = 1
pop_idx_I = np.zeros((6,2),dtype=int)
pop_idx_I[:,0] = 0
pop_idx_I[:,1] = 1
VE[2:3,0] = 1.2
VE[4,1] = 1.3
VI[2:3,0] = 0.8
VI[4,1] = 0.6

NE,NI = 6,6
NPATCH = 2

def psample(ldt):
    if ldt>5:
        outspike = np.maximum(0,round(ldt + np.sqrt(ldt)*np.random.randn()))
    else:
        kra = np.arange(15)
        pra = np.cumsum((ldt**kra)*np.exp(-ldt)/sm.factorial(kra))
        pra = pra[::-1]
        pra = np.append(pra,-1)
        pra = pra[::-1]
        pra = np.append(pra,2)
        mininner = np.minimum(np.random.random(),pra[-1])
        idx = np.where(pra-mininner<0)
        maxidx = max(idx[0][:])
#        print('maxidx: ',maxidx)
        outspike = maxidx
    return outspike
print(psample(2.5))
#(E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos) = getMFE_ifdyn(0,VE,VI,DEE,DEI,DIE,DII,DEY,DIY,pop_idx_E,pop_idx_I)
def getMFE_ifdyn(verbose,VE,VI,DEE,DEI,DIE,DII,DEY,DIY,pop_idx_E,pop_idx_I):
    ''' assume that voltages are distinct
    '''
    
    (TAU,VT,VR) = (20.0,1.0,0.0)
    (NE,NPATCH)  = np.shape(np.squeeze(VE))
    (NI,NPATCH1) = np.shape(np.squeeze(VI))
    ''' checking '''
    if NPATCH!=NPATCH1:
        print('DIMENSION MISMATCH!')

    VE,pop_idx_E = np.reshape(VE.T,(NE*NPATCH,1)),np.reshape(pop_idx_E.T,(NE*NPATCH,1))
    VI,pop_idx_I = np.reshape(VI.T,(NI*NPATCH1,1)),np.reshape(pop_idx_I.T,(NI*NPATCH1,1))
    VE,VI = np.squeeze(VE),np.squeeze(VI)
    pop_idx_E,pop_idx_I = np.squeeze(pop_idx_E),np.squeeze(pop_idx_I)
    ''' if use squeeze, the could get 1-D I_fired and E_fired '''

    VEj = np.argsort(VE)
    VE  = np.sort(VE)
    pop_idx_E = pop_idx_E[VEj]
    VIj = np.argsort(VI)
    VI  = np.sort(VI)
    pop_idx_I = pop_idx_I[VIj]
    
    VEr = np.argsort(VEj)
    VEjs= np.sort(VEj)
    VIr = np.argsort(VIj)
    VIjs= np.sort(VIj)
    
    VE_orig,VI_orig = VE.copy(),VI.copy() 
    ''' have already sorted '''
    
    '''
    find E_fired/remaining I_fired/remaining
    '''
    I_fired = np.where(VI>=VT)
    I_fired = np.squeeze(I_fired)
    LI      = np.size(I_fired)
    I_remaining = np.where(VI<VT)
    I_remaining = np.squeeze(I_remaining)
    accumulated_DII,accumulated_DEI = np.zeros((NPATCH*NI,1)),np.zeros((NPATCH*NE,1))
    could_use_min_to_I,could_use_min_to_E = np.zeros((NPATCH*NI,1)),np.zeros((NPATCH*NE,1))
    for i in range(LI):
        ''' all about Inh '''
        if LI == 1:
            type_I = pop_idx_I[I_fired]
        else:
            type_I = pop_idx_I[I_fired[i]]
        target_I_head = pop_idx_I
        target_I_remaining = np.squeeze(pop_idx_I[I_remaining])
        ''' all about Exc '''
        target_E_head = pop_idx_E
        accumulated_DII += np.reshape(DII[target_I_head,type_I],(len(target_I_head),1))
        could_use_min_to_I = np.column_stack((could_use_min_to_I,DII[target_I_head,type_I]))
        VI[I_remaining] -= np.squeeze(DII[target_I_remaining,type_I])
        accumulated_DEI += np.reshape(DEI[target_E_head,type_I],(len(target_E_head),1))
        could_use_min_to_E = np.column_stack((could_use_min_to_E,DEI[target_E_head,type_I]))
        VE -= DEI[target_E_head,type_I]
        
    E_fired = np.where(VE>=VT)
    E_fired = np.squeeze(E_fired)
    LE      = np.size(E_fired)
    E_remaining = np.where(VE<VT)
    E_remaining = np.squeeze(E_remaining)
    total_V_to_add_to_E,total_V_to_add_to_I = np.zeros((NPATCH*NE,1)),np.zeros((NPATCH*NI,1))
    could_use_add_to_E,could_use_add_to_I = np.zeros((NPATCH*NE,1)),np.zeros((NPATCH*NI,1))
    for i in range(LE):
        ''' all about Exc '''
        if LE == 1:
            type_E = pop_idx_E[E_fired]
        else:
            type_E = pop_idx_E[E_fired[i]]
        target_E_head = pop_idx_E
        ''' all about Inh '''
        target_I_head = pop_idx_I

        total_V_to_add_to_E += np.reshape(DEE[target_E_head,type_E],(len(target_E_head),1))
        #print('121',total_V_to_add_to_E)
        could_use_add_to_E = np.column_stack((could_use_add_to_E,DEE[target_E_head,type_E]))
        #print('123',could_use_add_to_E)
        total_V_to_add_to_I += np.reshape(DIE[target_I_head,type_E],(len(target_I_head),1))
        #print('125',total_V_to_add_to_I)
        could_use_add_to_I = np.column_stack((could_use_add_to_I,DIE[target_I_head,type_E]))
        #print('127',could_use_add_to_I)
    ''' START EACH CONDITIONS
    '''
    #for iter in range(7):
    iter = 0
    while (max(np.squeeze(total_V_to_add_to_E)) >0)|(max(np.squeeze(total_V_to_add_to_I)) >0):
        iter+=1
        print('loop:',iter)
        #print(VE[E_remaining])
        possible_E_spikes = np.where(VE[E_remaining]>=(VT - np.squeeze(total_V_to_add_to_E[E_remaining])))
        max_E,ind_E = max(VE[E_remaining]),np.argmax(VE[E_remaining])
        ind_E = E_remaining[ind_E]
        possible_I_spikes = np.where(VI[I_remaining]>=(VT - np.squeeze(total_V_to_add_to_I[I_remaining])))
        max_I,ind_I = max(VI[I_remaining]),np.argmax(VI[I_remaining])
        ind_I = I_remaining[ind_I]
        
#        print('137 pe:',possible_E_spikes)
#        print('137 max_E:',max_E)
#        print('137 ind_E:',ind_E)
#        
#        print('137 pi:',possible_I_spikes)
#        print('137 max_I:',max_I)
#        print('137 ind_I:',ind_I)
        
        ce = np.shape(possible_E_spikes)[1]
        ci = np.shape(possible_I_spikes)[1]
        #print(ce,':',total_V_to_add_to_E[E_remaining])
        #print(ci,':',total_V_to_add_to_I[I_remaining])
        if((ce<1) & (ci<1)):
            V_to_add_to_E = total_V_to_add_to_E
            V_to_add_to_I = total_V_to_add_to_I
            VE[E_remaining] += V_to_add_to_E[E_remaining,0]
            VI[I_remaining] += V_to_add_to_I[I_remaining,0]
            total_V_to_add_to_E = np.zeros_like(total_V_to_add_to_E)
            total_V_to_add_to_I = np.zeros_like(total_V_to_add_to_I)
        elif((ce>0)&(ci<1)):
            V_to_add_to_E = VT - max_E
            V_to_add_to_I = np.minimum(total_V_to_add_to_I,VT-max_E)
            #print('155 vti:',V_to_add_to_I)
            E_fired = np.append(E_fired,ind_E)
            E_remaining = np.setdiff1d(E_remaining,ind_E)
            LE += 1

            type_E = pop_idx_E[ind_E]
            target_E_head = pop_idx_E
            target_E_remaining = pop_idx_E[E_remaining]
            target_I_head = pop_idx_I
            target_I_remaining = pop_idx_I[I_remaining]

            VE[E_remaining] += V_to_add_to_E*np.ones_like(VE[E_remaining])
            VI[I_remaining] += np.squeeze(V_to_add_to_I[I_remaining])

            total_V_to_add_to_E = total_V_to_add_to_E - V_to_add_to_E + np.reshape(DEE[target_E_head,type_E],(len(target_E_head),1))
            could_use_add_to_E  = np.column_stack((could_use_add_to_E,DEE[target_E_head,type_E]))
            total_V_to_add_to_I = total_V_to_add_to_I - V_to_add_to_I + np.reshape(DIE[target_I_head,type_E],(len(target_I_head),1))
            could_use_add_to_I  = np.column_stack((could_use_add_to_I,DIE[target_I_head,type_E]))
        elif((ce<1)&(ci>0)):
            V_to_add_to_I = VT - max_I
            V_to_add_to_E = np.minimum(total_V_to_add_to_E,VT-max_I)
            type_I = pop_idx_I[ind_I]
            target_E_head = pop_idx_E
            target_E_remaining = pop_idx_E[E_remaining]
            VE[E_remaining] = VE[E_remaining] - np.squeeze(DEI[target_E_remaining,type_I]) + np.squeeze(V_to_add_to_E[E_remaining])
            accumulated_DEI += np.reshape(DEI[target_E_head,type_I],(len(target_E_head),1))
            #print('184 adi:',VE[E_remaining]+ np.squeeze(V_to_add_to_E[E_remaining]))
            #print('minus185:',np.squeeze(DEI[target_E_remaining,type_I]))
            #print('186 tt:',VE[E_remaining])
            ''' I-fired '''
            I_fired = np.append(I_fired,ind_I)
            I_remaining = np.setdiff1d(I_remaining, ind_I)
            LI += 1
            target_I_head = pop_idx_I
            target_I_remaining = pop_idx_I[I_remaining]
            #print('193: ',VI[I_remaining])

            VI[I_remaining] = VI[I_remaining] - np.squeeze(DII[target_I_remaining,type_I]) + V_to_add_to_I * np.ones_like(VI[I_remaining])
            accumulated_DII += np.reshape(DII[target_I_head,type_I],(len(target_I_head),1))

            total_V_to_add_to_E -= V_to_add_to_E
            total_V_to_add_to_I -= V_to_add_to_I
            #print('198: ',total_V_to_add_to_I.T)
        elif((ce>0)&(ci>0)):
            ''' E-fired '''
            temp_add_E = 0.0
            v_theo_add_E = VT - VE_orig[ind_E] + accumulated_DEI[ind_E]
            temp_E_ratio = 0.0
            new_could_use_add_E = 0

            use_voltage_target_E = np.squeeze(could_use_add_to_E[ind_E,:])
            for idxE in range(len(use_voltage_target_E)):
                temp_add_E += use_voltage_target_E[idxE]
                if temp_add_E>= v_theo_add_E:
                    temp_E_residual = temp_add_E - v_theo_add_E
                    temp_E_cross    = use_voltage_target_E[idxE] - temp_E_residual
                    temp_E_ratio    = temp_E_cross/use_voltage_target_E[idxE]
                    temp_E_ratio_check = 1.0 - temp_E_residual/use_voltage_target_E[idxE]
                    if temp_E_ratio!= temp_E_ratio_check:
                        print('Excitatory ratio mismatch! a: ',temp_E_ratio,' b: ',temp_E_ratio_check)
                    new_could_use_add_E = idxE + temp_E_ratio
                    break

            ''' I-fired '''
            temp_add_I = 0.0
            v_theo_add_I = VT-VI_orig[ind_I] +accumulated_DII[ind_I]
            temp_I_ratio = 0.0
            new_could_use_add_I = 0

            use_voltage_target_I = np.squeeze(could_use_add_to_I[ind_I,:])
            for idxI in range(len(use_voltage_target_I)):
                temp_add_I += use_voltage_target_I[idxI]
                if temp_add_I >= v_theo_add_I:
                    temp_I_residual = temp_add_I - v_theo_add_I
                    temp_I_cross = use_voltage_target_I[idxI] - temp_I_residual
                    temp_I_ratio = temp_I_cross/use_voltage_target_I[idxI]
                    temp_I_ratio_check = 1.0 - temp_I_residual/use_voltage_target_I[idxI]
                    if temp_I_ratio!=temp_I_ratio_check:
                        print('Inhibitory ratio mismatch! a: ',temp_I_ratio,' b: ',temp_I_ratio_check)
                    new_could_use_add_I = idxI + temp_I_ratio
                    break
            #print('233 ratio:',new_could_use_add_E,new_could_use_add_I)
            if new_could_use_add_E < new_could_use_add_I:
                ''' E-fired '''
                E_fired = np.append(E_fired,ind_E)
                E_remaining = np.setdiff1d(E_remaining,ind_E)
                LE += 1
                V_to_add_to_E = VT - max_E
                V_to_add_to_I = np.minimum(total_V_to_add_to_I,VT-max_E)
                type_E = pop_idx_E[ind_E]
                target_E_head = pop_idx_E
                target_E_remaining = pop_idx_E[E_remaining]

                target_I_head = pop_idx_I
                target_I_remaining = pop_idx_I[I_remaining]

                could_use_add_to_E = np.column_stack((could_use_add_to_E,DEE[target_E_head,type_E]))
                total_V_to_add_to_E = total_V_to_add_to_E + np.reshape(DEE[target_E_head,type_E],(len(target_E_head),1)) - V_to_add_to_E
                could_use_add_to_I = np.column_stack((could_use_add_to_I,DIE[target_I_head,type_E]))
                total_V_to_add_to_I = total_V_to_add_to_I + np.reshape(DIE[target_I_head,type_E],(len(target_I_head),1)) - V_to_add_to_I

                VE[E_remaining] = VE[E_remaining] + V_to_add_to_E * np.ones_like(VE[E_remaining])
                VI[I_remaining] = VI[I_remaining] + np.squeeze(V_to_add_to_I[I_remaining])
            elif new_could_use_add_I <= new_could_use_add_E:
                ''' I-fired '''
                I_fired = np.append(I_fired,ind_I)
                I_remaining = np.setdiff1d(I_remaining,ind_I)
                LI += 1
                V_to_add_to_I = VT - max_I
                V_to_add_to_E = np.minimum(total_V_to_add_to_E,VT-max_I)

                type_I = pop_idx_I[ind_I]
                target_I_head = pop_idx_I
                target_I_remaining = pop_idx_I[I_remaining]

                target_E_head = pop_idx_E
                target_E_remaining = pop_idx_E[E_remaining]

                could_use_min_to_E = np.column_stack((could_use_min_to_E,DEI[target_E_head,type_I]))
                total_V_to_add_to_E = total_V_to_add_to_E - V_to_add_to_E
                accumulated_DEI += np.reshape(DEI[target_E_head,type_I],(len(target_E_head),1))

                could_use_min_to_I = np.column_stack((could_use_min_to_I,DII[target_I_head,type_I]))
                total_V_to_add_to_I = total_V_to_add_to_I - V_to_add_to_I
                accumulated_DII += np.reshape(DII[target_I_head,type_I],(len(target_I_head),1))

                VE[E_remaining] = VE[E_remaining] - np.squeeze(DEI[target_E_remaining,type_I]) + np.squeeze(V_to_add_to_E[E_remaining])
                VI[I_remaining] = VI[I_remaining] - np.squeeze(DII[target_I_remaining,type_I]) + V_to_add_to_I*np.ones_like(VI[I_remaining])

    VE,VI = np.reshape(VE,(NE*NPATCH,1)),np.reshape(VI,(NI*NPATCH,1))
    print('289: fired inh',np.size(I_fired))
    E_fired,I_fired = np.reshape(E_fired,(np.size(E_fired),1)),np.reshape(I_fired,(np.size(I_fired),1))
    ncurrE,ncurrI = len(VE),len(VI)
    VE_pre,VI_pre = VE,VI 
    VE_pre[ncurrE-LE:,0] = 1.0
    VI_pre[ncurrI-LI:,0] = 1.0

    VE_pos,VI_pos = VE,VI 
    VE_pos[ncurrE-LE:,0] = 0.0
    VI_pos[ncurrI-LI:,0] = 0.0

    E_fired,I_fired = VEj[E_fired], VIj[I_fired]
    VE_pre,VI_pre = VE_pre[VEr],VI_pre[VIr]
    VE_pos,VI_pos = VE_pos[VEr],VI_pos[VIr]

    pop_idx_E_col,pop_idx_I_col = pop_idx_E[VEr],pop_idx_I[VIr]

    VE_pre = np.reshape(VE_pre,(NPATCH,NE))
    VE_pre = VE_pre.T
    VI_pre = np.reshape(VI_pre,(NPATCH,NI))
    VI_pre = VI_pre.T
    VE_pos = np.reshape(VE_pos,(NPATCH,NE))
    VE_pos = VE_pos.T
    VI_pos = np.reshape(VI_pos,(NPATCH,NI))
    VI_pos = VI_pos.T

    pop_idx_E_col = np.reshape(pop_idx_E_col,(NPATCH,NE))
    pop_idx_E_col = pop_idx_E_col.T
    pop_idx_I_col = np.reshape(pop_idx_I_col,(NPATCH,NI))
    pop_idx_I_col = pop_idx_I_col.T

    return (E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos)
#(E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos) = getMFE_ifdyn(0,VE,VI,DEE,DEI,DIE,DII,0,0,pop_idx_E,pop_idx_I)
#
#
## using E_fired and I_fired to find fired-cell numbers
#E_fired_num,I_fired_num = np.zeros((NE*NPATCH,1)),np.zeros((NI*NPATCH,1))
#E_fired_num[E_fired] = 1
#I_fired_num[I_fired] = 1
#E_fired_num = np.reshape(E_fired_num,(NPATCH,NE)).T
#I_fired_num = np.reshape(I_fired_num,(NPATCH,NI)).T
#E_ind_num = np.sum(E_fired_num,axis = 0)
#I_ind_num = np.sum(I_fired_num,axis = 0) 
#print('E_fired： ',E_fired)
#print('E_fired_num： ',E_ind_num)
#
#print('I_fired： ',I_fired)
#print('I_fired_num： ',I_ind_num)
#
#print('VE-pos ',VE_pos )
#print('VI-pos ',VI_pos )
#
#VEpos,VIpos,VEpre,VIpre = VE_pos,VI_pos,VE_pre,VI_pre
#Vedges = 0.1 * np.arange(10)
#Vbins  = 0.5 * (Vedges[1:] + Vedges[0:-1])
#rE,rI = np.zeros((len(Vbins),NPATCH)),np.zeros((len(Vbins),NPATCH))
#for i in range(NPATCH):
#    VEposu = np.squeeze(VEpos[:,i])
#    rE_tmp,Vedge = np.histogram(VEposu, Vedges)
#    rE[:,i] = rE_tmp/np.sum(rE_tmp)
#    VIposu = np.squeeze(VIpos[:,i])
#    rI_tmp,Vedge = np.histogram(VIposu, Vedges)
#    rI[:,i] = rI_tmp/np.sum(rI_tmp)
#V1 = Vbins
#V2 = V1*Vbins
#V3 = V2*Vbins
#V4 = V3*Vbins   
#h = Vbins[2] - Vbins[1]
#vbarE = np.zeros(NPATCH)
#wbarE,vbar3E,vbar4E = np.zeros_like(vbarE),np.zeros_like(vbarE),np.zeros_like(vbarE)
#vbarI = np.zeros(NPATCH)
#wbarI,vbar3I,vbar4I = np.zeros_like(vbarI),np.zeros_like(vbarI),np.zeros_like(vbarI)
#for i in range(NPATCH):
#    rE_tmp = np.squeeze(rE[:,i])
#    rI_tmp = np.squeeze(rI[:,i])
#    vbarE[i] = np.sum(V1*rE_tmp ) * h
#    wbarE[i] = np.sum(V2*rE_tmp ) * h
#    vbar3E[i] = np.sum(V3*rE_tmp ) * h
#    vbar4E[i] = np.sum(V4*rE_tmp ) * h
#    
#    vbarI[i] = np.sum(V1*rI_tmp ) * h
#    wbarI[i] = np.sum(V2*rI_tmp ) * h
#    vbar3I[i] = np.sum(V3*rI_tmp ) * h
#    vbar4I[i] = np.sum(V4*rI_tmp ) * h
#print('vbarE',vbarE)    
#''' refresh data in simulation '''
#idxE,idxI = 0,0
#ind_rec   = 0
#for p in self.population_list:
#    ind_rec +=1
#    if(counter>0):
#        if(ind_rec>numCGPatch):
#            if p.ei_pop == 'e':
#                ''' 
#                print previous informatio, before MFE
#                '''
#                print('before MFE, Exc cell: ')
#                print('firing rate: ',p.curr_firing_rate)
#                print('moment: v1 %.2f, v2 %.2f, v3 %.2f, v4 %.2f'%(p.v1,p.v2,p.v3,p.v4))
#                print('Lag-parame: ',p.La0,p.La1)
#                ''' me[i]=0'''
#                p.firing_rate = 0.0
#                ''' v1,v2,v3,v4 '''
#                p.v1,p.v2,p.v3,p.v4 = vbarE[idxE],wbarE[idxE],vbar3E[idxE],vbar4E[idxE]
#                ''' la0/1 '''
#                p.La0,p.La1 = 0.0,0.0
#                print('after MFE, Exc cell: ')
#                print('firing rate: ',p.curr_firing_rate)
#                print('moment: v1 %.2f, v2 %.2f, v3 %.2f, v4 %.2f'%(p.v1,p.v2,p.v3,p.v4))
#                print('Lag-parame: ',p.La0,p.La1)
#                idxE += 1
#            else:
#                print('before MFE, Exc cell: ')
#                print('firing rate: ',p.curr_firing_rate)
#                print('moment: v1 %.2f, v2 %.2f, v3 %.2f, v4 %.2f'%(p.v1,p.v2,p.v3,p.v4))
#                print('Lag-parame: ',p.La0,p.La1)
#                p.firing_rate = 0.0
#                p.v1,p.v2,p.v3,p.v4 = vbarI[idxI],wbarI[idxI],vbar3I[idxI],vbar4I[idxI]
#                p.La0,p.La1 = 0.0,0.0
#                rint('after MFE, Exc cell: ')
#                print('firing rate: ',p.curr_firing_rate)
#                print('moment: v1 %.2f, v2 %.2f, v3 %.2f, v4 %.2f'%(p.v1,p.v2,p.v3,p.v4))
#                print('Lag-parame: ',p.La0,p.La1)
#                idxI += 1



