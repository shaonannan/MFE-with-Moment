from connectiondistributioncollection import ConnectionDistributionCollection
import time
import numpy as np



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
    def __init__(self,population_list,connection_list,Net_settings,verbose=True):
        
        self.verbose = verbose
        self.population_list = population_list
        self.connection_list = [c for c in connection_list if c.nsyn!=0.0]
        self.Net_settings    = Net_settings
        tfinal = Net_settings['Final_time']
        dt     = Net_settings['dt']
        self.ntt = int(tfinal/dt)
        self.m_record = None
    
    def initialize(self,t0=0.0):
        """
        initialize by hand, first put all sub-population and connection-pair
        !!! put them on the same platform!!! simulationBridge
        """
        
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
            
    def update(self,t0 = 0.0,dt = 1e-1,tf = 200.0):
        self.dt = dt#Variable(torch.Tensor([dt]))
        self.tf = tf#Variable(torch.Tensor([tf]))      
        # initialize:
        start_time = time.time()
        self.initialize(t0)
        self.initialize_time_period = time.time()-start_time
        
        # start_running
        start_time = time.time()
        counter = 0
        numCGPatch = self.Net_settings['nmax'] * 2
        print('nET:',self.Net_settings['nmax']*2)
        while self.t < self.tf:
            self.t+=self.dt
            ind_rec = 0
            #if self.verbose: print ('time: %s' % self.t)
            for p in self.population_list:
                p.update()
                ind_rec += 1
                #print('num:',numCGPatch)
                if(ind_rec>numCGPatch):
                    # print('num: ',numCGPatch,np.shape(self.v_record))
                    # print('ind_rec:',ind_rec,'numCG:',numCGPatch)
                    self.v_record[ind_rec-numCGPatch,counter] = p.local_pevent#curr_firing_rate#
                    self.m_record[ind_rec-numCGPatch,counter] = p.curr_firing_rate#
                if(np.mod(counter,100)==0):
                    if(ind_rec<=numCGPatch):
                        print('')
                 
                    else:
                        if p.ei_pop == 'e':
                            print('excite : %.5f'%p.local_pevent)  
                        else:
                            print('inh : %.5f'%p.local_pevent)  
                
            for c in self.connection_list:
                c.update()
            counter +=1
        return self.m_record,self.v_record
