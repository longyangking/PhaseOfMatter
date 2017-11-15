import numpy as np 
from multiprocessing import Pool

class Isingmodel:
    def __init__(self,size,J,beta,iterations=1000,populations=10,verbose=False,tolerance=0.1):
        self.size = size
        self.J = J

        self.populations = populations
        self.iterations = int(iterations)
        self.groundenergy = None
        self.groundstate = None
        self.beta = beta
        self.verbose = verbose

        self.tolerance = tolerance

    def Hamiltonian(self,state):
        '''
        Calculate total energy of whole spin system
        '''
        energy = 0
        (Nx,Ny) = state.shape
        for i in range(Nx):
            for j in range(Ny):
                nexti = (i+1)%Nx
                nextj = (j+1)%Ny
                pali = state[i,j]*state[nexti,j]
                palj = state[i,j]*state[i,nextj]
                energy = -self.J*(pali + palj)
        return energy

    def spinflip(self,index,state):
        '''
        Spin flip operation
        '''
        spin = state[index]
        if spin == 1:
            state[index] = -1
        else:
            state[index] = 1

    def montecarlo(self,state):
        '''
        Monte Carlo Simulations to get ground state
        '''

        for i in range(self.iterations):
            neighbors = np.roll(state,1,axis=1) + np.roll(state,-1,axis=1) + np.roll(state,1,axis=0) + np.roll(state,-1,axis=0)
            
            deltaE = 2*self.J*(state*neighbors)
            P_flip = np.exp(-deltaE*self.beta)

            transitions = (np.random.random(state.shape) < P_flip)*(np.random.random(state.shape) < self.tolerance)*-2 + 1
        
            state = state*transitions

            if self.verbose:
                if (i+1)%(int(self.iterations/10)) == 0:
                    print('Monte Carlo calculating...{num}%'.format(num=100*(i+1)/self.iterations))

            energy = -np.sum(np.sum(deltaE))/2

        return energy,state

    def init(self,parallel=0):
        '''
        Calculate ground state
        '''
        datas = list()
        states = list()
        
        for i in range(self.populations):
            states.append(np.random.choice([-1,1],size=self.size))
        
        if parallel:
            print('Compuation in parallel with core: {num}'.format(num=parallel))
            pool = Pool(parallel)  
            datas = pool.map(self.montecarlo,states) 
            pool.close()
            pool.join()
        else:
            count = 1
            for state in states:
                datas.append(self.montecarlo(state))
                if self.verbose:
                    print('Series: {count}th finished!'.format(count=count))
                count += 1
                    
        energies = [data[0] for data in datas]
        states = [data[1] for data in datas]

        index = np.argmin(energies)
        self.groundenergy = energies[index]
        self.groundstate = states[index]
        return True

    def getgroundstate(self):
        return self.groundstate

    def getgroundenergy(self):
        return self.groundenergy

    def getgroundstate(self):
        return self.groundstate


if __name__=='__main__':
    ising = Isingmodel((100,100),1.0,1,verbose=True,populations=1)
    ising.init(parallel=0)
    import matplotlib.pyplot as plt 
    groundstate = ising.getgroundstate()
    groundenergy = ising.getgroundenergy()
    print('Ground Energy: {num}'.format(num=groundenergy))
    plt.imshow(groundstate)
    plt.show()