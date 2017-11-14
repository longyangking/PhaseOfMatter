import numpy as np 
from multiprocessing import Pool

class Isingmodel:
    def __init__(self,size,J,beta,iterations=1e4,populations=10,verbose=False):
        self.size = size
        self.states = list()
        for i in range(populations):
            self.states.append(np.random.choice([-1,1],size=size))
        self.J = J
        self.iterations = int(iterations)
        self.groundenergy = None
        self.groundstate = None
        self.beta = beta
        self.verbose = verbose

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
        iterations = self.iterations

        newstate = state.copy()
        (Nx,Ny) = state.shape

        for i in range(iterations):
            energy = self.Hamiltonian(state)
            x = np.random.randint(Nx)
            y = np.random.randint(Ny)
            self.spinflip((x,y),newstate)
            energynew = self.Hamiltonian(newstate)

            #mu = np.min([np.exp(-self.beta*(energynew-energy)),1])
            if energynew < energy:
                self.spinflip((x,y),state)
            else:
                if np.random.random() < np.exp(-self.beta*(energynew-energy)):
                    self.spinflip((x,y),state)
            
            if self.verbose:
                if (i+1)%(int(iterations/10)) == 0:
                    print('Monte Carlo calculating...{num}%'.format(num=100*(i+1)/iterations))

        energy = self.Hamiltonian(state)
        return energy

    def init(self,parallel=0):
        '''
        Calculate ground state
        '''
        energies = list()
        if parallel:
            print('Compuation in parallel with core: {num}'.format(num=parallel))
            pool = Pool(4)  
            energies = pool.map(self.montecarlo,self.states) 
            pool.close()
            pool.join()
        else:
            count = 1
            for state in self.states:
                energies.append(self.montecarlo(state))
                if self.verbose:
                    print('Series: {count}th finished!'.format(count=count))
                count += 1
                    
        index = np.argmin(energies)
        self.groundenergy = energies[index]
        self.groundstate = self.states[index]
        return True

    def getgroundstate(self):
        return self.groundstate

    def getgroundenergy(self):
        return self.groundenergy

    def getgroundstate(self):
        return self.groundstate


if __name__=='__main__':
    ising = Isingmodel((4,4),1.0,0.3,verbose=True,populations=20)
    ising.init(parallel=4)
    import matplotlib.pyplot as plt 
    groundstate = ising.getgroundstate()
    groundenergy = ising.getgroundenergy()
    print('Ground Energy: {num}'.format(num=groundenergy))
    plt.imshow(groundstate)
    plt.show()