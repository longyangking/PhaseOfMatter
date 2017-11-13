import numpy as np 

class Isingmodel:
    def __init__(self,size,J,iterations=1e4):
        self.size = size
        self.state = np.random.choice([-1,1],size=size)
        self.J = J
        self.iterations = iterations
        self.groundenergy = None
        self.groundstate = None

    def Hamiltonian(self):
        '''
        Calculate total energy of whole spin system
        '''
        energy = 0
        Nx,Ny = self.size
        for i in range(Nx):
            for j in range(Ny):
                nexti = (i+1)%Nx
                nextj = (j+1)%Ny
                pali = self.state[i,j]*self.state[nexti,j]
                palj = self.state[i,j]*self.state[i,nextj]
                energy = -self.J*(pali + palj)
        return energy

    def spinflip(self,index):
        '''
        Spin flip operation
        '''
        spin = self.state[index]
        if spin == 1:
            self.state[index] = -1
        else:
            self.state[index] = 1

    def montecarlo(self,iterations=None):
        '''
        Monte Carlo Simulations to get ground state
        '''
    
    def groundstate(self):
        '''
        Calculate ground state
        '''

        return self.groundstate

    def getgroundstate(self):
        return self.groundstate

    def getgroundenergy(self):
        return self.groundenergy

    def getgroundstate(self):
        return self.groundstate