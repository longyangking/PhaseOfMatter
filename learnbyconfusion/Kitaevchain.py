import numpy as np 

class Kitaevchain:
    def __init__(self,L,t,mu,iterations=1e4):
        self.L = L
        self.t = t
        self.mu = mu

        self.iterations = int(iterations)
        self.groundstate = np.zeros(L)

    def Hamiltonian(self,state):
        energy = 0
        for i in self.L:
            nexti = (i+1)%self.L
            count = 0
            if state[i] != 0:
                count += 1
            if (state[i] != 0) and (state[nexti]! = 0):
                count += 1
            if (state[nexti] != 0):

    def montecarlo(self):
        