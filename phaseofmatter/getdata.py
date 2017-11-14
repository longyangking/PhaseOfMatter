from Isingmodel import Isingmodel
import numpy as np 
import h5py
from multiprocessing import Pool
import matplotlib.pyplot as plt

def initmodel(model):
    model.init()
    groundenergy = model.getgroundenergy()
    print('Ground energy: {num}'.format(num=groundenergy))
    return model.getgroundstate()

if __name__=='__main__':
    N = 2
    core = 4
    isings = list()
    Nx,Ny = 4,4
    size = (Nx,Ny)
    J = 1.0
    beta = 0.3

    for i in range(N):
        ising = Isingmodel(size,J,beta,populations=20)
        isings.append(ising)

    pool = Pool(core)  
    groundstates = pool.map(initmodel,isings) 
    pool.close()
    pool.join()

    #groundstates = np.zeros((N,Nx*Ny))
    #for i in range(N):
    #    ising = isings[i]

    #    groundstate = ising.getgroundstate()
    #    groundstates[i,:] = groundstate.flatten()
    groundstate = groundstates[0]
    plt.imshow(groundstate)
    plt.show()

    filename = 'data.h5'
    h5file = h5py.File(filename,'w')
    h5file.create_dataset('groundstates',data=groundstates)
    h5file.close()