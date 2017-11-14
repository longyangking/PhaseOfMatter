import matplotlib.pyplot as plt
from Isingmodel import Isingmodel

if __name__=='__main__':
    ising = Isingmodel((20,20),1.0,0.001,verbose=True,populations=4)
    ising.init(parallel=4)
    
    groundstate = ising.getgroundstate()
    groundenergy = ising.getgroundenergy()
    print('Ground Energy: {num}'.format(num=groundenergy))
    plt.imshow(groundstate,cmap="gray")
    plt.show()