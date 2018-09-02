import matplotlib.pyplot as plt
from Isingmodel import Isingmodel

if __name__=='__main__':
    ising = Isingmodel((100,100),1.0,0.4,verbose=True,populations=8)
    ising.init(parallel=4)
    import matplotlib.pyplot as plt 
    groundstate = ising.get_groundstate()
    groundenergy = ising.get_groundenergy()
    print('Ground Energy: {num}'.format(num=groundenergy))
    plt.imshow(groundstate,cmap='gray')
    plt.show()