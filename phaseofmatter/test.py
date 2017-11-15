import matplotlib.pyplot as plt
from Isingmodel import Isingmodel

if __name__=='__main__':
    ising = Isingmodel((100,100),1.0,1.0,verbose=True,populations=8)
    ising.init(parallel=4)
    import matplotlib.pyplot as plt 
    groundstate = ising.getgroundstate()
    groundenergy = ising.getgroundenergy()
    print('Ground Energy: {num}'.format(num=groundenergy))
    plt.imshow(groundstate,cmap='gray')
    plt.show()