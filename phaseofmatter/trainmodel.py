import numpy as np 
import h5py
from multiprocessing import Pool
from Isingmodel import Isingmodel
from ml import MLModel
import matplotlib.pyplot as plt

def calmodel(model):
    model.init()
    groundenergy = model.get_groundenergy()
    #print("Ground energy [{0}]".format(groundenergy))
    groundstate = model.get_groundstate()
    return groundstate

if __name__ == "__main__":
    print("Start to prepare data...")

    N = 10000
    n_train = int(N*0.8)
    n_test = N - n_train

    cores = 24  # 24 CPU cores used to calculate Ising model
    isings = list()
    Nx,Ny = 50,50
    size = (Nx,Ny)
    J = 1.0
    betas = np.linspace(1, 0, N)

    for i in range(N):
        ising = Isingmodel(size,J,betas[i],populations=1)
        isings.append(ising)

    print('Calculation in parallel with cores [{0}]'.format(cores))
    pool = Pool(cores)
    groundstates = pool.map(calmodel,isings) 
    pool.close()
    pool.join()

    filename = 'data.h5'
    print('Save the data with file [{0}] in h5 format'.format(filename))
    h5file = h5py.File(filename,'w')
    h5file.create_dataset('groundstates',data=groundstates)
    h5file.create_dataset('betas',data=betas)
    h5file.close()

    state_shape = (Nx, Ny, 1)
    groundstates = np.array(groundstates).reshape(-1, *state_shape)
    idx = np.arange(N)
    np.random.shuffle(idx)
    Xs_train = groundstates[idx[:n_train]]
    Xs_test = groundstates[idx[n_train:]]
    betas = np.array(betas).reshape(-1, 1)
    values = betas < 0.5
    ys_train = values[idx[:n_train]]
    ys_test = values[idx[n_train:]]
    print("Training data size: [{0}]".format(Xs_train.shape))

    print("Training model with state shape: [{0}]".format(state_shape))
    mlmodel = MLModel(state_shape=state_shape, verbose=1)
    mlmodel.fit(Xs_train, ys_train, epochs=100, batch_size=32)
    filename = 'model.h5'
    print("Saving model with file name [{0}]".format(filename))
    mlmodel.save_model(filename)

    print("Testing model with data size: [{0}]".format(Xs_test.shape))
    ys_pred = mlmodel.predict(Xs_test)
    loss, acc = mlmodel.evaluate(Xs_test, ys_test)
    print("Loss : [{0:.4f}]; Acc : [{1:.2f}%]".format(loss, acc*100))

    plt.figure()
    plt.scatter(range(N), values, label="Ground Truth")
    values_pred = mlmodel.predict(groundstates)
    plt.plot(range(N), values_pred, c='r',label="Predicted by ML")
    plt.xlabel("Sample index")
    plt.ylabel("Phase Transition")
    plt.legend()
    plt.show()