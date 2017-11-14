#from isingnet import PhaseDetector
import numpy as np 
import h5py

filename = 'data.h5'
data = h5py.File(filename,'r')

groundstates = data['groundstates']
N = len(groundstates)
input_shape = groundstates[0].shape



data.close()