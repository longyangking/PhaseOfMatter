from isingnet import PhaseDetector
import numpy as np 
import h5py

filename = 'data.h5'
data = h5py.File(filename,'r')

groundstates = data['groundstates']
N = len(groundstates)
input_shape = groundstates[0].shape

x_train = 
y_train = 

phasedetector = PhaseDetector()
phasedetector.init(input_shape=input_shape,num_classes=1)
phasedetector

data.close()