from keras.models import Sequential
from keras.layers import Dense
import scipy.io as sio
import numpy as np


mat_contents = sio.loadmat('training_rects_v1.mat')
num_total_task = 5000
input_data = np.transpose(mat_contents['In']).reshape((5000, 40, 16))
output_data = np.transpose(mat_contents['Out']).reshape((5000, 40, 6))
meta_data = np.transpose(mat_contents['Meta']).reshape((5000, 40, 3))
# c = np.concatenate((input_data, output_data, meta_data), axis=1)
ind_array = np.arange(0, 5000, 1)
print("s")