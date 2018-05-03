import csv
import numpy as np
import pickle
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import scipy.io as sio

from dg import DataGenerator
from tensorflow.python.platform import flags
from keras.layers.normalization import BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation

FLAGS = flags.FLAGS

## Dataset/method options
datasource = 'ball_file'#, 'sinusoid or omniglot or miniimagenet')
# oracle means task id is input (only suitable for sinusoid)

## Training options
train_iterations = 10000#, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
update_lr = 1e-3#, 'step size alpha for inner gradient update.') # 0.1 for omniglot
mini_batch_size = 250

flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')

## Model options
stop_grad = False#, 'if True, do not use second derivatives in meta-optimization (for speed)'

## Logging, saving, and testing options
logdir = 'sl_logs'#, 'directory for summaries and checkpoints.'
resume =  True#, 'resume training if there is a model available'
train =  True#, 'True to train, False to test.'
test_iter = 10000#, 'iteration to load model (-1 for latest model)'
test_set =  False#, 'Set to true to test on the the test set, False for the validation set.'
dim_hidden = [40]

def get_training_input_label():
    mat_contents = sio.loadmat('training_rects_onetask.mat')
    input_data = np.transpose(mat_contents['In'])
    output_data = np.transpose(mat_contents['Out'])
    meta_data = np.transpose(mat_contents['Meta'])
    return input_data[:-100], output_data[:-100]

def get_testing_input_label():
    mat_contents = sio.loadmat('training_rects_onetask.mat')
    input_data = np.transpose(mat_contents['In'])
    output_data = np.transpose(mat_contents['Out'])
    meta_data = np.transpose(mat_contents['Meta'])
    return input_data[-100:], output_data[-100:]

class Experiment(object):
    '''
    For 3 graphs:
    1. MAML ball error graph in the paper
    2. X: Training samples, Y: testing error against MAML same thing same graph MAML 20 samples one gradient update to 0.5 loss, SL need
    2000 data with 200 epochs to reach same accuracy
    3. X: training iterations Y: training loss when MAML against all together train
    '''
    def __init__(self):
        self.data_generator = DataGenerator(num_samples_per_class=40,
                                            batch_size=1, datasource=datasource)

        self.model = Sequential()
        self.model.add(Dense(32, input_dim=14, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(32, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))


        self.model.add(Dense(4))


        # self.m1 = np.random.uniform(self.data_generator.m_range[0], self.data_generator.m_range[1])
        # self.m2 = np.random.uniform(self.data_generator.m_range[0], self.data_generator.m_range[1])
        # self.cr = np.random.uniform(self.data_generator.cr_range[0], self.data_generator.cr_range[1])
        # self.N_norm = np.random.uniform(self.data_generator.N_range[0], self.data_generator.N_range[1])

        # decay = 0.05/300
        # adam = optimizers.adam(lr=0.05, decay=decay)
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        print(self.model.metrics_names)



    def get_exp_string(self):
        result = "update_lr" + str(update_lr) + \
                 "mini_batch_size" + str(mini_batch_size)\
                 + "dim_hidden" + str(tuple(dim_hidden))
        return result

    def train(self):

        test_losses = []
        train_iterations_list = []
        input, label, meta_data = self.data_generator.generate()
        input, label = input[0], label[0]
        for iter in range(train_iterations):

            # self.model.fit(input, label, batch_size=250, verbose=0)
            result = self.model.train_on_batch(input, label)
            test_losses.append(result)
            train_iterations_list.append(iter)
            if result < 0.06:
                break

        return train_iterations_list, test_losses


    # def plot(self, test_iterations, test_losses):


    def test(self):
        test_losses = []
        input, label = self.data_generator.generate_ball_batch(self.m1, self.m2, self.cr, self.N_norm)
        input, label = input[0], label[0]

        loss = self.model.evaluate(input, label, batch_size=250)



        print('Mean testing loss, stddev, and confidence intervals')
        print((loss))
        return loss

    def train_sh(self, test_every_iter=1000):



        # for iter in range(train_iterations):
        input, label = get_training_input_label()

        # self.model.fit(input, label, batch_size=250, verbose=0)
        history = self.model.fit(input, label, batch_size=250, verbose=1, epochs=1000)
        self.test_sh()



    # def plot(self, test_iterations, test_losses):


    def test_sh(self):
        input, label = get_testing_input_label()

        loss = self.model.evaluate(input, label, batch_size=250)



        print('Mean testing loss, stddev, and confidence intervals')
        print((loss))
        return loss
if __name__ == "__main__":
    exp = Experiment().train()