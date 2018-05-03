""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images
import scipy.io as sio

FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, datasource, rect_truncated=False, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)
        self.rect_truncated = rect_truncated
        self.testing_size = 100
        if datasource == 'sinusoid':
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
            self.generate = self.generate_sinusoid_batch
        elif datasource == 'ball':
            # 'm1': m1, 'm2': m2, 'cr': cr
            self.m_range = config.get('m_range', [1.0, 10.0])   # m1 and m2
            self.cr_range = config.get('cr_range', [0.0, 1.0])
            self.vel_range = config.get('vel_range', [-10.0, 10.0])   # range of input velocity
            self.N_range = config.get('N_range', [1.0, 10.0]) # range of input radius (ball sizes)
            self.dim_input = 6
            self.dim_output = 4
            self.generate = self.generate_ball_batch
        elif datasource == 'ball_file':
            # 'm1': m1, 'm2': m2, 'cr': cr
            self.dim_input = 14
            self.dim_output = 4
            self.generate = self.generate_training_ball_from_file
            self.generate_test = self.generate_testing_ball_from_file
            self.ball_generator = self.ball_file_generator(batch_size)
            self.mat_contents = sio.loadmat('training_circles_v1.mat')
            self.num_total_task  = 5000
            self.input_data = np.transpose(self.mat_contents['In']).reshape((self.num_total_task, 40, 14))
            self.output_data = np.transpose(self.mat_contents['Out']).reshape((self.num_total_task, 40, 4))
            self.meta_data = np.transpose(self.mat_contents['Meta']).reshape((self.num_total_task, 40, 3))
        elif datasource == 'rect_file':
            # 'm1': m1, 'm2': m2, 'cr': cr

            self.generate = self.generate_ball_from_rect
            self.rect_generator = self.rect_file_generator(batch_size)
            self.mat_contents = sio.loadmat('training_rects_v2.mat')
            self.num_total_task  = 10000

            self.input_data = np.transpose(self.mat_contents['In']).reshape((self.num_total_task, 50, 16))
            self.output_data = np.transpose(self.mat_contents['Out']).reshape((self.num_total_task, 50, 6))
            self.meta_data = np.transpose(self.mat_contents['Meta']).reshape((self.num_total_task, 50, 3))
            if self.rect_truncated:
                self.input_data = np.delete(self.input_data, [12, 13, 14, 15], axis=2)
                self.output_data = np.delete(self.output_data, [0, 3], axis=2)
                self.dim_output = 4
                self.dim_input = 12
            else:
                self.dim_output = 6
                self.dim_input = 16

        else:
            raise ValueError('Unrecognized data source')

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return init_inputs, outputs#, amp, phase

    def generate_ball_batch(self, m1=None, m2=None, cr=None, N_norm=None, batch_size=None, num_samples_per_class=None):
        # Note train arg is not used.

        if not batch_size: batch_size = self.batch_size
        if not num_samples_per_class: num_samples_per_class = self.num_samples_per_class

        xs = np.zeros([batch_size, num_samples_per_class, self.dim_input])
        ys = np.zeros([batch_size, num_samples_per_class, self.dim_output])


        for i in range(batch_size):
            self.g_batch(m1, m2, cr, N_norm, batch_size, num_samples_per_class, xs, ys, i)
        return xs, ys


    def g_batch(self,m1, m2, cr, N_norm, batch_size, num_samples_per_class, xs, ys, i):
        # for each task (different m1, m2, cr (coefficient of restitution))
        if m1 is None: m1 = np.random.uniform(self.m_range[0], self.m_range[1])
        if m2 is None: m2 = np.random.uniform(self.m_range[0], self.m_range[1])
        if cr is None: cr = np.random.uniform(self.cr_range[0], self.cr_range[1])
        if N_norm is None: N_norm = np.random.uniform(self.N_range[0], self.N_range[1])

        for j in range(num_samples_per_class):
            u1 = np.random.uniform(self.vel_range[0], self.vel_range[1], (2, 1))
            u2 = np.random.uniform(self.vel_range[0], self.vel_range[1], (2, 1))

            theta_N = np.random.rand() * 2 * np.pi
            Nx = N_norm * np.cos(theta_N)
            Ny = N_norm * np.sin(theta_N)

            # J is the transform from standard 2D coordinate to the coordinate where +x-axis is N direction
            # i.e., J will transform standard v vector to (v_normal, v_tangential)
            J = np.array([[np.cos(theta_N), np.sin(theta_N)], [-np.sin(theta_N), np.cos(theta_N)]])
            Ju1 = J.dot(u1)
            Ju2 = J.dot(u2)
            (u1N, u1T) = tuple(Ju1)
            (u2N, u2T) = tuple(Ju2)

            if u1N - u2N <= 0:
                v1N = u1N
                v2N = u2N
            else:
                v1N = (m1 * u1N + m2 * u2N + m2 * cr * (u2N - u1N)) / (m1 + m2)
                v2N = (m1 * u1N + m2 * u2N + m1 * cr * (u1N - u2N)) / (m1 + m2)

            # transform back to standard coordinate
            v1 = J.T.dot(np.array([v1N, u1T]).reshape(-1, 1))
            v2 = J.T.dot(np.array([v2N, u2T]).reshape(-1, 1))

            cur_x = np.concatenate((u1.reshape(-1), u2.reshape(-1), np.array([Nx, Ny])))
            xs[i, j, :] = cur_x
            cur_y = np.concatenate((v1.reshape(-1), v2.reshape(-1)))
            ys[i, j, :] = cur_y

    def ball_file_generator(self, num_task):
        num_train_total_tasks = self.num_total_task - self.testing_size
        ind_array = np.arange(0, num_train_total_tasks, 1)

        current_i = 0
        while 1:
            if current_i + num_task > num_train_total_tasks:
                np.random.shuffle(ind_array)
                current_i = 0

            batch_indexes = ind_array[current_i:current_i + num_task]
            current_i = current_i + num_task
            yield (self.input_data[batch_indexes], self.output_data[batch_indexes],
                   self.meta_data[batch_indexes])

    def rect_file_generator(self, num_task):

        num_train_total_tasks = self.num_total_task - self.testing_size
        ind_array = np.arange(0, num_train_total_tasks, 1)
        current_i = 0
        while 1:
            if current_i + num_task > num_train_total_tasks:
                np.random.shuffle(ind_array)
                current_i = 0

            batch_indexes = ind_array[current_i:current_i + num_task]
            current_i = current_i + num_task

            yield (self.input_data[batch_indexes], self.output_data[batch_indexes],
                   self.meta_data[batch_indexes])

    def generate_training_ball_from_file(self):
        return next(self.ball_generator)

    def generate_testing_ball_from_file(self):
        # index = np.random.choice(range(self.num_total_task - self.testing_size,
        #                                self.num_total_task), 1)
        # return self.input_data[index], self.output_data[index], self.meta_data[index]
        index = np.random.choice(range(0,
                                       self.num_total_task-self.testing_size), 1)
        return self.input_data[index], self.output_data[index], self.meta_data[index]

    def generate_ball_from_rect(self):
        return next(self.rect_generator)

    def generate_testing_rect_from_file(self):
        index = np.random.choice(range(self.num_total_task - self.testing_size,
                                       self.num_total_task), 1)
        return self.input_data[index], self.output_data[index], self.meta_data[index]
