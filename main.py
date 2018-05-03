"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import scipy.io as sio

from dg import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'ball_file', 'sinusoid or omniglot or miniimagenet or ball')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('metatrain_iterations', 70000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 20, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 2, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'log/rect', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')

rect_truncated = False
if FLAGS.train:
    test_num_updates = 5
else:
    test_num_updates = 10


def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 1000
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5


    train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    for itr in range(resume_itr, FLAGS.metatrain_iterations):
        feed_dict = {}
        if FLAGS.datasource == 'sinusoid':
            batch_x, batch_y, amp, phase = data_generator.generate()

            if FLAGS.baseline == 'oracle':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(FLAGS.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            # ME: for each task, first 10 generated samples for training, and 11~20 for testing.
            inputa = batch_x[:, :FLAGS.update_batch_size, :]
            labela = batch_y[:, :FLAGS.update_batch_size, :]
            inputb = batch_x[:, FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

        elif FLAGS.datasource == 'ball':
            batch_x, batch_y = data_generator.generate()
            # ME: for each task, first 10 generated samples for training, and 11~20 for testing.
            inputa = batch_x[:, :FLAGS.update_batch_size, :]
            labela = batch_y[:, :FLAGS.update_batch_size, :]
            inputb = batch_x[:, FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

        elif FLAGS.datasource == "ball_file" or FLAGS.datasource == "rect_file":
            batch_x, batch_y, meta_data = data_generator.generate()
            # ME: for each task, first 10 generated samples for training, and 11~20 for testing.
            inputa = batch_x[:, :FLAGS.update_batch_size, :]
            labela = batch_y[:, :FLAGS.update_batch_size, :]
            inputb = batch_x[:, FLAGS.update_batch_size:, :]  # b used for testing
            labelb = batch_y[:, FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}


        input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            # ME: During training, only one update step, total_losses2 has only one element
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:

            print_str = 'Iteration ' + str(itr)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            # this block seems unnecessary
            batch_x, batch_y, meta_data = data_generator.generate()
            inputa = batch_x[:, :FLAGS.update_batch_size, :]
            inputb = batch_x[:, FLAGS.update_batch_size:, :]
            labela = batch_y[:, :FLAGS.update_batch_size, :]
            labelb = batch_y[:, FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

            input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]
            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 1

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    for _ in range(NUM_TEST_POINTS):
        meta_data = None
        if FLAGS.datasource == 'sinusoid':
            batch_x, batch_y, amp, phase = data_generator.generate_test(train=False)

            inputa = batch_x[:, :FLAGS.update_batch_size, :]
            inputb = batch_x[:,FLAGS.update_batch_size:, :]
            labela = batch_y[:, :FLAGS.update_batch_size, :]
            labelb = batch_y[:,FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
            result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)

        elif FLAGS.datasource == 'ball':
            batch_x, batch_y,  = data_generator.generate()
            # ME: for each task, first 10 generated samples for training, and 11~20 for testing.
            inputa = batch_x[:, :FLAGS.update_batch_size, :]
            labela = batch_y[:, :FLAGS.update_batch_size, :]
            inputb = batch_x[:, FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                         model.meta_lr: 0.0}
            result = sess.run([model.total_loss1] + model.total_losses2, feed_dict)

        else:#FLAGS.datasource == "ball_file" FLAGS.datasource == "rect_file":
            batch_x, batch_y, meta_data = data_generator.generate_test()
            # ME: for each task, first 10 generated samples for training, and 11~20 for testing.
            inputa = batch_x[:, :FLAGS.update_batch_size, :]
            labela = batch_y[:, :FLAGS.update_batch_size, :]
            inputb = batch_x[:, FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, FLAGS.update_batch_size:, :]
            # feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
            #              model.meta_lr: 0.0}
            assert len(inputa) == 1
            # result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
            tensors = model.task_rollout(inputa[0], inputb[0], labela[0], labelb[0])

            output_as_mat_file(sess, model, meta_data, batch_x, tensors)

        # metaval_accuracies.append(result)

    # metaval_accuracies = np.array(metaval_accuracies)
    # means = np.mean(metaval_accuracies, 0)
    # stds = np.std(metaval_accuracies, 0)
    # ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    # print('Mean validation accuracy/loss, stddev, and confidence intervals')
    # print(means)
    # print(stds)
    # print(ci95)

def main():




    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid':
        # ME: update_batch_size = 10 (20 samples/task); meta_batch_size = 25 (25 tasks)
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size, datasource='sinusoid')
    elif FLAGS.datasource == 'ball':
        # ME: update_batch_size = 10 (20 samples/task); meta_batch_size = 25 (25 tasks)
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size, datasource='ball')
    elif FLAGS.datasource == 'ball_file':
        # ME: update_batch_size = 10 (20 samples/task); meta_batch_size = 25 (25 tasks)
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size, datasource='ball_file')
    else: # 'rect_file"
        # ME: update_batch_size = 10 (20 samples/task); meta_batch_size = 25 (25 tasks)
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size, datasource='rect_file', rect_truncated=rect_truncated)


    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input


    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    model.construct_model()

    model.summ_op = tf.summary.merge_all()

    saver  = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    exp_string = get_exp_string(model)
    resume_itr = 0

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        # ME: test_num_updates = 10; 10 gradient updates
        test(model, saver, sess, exp_string, data_generator, test_num_updates)


def get_exp_string(model):
    exp_string = FLAGS.datasource + '.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + \
                 str(FLAGS.update_batch_size) + '.numstep' + \
                 str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.update_lr)+ \
                 "hidden_layers:" + str(tuple(model.dim_hidden)) + \
                 "truncated:" + str(rect_truncated)
    # exp_string = 'cls_5.mbs_25.ubs_20.numstep1.updatelr0.001nonorm'
    return exp_string


def output_as_mat_file(sess, model, meta_data, input_data, tensors):

    task_outputa, task_outputbs, task_lossa, task_lossesb, weights = sess.run(tensors)
    total_loss1 = task_lossa
    total_losses2 = task_lossesb
    print("weights")
    print(weights)
    with open('ball_file.mat', 'wb') as f:
        for i in range(len(model.dim_hidden) + 1):

            Wname = 'W' + str(i)
            Bname = 'B' + str(i)
            Wmat = weights['w' + str(i+1)]
            Wmat = np.transpose(Wmat)
            print(Wmat.shape)
            Bmat = weights['b' + str(i+1)]
            print(Bmat.shape)
            sio.savemat(f, {Wname: Wmat})
            sio.savemat(f, {Bname: Bmat})
        sio.savemat(f, {"Meta": list(meta_data[0][0]) + list(input_data[0][0][-4:])})

    f.close()
    result = [total_loss1] + total_losses2
    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print(result)
if __name__ == "__main__":
    main()
