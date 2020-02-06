# Copyright (c) Microsoft Corporation.  Licensed under the MIT license.
"""
Evaluating bits per dimensions for trained PixelCNN++ models.
Code based on OpenAI PixelCNN++
"""
import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf

import pixel_cnn_pp.nn as nn
from pixel_cnn_pp.model import model_spec
import data.cifar10_data as cifar10_data
import data.f_mnist_data as f_mnist_data

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-o', '--save_dir', type=str, default='results/weights/pxpp',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str,
                    default='cifar', help='Can be cifar | f_mnist')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu',
                    help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
# optimization
parser.add_argument('-l', '--learning_rate', type=float,
                    default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=10,
                    help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100,
                    help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5,
                    help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-g', '--nr_gpu', type=int, default=1,
                    help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995,
                    help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')

parser.add_argument('--noise_type', type=str, default='clean',
                    help='Noise type is a string formated as '
                         'DatasetName_ModelName[_fs_|_ls_|_advBIM_]_AttackMethod[_adv_]_epsNumber1[_denoised_]_epsNumber2.'
                         'Here items within [] are optional. Separator | means only one of them can be chosen.'
                         'DatasetName can be cifar or f_mnist.'
                         'ModelName can be resnet or vgg.'
                         'AttackMethod can be clean or random or FGSM or BIM or deepfool or CW.'
                         'Number1 is an integer representing the attack eps.'
                         'Number2 is an integer representing the defense eps.')

args = parser.parse_args()
args.class_conditional = False
print('input args:\n', json.dumps(vars(args), indent=4,
                                  separators=(',', ':')))  # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits
DataLoader = {'cifar': cifar10_data.DataLoader,
              'f_mnist':f_mnist_data.DataLoader}[args.data_set]

input_data_path = os.path.join("results", "data", args.noise_type)

eval_data = DataLoader(input_data_path, 'test', args.batch_size *
                       args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
obs_shape = eval_data.get_observation_size()  # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size,) + obs_shape)
      for i in range(args.nr_gpu)]

# if the model is class-conditional we'll set up label placeholders +
# one-hot encodings 'h' to condition on
if args.class_conditional:
    num_labels = eval_data.get_num_labels()
    y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
    h_init = tf.one_hot(y_init, num_labels)
    y_sample = np.split(
        np.mod(np.arange(args.batch_size * args.nr_gpu), num_labels), args.nr_gpu)
    h_sample = [tf.one_hot(tf.Variable(
        y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
    ys = [tf.placeholder(tf.int32, shape=(args.batch_size,))
          for i in range(args.nr_gpu)]
    hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]
else:
    h_init = None
    h_sample = [None] * args.nr_gpu
    hs = h_sample

# If dataset is f_mnist, then reduce num_filters by 4
if args.data_set == 'f_mnist':
    args.nr_filters //= 4

# create the model
model_opt = {'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters,
             'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity,
             'data_set': args.data_set}

model = tf.make_template('model', model_spec)

# run once for data dependent initialization of parameters
gen_par = model(x_init, h_init, init=True,
                dropout_p=args.dropout_p, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))

# get loss gradients over multiple GPUs
grads = []
loss_gen = []
logits_gen = []
loss_gen_test = []
loss_gen_test_per_sample = []
logits_gen_test = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # train
        gen_par = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        logits_gen.append(nn.mix_logistic_to_logits(xs[i], gen_par, data_set=args.data_set))
        loss_gen.append(nn.xent_from_softmax(xs[i], gen_par, data_set=args.data_set))
        # gradients
        grads.append(tf.gradients(loss_gen[i], all_params))
        # test
        gen_par = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
        logits_gen_test.append(nn.mix_logistic_to_logits(xs[i], gen_par, data_set=args.data_set))
        loss_gen_test_per_sample.append(nn.xent_from_softmax(xs[i], gen_par, sum_all=False, data_set=args.data_set))

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1, args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    optimizer = tf.group(nn.adam_updates(
        all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[0] / (args.nr_gpu * np.log(2.) * np.prod(obs_shape) * args.batch_size)
bits_per_dim_test = tf.concat([x / (np.log(2.) * np.prod(obs_shape)) for x in loss_gen_test_per_sample],
                              axis=0)

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x, y = data
    else:
        x = data
        y = None
    # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    x = np.cast[np.float32]((x - 127.5) / 127.5)
    if init:
        feed_dict = {x_init: x}
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if y is not None:
            y = np.split(y, args.nr_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
    return feed_dict


# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('starting training')
test_bpd = []
lr = args.learning_rate

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    begin = time.time()

    # init
    # manually retrieve exactly init_batch_size examples
    feed_dict = make_feed_dict(
        eval_data.next(args.init_batch_size), init=True)
    eval_data.reset()  # rewind the iterator back to 0 to do one full epoch
    sess.run(initializer, feed_dict)
    print('initializing the model...')
    ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    # compute likelihood over test data
    test_losses = []
    for batch, d in enumerate(eval_data):
        feed_dict = make_feed_dict(d)
        l = sess.run(bits_per_dim_test, feed_dict)
        print("batch {}, mean loss: {:.6f}".format(batch, np.mean(l)))
        test_losses.extend(l)

    test_losses = np.asarray(test_losses)
    test_loss_gen = np.mean(test_losses)
    np.save(os.path.join("results", "data", args.noise_type, "bits_per_dim.npy"), test_losses)
    test_bpd.append(test_loss_gen)

    # log progress to console
    print("Elapsed time %ds, test bits_per_dim = %.4f" % (
        time.time() - begin, test_loss_gen))
    sys.stdout.flush()
