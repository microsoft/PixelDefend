# Copyright (c) Microsoft Corporation.  Licensed under the MIT license.
"""
Use fast pixelcnn++ to speed up decoding.

"""
import fast_pixel_cnn_pp.model as model
import fast_pixel_cnn_pp.fast_nn as fast_nn
import fast_pixel_cnn_pp.plotting as plotting
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import time
import data.cifar10_data as cifar10_data
import data.f_mnist_data as f_mnist_data

from utils import *
from shutil import rmtree
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    '-b',
    '--batch_size',
    type=int,
    default=250,
    help='Number of images to generate simultaneously')

parser.add_argument(
    '-s', '--seed', type=int, default=2702, help='Seed for random generation')
parser.add_argument(
    '-c',
    '--checkpoint',
    type=str,
    default=None,
    help='Location of the pretrained checkpoint')
parser.add_argument(
    '-v',
    '--save_dir',
    type=str,
    default='results/logs/pxpp',
    help='Location to save generated images to')
parser.add_argument(
    '-d',
    '--data_set',
    type=str,
    default='cifar',
    help='Dataset to use. Can be cifar | f_mnist'
)
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu',
                    help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument(
    '--noise_type',
    type=str,
    default='random',
    help='Noise type is some folder name in results/data/')
parser.add_argument('--train', action="store_true", help="whether to decode training dataset or not")
parser.add_argument('--eps', type=int, default=10, help='The epsilon for denoising image')
parser.add_argument('--pxl_fixed', action='store_true', help='Turn on pixel-level adaptive eps (fixed)')
args = parser.parse_args()

if args.data_set == 'f_mnist':
    args.nr_filters //= 4

model_opt = {'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters,
             'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity,
             'data_set': args.data_set, 'pxl_fixed':args.pxl_fixed}

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

def denoise(data, output_data_path):
    if os.path.exists(output_data_path):
        ans = verify("Denoised data already exist in {}. Override? (y/n)".format(output_data_path))
        if ans:
            rmtree(output_data_path)
            os.makedirs(output_data_path)
        else:
            raise FileExistsError()
    else:
        os.makedirs(output_data_path)

    with open(os.path.join(output_data_path, 'hps.txt'), 'w') as fout:
        json.dump(vars(args), fout)

    print('Creating model')
    if args.data_set == 'cifar':
        input_channels = 4  # 3 channels for RGB and 1 channel of all ones
        args.image_size = 32
    elif args.data_set == 'f_mnist':
        input_channels = 2
        args.image_size = 28
    else:
        raise NotImplementedError("Dataset {} is not supported!".format(args.data_set))

    image_size = (args.batch_size, args.image_size, args.image_size, input_channels)

    row_input = tf.placeholder(tf.float32, [args.batch_size, 1, args.image_size, input_channels], name='row_input')
    pixel_input = tf.placeholder(tf.float32, [args.batch_size, 1, 1, input_channels], name='pixel_input')
    ref_img = tf.placeholder(tf.float32, [args.batch_size, 1, 1, input_channels - 1], name="refer_image")
    row_id = tf.placeholder(tf.int32, [], name='row_id')
    col_id = tf.placeholder(tf.int32, [], name='col_id')
    ema = tf.train.ExponentialMovingAverage(0.9995)

    model_spec = tf.make_template('model', model.model_spec)
    sample, fast_nn_out, v_stack = model_spec(row_input, pixel_input, row_id, col_id, image_size, seed=args.seed,
                                              ref_img=ref_img, decode_eps=args.eps, denoise=True, **model_opt)

    all_cache_variables = [v for v in tf.global_variables() if 'cache' in v.name]
    initialize_cache = tf.variables_initializer(all_cache_variables)
    reset_cache = fast_nn.reset_cache_op()

    vars_to_restore = {
        k: v
        for k, v in ema.variables_to_restore().items() if 'cache' not in k
    }
    saver = tf.train.Saver(vars_to_restore)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(initialize_cache)

    if args.checkpoint is None:
        prefix = os.path.join("results", "weights", "pxpp")
        args.checkpoint = os.path.join(prefix, "params_{}.ckpt".format(args.data_set))

    print('Loading checkpoint %s' % args.checkpoint)
    saver.restore(sess, args.checkpoint)

    denoised_data = []
    for batch, datum in enumerate(data):
        x, y = datum
        x = normalize_image(x)

        output_images = np.copy(x)
        print('Denoising {}th batch of images'.format(batch + 1))
        sess.run(reset_cache)
        start_time = time.time()
        for row in range(args.image_size):
            # Implicit downshift.
            if row == 0:
                x_row_input = np.zeros((args.batch_size, 1, args.image_size, input_channels))
            else:
                x_row_input = output_images[:, (row - 1):row, :, :]
                x_row_input = np.concatenate((x_row_input, np.ones((args.batch_size, 1, args.image_size, 1))), axis=3)

            sess.run(v_stack, {row_input: x_row_input, row_id: row})

            for col in range(args.image_size):
                # Implicit rightshift.
                if col == 0:
                    x_pixel_input = np.zeros((args.batch_size, 1, 1, input_channels))
                else:
                    x_pixel_input = output_images[:, row:(row + 1), (col - 1):col, :]
                    x_pixel_input = np.concatenate((x_pixel_input, np.ones((args.batch_size, 1, 1, 1))), axis=3)

                feed_dict = {
                    row_id: row,
                    col_id: col,
                    pixel_input: x_pixel_input,
                    ref_img: output_images[:, row:(row + 1), col:(col + 1), :]
                }
                pixel_output = sess.run(sample, feed_dict)
                output_images[:, row:(row + 1), col:(col + 1), :] = pixel_output

        end_time = time.time()
        print('Time elapsed: %.2f seconds' % (end_time - start_time,))

        if batch == 0:
            plt.close('all')
            image_tile = plotting.img_tile(output_images, border_color=1.0, stretch=True)
            plotting.plot_img(image_tile)
            plt.savefig(os.path.join(args.save_dir, args.noise_type + '_denoised_eps_%d.png' % args.eps))

        denoised_data.extend(unnormalize_image(output_images).astype(np.uint8))

    denoised_data = np.array(denoised_data)
    save_denoised_data(denoised_data, data.labels, output_data_path, data_set=args.data_set)


def main():
    rng = np.random.RandomState(args.seed)
    tf.set_random_seed(args.seed)

    input_data_path = os.path.join("results", "data", args.noise_type)

    if args.data_set == "cifar":
        if not args.train:
            noisy_data = cifar10_data.DataLoader(input_data_path, 'test', args.batch_size, rng=rng, shuffle=False,
                                                 return_labels=True)
        else:
            noisy_data = cifar10_data.DataLoader(input_data_path, 'train', args.batch_size, rng=rng, shuffle=False,
                                                 return_labels=True)
    elif args.data_set == 'f_mnist':
        if not args.train:
            noisy_data = f_mnist_data.DataLoader(input_data_path, 'test', args.batch_size, rng=rng, shuffle=False,
                                               return_labels=True)
        else:
            noisy_data = f_mnist_data.DataLoader(input_data_path, 'train', args.batch_size, rng=rng, shuffle=False,
                                               return_labels=True)
    else:
        raise NotImplementedError("Dataset {} is not supported!".format(args.data_set))

    if not args.train:
        output_data_path = input_data_path + '_denoised_eps{}'.format(args.eps)
        if args.pxl_fixed:
            output_data_path = input_data_path + '_denoise_pxl_fixed_eps{}'.format(args.eps)
    else:
        output_data_path = input_data_path + '_train_denoised_eps{}'.format(args.eps)
    denoise(noisy_data, output_data_path)


if __name__ == '__main__':
    main()
