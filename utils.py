# Copyright (c) Microsoft Corporation.  Licensed under the MIT license.
import numpy as np
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS


def analyze(samples, X):
    samples = -samples
    X = -X
    samples = np.sort(samples)
    results = np.zeros_like(X)
    for i in range(len(results)):
        count = 0
        while count < len(samples) and samples[count] < X[i]:
            count += 1
        results[i] = count / len(samples)
    return results


def label_smooth(y, weight=0.9):
    # requires y to be one_hot!
    return tf.clip_by_value(y, clip_value_min=(1.0 - weight) / (FLAGS.num_classes - 1.), clip_value_max=weight)


def get_weights_path():
    prefix = os.path.join('results', 'weights')
    folder = FLAGS.model
    if FLAGS.adv_std != 8:
        folder += "_std{}".format(FLAGS.adv_std)

    if FLAGS.adversarial:
        folder = folder + '_adv'

    if FLAGS.adversarial_BIM:
        folder = folder + '_advBIM'

    if FLAGS.label_smooth:
        folder = folder + '_ls'

    folder = FLAGS.dataset + '_' + folder
    path = os.path.join(prefix, folder)
    if not os.path.exists(path):
        os.makedirs(path)

    ckpt_path = os.path.join(path, 'model.ckpt')
    return path, ckpt_path


def get_input_data_path():
    if FLAGS.mode == "train" or FLAGS.mode == "attack":
        data = "clean"
        data = FLAGS.dataset + '_' + data
    elif FLAGS.mode == "eval":
        # if needs to evaluate adversarial or denoised data, needs
        # to specify it explicitly in FLAGS.noise_type
        data = FLAGS.noise_type

    prefix = os.path.join("results", "data")
    return os.path.join(prefix, data)


def verify(text):
    print(text)
    while True:
        command = input()
        if command.lower() == 'y':
            return True
        elif command.lower() == 'n':
            return False
        else:
            print("Please enter y or n")


def save_denoised_data(x, y, output_path, data_set='cifar'):
    file_name = "test_batch"
    full_path = os.path.join(output_path, file_name)
    if os.path.exists(full_path):
        ans = verify("Warning: output data already exist. Do you want to override? (y/n)")
        if ans:
            os.unlink(full_path)
        else:
            return False

    os.makedirs(output_path, exist_ok=True)
    if data_set == 'cifar':
        x = np.transpose(x, (0, 3, 1, 2))

    import pickle
    with open(full_path, 'wb') as fout:
        pickle.dump({'data': x, 'labels': y}, fout)


def get_attack_output_name(type):
    eps = FLAGS.eps
    path = FLAGS.model

    if FLAGS.adversarial_BIM:
        path += '_advBIM'

    if FLAGS.label_smooth:
        path += '_ls'

    if FLAGS.feature_squeeze:
        path += '_fs'

    path += "_" + type
    if FLAGS.adversarial and type != "clean":
        path += "_adv"
        if FLAGS.adv_std != 8:
            path += '_std{}'.format(FLAGS.adv_std)
    path += "_eps{}".format(eps)
    path = FLAGS.dataset + '_' + path
    prefix = os.path.join("results", "data", path)
    file_name = "test_batch"
    full_path = os.path.join(prefix, file_name)
    return prefix, full_path


def save_output_data(x, y, type):
    prefix, full_path = get_attack_output_name(type)
    if os.path.exists(full_path):
        ans = verify("Warning: output data already exist. Do you want to override? (y/n)")
        if ans:
            os.unlink(full_path)
        else:
            return False

    os.makedirs(prefix, exist_ok=True)
    if FLAGS.dataset == 'cifar':
        x = np.transpose(x, (0, 3, 1, 2))

    import pickle
    with open(full_path, 'wb') as fout:
        pickle.dump({'data': x, 'labels': y}, fout)


def per_image_standardization(images):
    image_mean, image_std = tf.nn.moments(images, axes=[1, 2, 3])
    image_std = tf.sqrt(image_std)[:, None, None, None]
    images_standardized = (images - image_mean[:, None, None, None]) / tf.maximum(image_std, 1.0 / np.sqrt(
        FLAGS.image_size ** 2 * 3))
    return images_standardized


def random_flip_left_right(images):
    images_flipped = tf.reverse(images, axis=[2])
    flip = tf.cast(tf.contrib.distributions.Bernoulli(probs=tf.ones((tf.shape(images)[0],)) * 0.5).sample(), tf.bool)
    final_images = tf.where(flip, x=images, y=images_flipped)
    return final_images


def feature_squeeze(images, dataset='cifar'):
    # color depth reduction
    if dataset == 'cifar':
        npp = 2 ** 5
    elif dataset == 'f_mnist':
        npp = 2 ** 3

    npp_int = npp - 1
    images = images / 255.
    x_int = tf.rint(tf.multiply(images, npp_int))
    x_float = tf.div(x_int, npp_int)
    return median_filtering_2x2(x_float, dataset=dataset)

def median_filtering_2x2(images, dataset='cifar'):
    def median_filtering_layer_2x2(channel):
        top = tf.pad(channel, paddings=[[0, 0], [1, 0], [0, 0]], mode="REFLECT")[:, :-1, :]
        left = tf.pad(channel, paddings=[[0, 0], [0, 0], [1, 0]], mode="REFLECT")[:, :, :-1]
        top_left = tf.pad(channel, paddings=[[0, 0], [1, 0], [1, 0]], mode="REFLECT")[:, :-1, :-1]
        comb = tf.stack([channel, top, left, top_left], axis=3)
        return tf.nn.top_k(comb, 2).values[..., -1]

    if dataset == 'cifar':
        c0 = median_filtering_layer_2x2(images[..., 0])
        c1 = median_filtering_layer_2x2(images[..., 1])
        c2 = median_filtering_layer_2x2(images[..., 2])
        return tf.stack([c0, c1, c2], axis=3)
    elif dataset == 'mnist':
        return median_filtering_layer_2x2(images[..., 0])[..., None]

def normalize_image(images):
    return (images.astype(np.int32) - 127.5) / 127.5


def unnormalize_image(images):
    return images * 127.5 + 127.5
