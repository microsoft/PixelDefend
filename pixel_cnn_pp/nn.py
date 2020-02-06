# Copyright (c) Microsoft Corporation.  Licensed under the MIT license.
"""
Various tensorflow utilities
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope


def int_shape(x):
    return list(map(int, x.get_shape()))


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))


def log_sum_exp(x, axis=None):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    ax = len(x.get_shape()) - 1 if axis is None else axis
    m = tf.reduce_max(x, ax)
    m2 = tf.reduce_max(x, ax, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), ax))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x - m), axis, keep_dims=True))


def discretized_mix_logistic_loss(x, l, sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(
        x)  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l)  # predicted distribution, e.g. (B,32,32,100)
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = tf.maximum(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix])
    m2 = tf.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                    * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                    coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = tf.concat([tf.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], 3)

    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = tf.nn.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min,
                                                            tf.where(cdf_delta > 1e-5,
                                                                     tf.log(tf.maximum(cdf_delta, 1e-12)),
                                                                     log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs, 3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs), [1, 2])

def mix_logistic_to_logits_mnist(x, l):
    """
    Bin the output to 256 discrete values. This discretizes the continuous mixture of logistic distribution
    output to categorical distributions.

    Assumptions:
    * The input has only 1 channel
    * The input has be reshaped to (batch_size, image_size, image_size, 1)
    * The input has scale [-1, 1]
    """

    xs = int_shape(
        x)  # true image (i.e. labels) to regress to, e.g. (B,28,28,1)
    ls = int_shape(l)  # predicted distribution, e.g. (B,28,28,100)
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 2])
    means = l[:, :, :, :, :nr_mix]
    log_scales = tf.maximum(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels

    inv_stdv = tf.exp(-log_scales)

    colors = tf.linspace(-1., 1., 257)
    color_maps = tf.zeros(xs + [1, 1]) + colors
    means = tf.expand_dims(means, axis=5)
    inv_stdv = tf.expand_dims(inv_stdv, axis=5)
    color_cdfs = tf.nn.sigmoid((color_maps[..., 1:-1] - means) * inv_stdv)
    color_pdfs = color_cdfs[..., 1:] - color_cdfs[..., :-1]
    normalized_0 = (color_maps[..., 1:2] - means) * inv_stdv
    normalized_255 = (color_maps[..., -2:-1] - means) * inv_stdv
    color_log_cdf0 = normalized_0 - tf.nn.softplus(normalized_0)
    color_log_cdf255 = -tf.nn.softplus(normalized_255)

    color_mids = tf.linspace(-1., 1., 513)[3:-2:2]
    color_mid_maps = tf.zeros(xs + [1, 1]) + color_mids
    color_mid_maps = inv_stdv * (color_mid_maps - means)
    color_mid_map_log_pdfs = color_mid_maps - tf.expand_dims(log_scales, axis=5) - 2. * tf.nn.softplus(color_mid_maps)

    color_log_pdfs = tf.where(color_pdfs > 1e-5,
                              x=tf.log(tf.maximum(color_pdfs, 1e-12)),
                              y=color_mid_map_log_pdfs - np.log(127.5))

    # color_log_pdfs = tf.log(tf.maximum(color_pdfs, 1e-12))

    color_log_probs = tf.concat([color_log_cdf0, color_log_pdfs, color_log_cdf255], axis=5)

    color_log_probs = color_log_probs + log_prob_from_logits(logit_probs)[:, :, :, None, :, None]

    return log_sum_exp(color_log_probs, axis=4)


def mix_logistic_to_logits(x, l, data_set='cifar'):
    """
    Bin the output to 256 discrete values. This discretizes the continuous mixture of logistic distribution
    output to categorical distributions.
    """
    if data_set == 'f_mnist':
        return mix_logistic_to_logits_mnist(x, l)

    if data_set != 'cifar':
        raise NotImplementedError("Dataset {} not supported!".format(data_set))

    xs = int_shape(
        x)  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l)  # predicted distribution, e.g. (B,32,32,100)
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = tf.maximum(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix])
    m2 = tf.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                    * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                    coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = tf.concat([tf.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], 3)

    inv_stdv = tf.exp(-log_scales)

    colors = tf.linspace(-1., 1., 257)
    color_maps = tf.zeros(xs + [1, 1]) + colors
    means = tf.expand_dims(means, axis=5)
    inv_stdv = tf.expand_dims(inv_stdv, axis=5)
    color_cdfs = tf.nn.sigmoid((color_maps[..., 1:-1] - means) * inv_stdv)
    color_pdfs = color_cdfs[..., 1:] - color_cdfs[..., :-1]
    normalized_0 = (color_maps[..., 1:2] - means) * inv_stdv
    normalized_255 = (color_maps[..., -2:-1] - means) * inv_stdv
    color_log_cdf0 = normalized_0 - tf.nn.softplus(normalized_0)
    color_log_cdf255 = -tf.nn.softplus(normalized_255)

    color_mids = tf.linspace(-1., 1., 513)[3:-2:2]
    color_mid_maps = tf.zeros(xs + [1, 1]) + color_mids
    color_mid_maps = inv_stdv * (color_mid_maps - means)
    color_mid_map_log_pdfs = color_mid_maps - tf.expand_dims(log_scales, axis=5) - 2. * tf.nn.softplus(color_mid_maps)

    color_log_pdfs = tf.where(color_pdfs > 1e-5,
                              x=tf.log(tf.maximum(color_pdfs, 1e-12)),
                              y=color_mid_map_log_pdfs - np.log(127.5))

    # color_log_pdfs = tf.log(tf.maximum(color_pdfs, 1e-12))

    color_log_probs = tf.concat([color_log_cdf0, color_log_pdfs, color_log_cdf255], axis=5)

    color_log_probs = color_log_probs + log_prob_from_logits(logit_probs)[:, :, :, None, :, None]

    return log_sum_exp(color_log_probs, axis=4)


def xent_from_softmax(x, l, sum_all=True, no_sum=False, data_set='cifar'):
    logits = mix_logistic_to_logits(x, l, data_set=data_set)
    recover_x = (x * 127.5) + 127.5
    recover_x = tf.cast(recover_x, dtype=tf.int32)
    recover_labels = tf.one_hot(recover_x, depth=256)
    if no_sum:
        return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=recover_labels)
    if sum_all:
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=recover_labels))
    else:
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=recover_labels),
                             axis=[1, 2, 3])

def decode_from_softmax_with_box_constraint_mnist(x, l, eps):
    '''
    Greedy decoding MNIST version. Assumes number of channel = 1
    '''
    recover_img = tf.cast(x * 127.5 + 127.5, dtype=tf.int32)
    lb = tf.maximum(recover_img - eps, 0)
    ub = tf.minimum(recover_img + eps, 255)
    template = tf.range(0, 256, dtype=tf.int32) + tf.zeros_like(recover_img, dtype=tf.int32)[..., None]

    lb = lb[..., None] + tf.zeros_like(template, dtype=tf.int32)
    ub = ub[..., None] + tf.zeros_like(template, dtype=tf.int32)
    template = tf.cast(tf.logical_or(tf.less(template, lb), tf.greater(template, ub)), tf.float32)

    logits0 = mix_logistic_to_logits(x, l, data_set='f_mnist')
    logits0 -= template * 1e30
    x0 = (tf.cast(tf.argmax(logits0, axis=4), dtype=tf.float32) - 127.5) / 127.5

    return x0

def decode_from_softmax_with_box_constraint(x, l, eps, data_set='cifar'):
    '''
    Greedy decoding. Equivalent to beam search with size = 1
    Note: The output is a unnormalized image
    '''
    if data_set == 'f_mnist':
        return decode_from_softmax_with_box_constraint_mnist(x, l, eps)

    if data_set != 'cifar':
        raise NotImplementedError("Dataset {} is not supported!".format(data_set))

    recover_img = tf.cast(x * 127.5 + 127.5, dtype=tf.int32)
    lb = tf.maximum(recover_img - eps, 0)
    ub = tf.minimum(recover_img + eps, 255)
    template = tf.range(0, 256, dtype=tf.int32) + tf.zeros_like(recover_img, dtype=tf.int32)[..., None]

    lb = lb[..., None] + tf.zeros_like(template, dtype=tf.int32)
    ub = ub[..., None] + tf.zeros_like(template, dtype=tf.int32)
    template = tf.cast(tf.logical_or(tf.less(template, lb), tf.greater(template, ub)), tf.float32)

    logits0 = mix_logistic_to_logits(x, l, data_set='cifar')
    logits0 -= template * 1e30
    x0 = (tf.cast(tf.argmax(logits0, axis=4), dtype=tf.float32) - 127.5) / 127.5

    new_x = tf.stack([x0[..., 0], x[..., 1], x[..., 2]], axis=3)
    logits1 = mix_logistic_to_logits(new_x, l, data_set='cifar')
    logits1 -= template * 1e30
    x1 = (tf.cast(tf.argmax(logits1, axis=4), dtype=tf.float32) - 127.5) / 127.5

    new_x = tf.stack([x0[..., 0], x1[..., 1], x[..., 2]], axis=3)
    logits2 = mix_logistic_to_logits(new_x, l, data_set='cifar')
    logits2 -= template * 1e30
    x2 = (tf.cast(tf.argmax(logits2, axis=4), dtype=tf.float32) - 127.5) / 127.5

    return tf.stack([x0[..., 0], x1[..., 1], x2[..., 2]], axis=3)


def sample_from_softmax_mnist(x, l, data_set='f_mnist'):
    '''
    Assume we only have one channel
    '''
    logits = mix_logistic_to_logits(x, l, data_set=data_set)
    dist = tf.contrib.distributions.Categorical(logits=logits)
    x = (tf.cast(dist.sample(), dtype=tf.float32) - 127.5) / 127.5
    return x

def sample_from_softmax(x, l, data_set='cifar'):
    '''
    A faster version of sampling from softmax outputs. This version can sample all 3 channels at once.
    '''
    if data_set == 'f_mnist':
        return sample_from_softmax_mnist(x, l)

    if data_set != 'cifar':
        raise NotImplementedError("Dataset {} not supported!".format(data_set))

    logits0 = mix_logistic_to_logits(x, l)
    dist0 = tf.contrib.distributions.Categorical(logits=logits0)
    x0 = (tf.cast(dist0.sample(), dtype=tf.float32) - 127.5) / 127.5
    new_x = tf.stack([x0[..., 0], x[..., 1], x[..., 2]], axis=3)
    logits1 = mix_logistic_to_logits(new_x, l)
    dist1 = tf.contrib.distributions.Categorical(logits=logits1)
    x1 = (tf.cast(dist1.sample(), dtype=tf.float32) - 127.5) / 127.5

    new_x = tf.stack([x0[..., 0], x1[..., 1], x[..., 2]], axis=3)
    logits2 = mix_logistic_to_logits(new_x, l)
    dist2 = tf.contrib.distributions.Categorical(logits=logits2)

    x2 = (tf.cast(dist2.sample(), dtype=tf.float32) - 127.5) / 127.5
    return tf.stack([x0[..., 0], x1[..., 1], x2[..., 2]], axis=3)


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(
        logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, 4)
    log_scales = tf.maximum(tf.reduce_sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, 4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, 4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales) * (tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:, :, :, 0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return tf.concat([tf.reshape(x0, xs[:-1] + [1]), tf.reshape(x1, xs[:-1] + [1]), tf.reshape(x2, xs[:-1] + [1])], 3)


def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v


def get_vars_maybe_avg(var_names, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, **kwargs))
    return vars


def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1 > 0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1 * v + (1. - mom1) * g
            v_hat = v_t / (1. - tf.pow(mom1, t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2 * mg + (1. - mom2) * tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2, t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)


def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name


@add_arg_scope
def dense(x, num_units, nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
    ''' fully connected layer '''
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', [int(x.get_shape()[
                                              1]), num_units], tf.float32, tf.random_normal_initializer(0, 0.05),
                                trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.matmul(x, V_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=-m_init * scale_init, trainable=True)
            x_init = tf.reshape(
                scale_init, [1, num_units]) * (x_init - tf.reshape(m_init, [1, num_units]))
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema)
            # tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            x = tf.matmul(x, V)
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            x = tf.reshape(scaler, [1, num_units]) * \
                x + tf.reshape(b, [1, num_units])

            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
            return x


@add_arg_scope
def conv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', nonlinearity=None, init_scale=1., counters={},
           init=False, ema=None, **kwargs):
    ''' convolutional layer '''
    name = get_name('conv2d', counters)
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', filter_size + [int(x.get_shape()[-1]), num_filters],
                                tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            x_init = tf.nn.conv2d(x, V_norm, [1] + stride + [1], pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=-m_init * scale_init, trainable=True)
            x_init = tf.reshape(scale_init, [
                1, 1, 1, num_filters]) * (x_init - tf.reshape(m_init, [1, 1, 1, num_filters]))
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema)
            # tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, 1, num_filters]) * \
                tf.nn.l2_normalize(V, [0, 1, 2])

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)

            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
            return x


@add_arg_scope
def deconv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', nonlinearity=None, init_scale=1.,
             counters={}, init=False, ema=None, **kwargs):
    ''' transposed convolutional layer '''
    name = get_name('deconv2d', counters)
    xs = int_shape(x)
    if pad == 'SAME':
        target_shape = [xs[0], xs[1] * stride[0],
                        xs[2] * stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1] * stride[0] + filter_size[0] -
                        1, xs[2] * stride[1] + filter_size[1] - 1, num_filters]
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', filter_size + [num_filters, int(x.get_shape(
            )[-1])], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 3])
            x_init = tf.nn.conv2d_transpose(x, V_norm, target_shape, [
                1] + stride + [1], padding=pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=-m_init * scale_init, trainable=True)
            x_init = tf.reshape(scale_init, [
                1, 1, 1, num_filters]) * (x_init - tf.reshape(m_init, [1, 1, 1, num_filters]))
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema)
            # tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, num_filters, 1]) * \
                tf.nn.l2_normalize(V, [0, 1, 3])

            # calculate convolutional layer output
            x = tf.nn.conv2d_transpose(
                x, W, target_shape, [1] + stride + [1], padding=pad)
            x = tf.nn.bias_add(x, b)

            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
            return x


@add_arg_scope
def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, num_units, **kwargs)
    return tf.reshape(x, s[:-1] + [num_units])


''' meta-layer consisting of multiple base layers '''


@add_arg_scope
def gated_resnet(x, a=None, h=None, nonlinearity=concat_elu, conv=conv2d, init=False, counters={}, ema=None,
                 dropout_p=0., **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]

    c1 = conv(nonlinearity(x), num_filters)
    if a is not None:  # add short-cut connection if auxiliary input 'a' is given
        c1 += nin(nonlinearity(a), num_filters)
    c1 = nonlinearity(c1)
    if dropout_p > 0:
        c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
    c2 = conv(c1, num_filters * 2, init_scale=0.1)

    # add projection of h vector if included: conditional generation
    if h is not None:
        with tf.variable_scope(get_name('conditional_weights', counters)):
            hw = get_var_maybe_avg('hw', ema, shape=[int_shape(h)[-1], 2 * num_filters], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        if init:
            hw = hw.initialized_value()
        c2 += tf.reshape(tf.matmul(h, hw), [xs[0], 1, 1, 2 * num_filters])

    # Is this 3,2 or 2,3 ?
    a, b = tf.split(c2, 2, 3)
    c3 = a * tf.nn.sigmoid(b)
    return x + c3


''' utilities for shifting the image around, efficient alternative to masking convolutions '''


def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], 1, xs[2], xs[3]]), x[:, :xs[1] - 1, :, :]], 1)


def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], xs[1], 1, xs[3]]), x[:, :, :xs[2] - 1, :]], 2)


@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2, 3], stride=[1, 1], **kwargs):
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0],
                   [int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2)], [0, 0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)


@add_arg_scope
def down_shifted_deconv2d(x, num_filters, filter_size=[2, 3], stride=[1, 1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size,
                 pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
    return x[:, :(xs[1] - filter_size[0] + 1), int((filter_size[1] - 1) / 2):(xs[2] - int((filter_size[1] - 1) / 2)), :]


@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2, 2], stride=[1, 1], **kwargs):
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0],
                   [filter_size[1] - 1, 0], [0, 0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)


@add_arg_scope
def down_right_shifted_deconv2d(x, num_filters, filter_size=[2, 2], stride=[1, 1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size,
                 pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
    return x[:, :(xs[1] - filter_size[0] + 1):, :(xs[2] - filter_size[1] + 1), :]
