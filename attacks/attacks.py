# Copyright (c) Microsoft Corporation.  Licensed under the MIT license.
import tensorflow as tf
import numpy as np
import types
from cleverhans.attacks_tf import fgsm, fgm
from .CW import CarliniLi
from .deepfool import deepfool
from utils import get_weights_path, median_filtering_2x2, feature_squeeze

FLAGS = tf.app.flags.FLAGS


def stretch(img):
    if np.max(img) > 255. or np.min(img) < 0.:
        img = img.astype(float)
        img -= np.min(img)
        img /= np.max(img) + 1e-12
        return img * 255.
    else:
        return img


class Attacker(object):
    @staticmethod
    def randomize(image, eps=8):
        l_eps = eps
        u_eps = 2 * eps - l_eps + 1
        l_eps = -l_eps
        noise = np.random.randint(l_eps, u_eps, size=image.shape)
        return np.minimum(np.maximum(image + noise, 0.0), 255.).astype(np.uint8)

    @staticmethod
    def steal_pixel_single_channel(image, number=1):
        pos_x = np.random.randint(0, 32, size=(image.shape[0], number))
        pos_y = np.random.randint(0, 32, size=(image.shape[0], number))
        pos_c = np.random.randint(0, 3, size=(image.shape[0], number))
        pos_b = np.arange(0, image.shape[0])
        res = np.copy(image)
        for i in range(number):
            res[pos_b, pos_x[:, i], pos_y[:, i], pos_c[:, i]] = np.random.randint(0, 256, size=(image.shape[0],))
        return res

    @staticmethod
    def steal_pixel(image, number=1):
        pos_x = np.random.randint(0, 32, size=(image.shape[0], number))
        pos_y = np.random.randint(0, 32, size=(image.shape[0], number))
        pos_b = np.arange(0, image.shape[0])
        res = np.copy(image)
        for i in range(number):
            res[pos_b, pos_x[:, i], pos_y[:, i], :] = np.random.randint(0, 256, size=(image.shape[0], 3))
        return res

    def __init__(self, sess, resnet):
        '''
        images: though it is not a placeholder, most of the time you should feed your image into this variable
        images_scaled: a placeholder that contains images scaled to -0.5 to 0.5
        '''
        self.sess = sess
        self.images = tf.placeholder(tf.float32, (None, FLAGS.image_size, FLAGS.image_size, FLAGS.channels), name="images")

        # combined attack for feature squeezing!
        if FLAGS.feature_squeeze:
            smoothed_images = median_filtering_2x2(self.images, dataset=FLAGS.dataset)

        self.labels = tf.placeholder(tf.int64, (None,), name="labels")
        self.resnet = resnet
        self.logits = resnet(self.images) if not FLAGS.feature_squeeze else resnet(smoothed_images)
        self.softmax = tf.nn.softmax(self.logits)
        self.eps = tf.placeholder(tf.float32, (), name="fgsm_eps")

        self.cw_model = types.SimpleNamespace()
        self.cw_model.image_size = FLAGS.image_size
        self.cw_model.num_channels = FLAGS.channels
        self.cw_model.predict = self.resnet
        self.cw_model.num_labels = 10

        labels_onehot = tf.one_hot(self.labels, depth=self.cw_model.num_labels)
        self.adv_image = fgm(self.images, self.softmax, y=labels_onehot, eps=self.eps, clip_min=0.0, clip_max=255.0)

        saver = tf.train.Saver()
        save_path, save_path_ckpt = get_weights_path()
        try:
            ckpt_state = tf.train.get_checkpoint_state(save_path)
        except tf.errors.OutOfRangeError as e:
            raise AssertionError('Cannot restore checkpoint: %s', e)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            raise FileNotFoundError('No model to eval yet at %s', save_path)

        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        self.index = tf.placeholder(tf.int32, (), name="index")
        self.grad = tf.gradients(self.logits[:, self.index], [self.images], name="grad")[0]

        # Images in CW attack are rescaled to [-0.5, 0.5]
        self.cw_attacker = CarliniLi(self.sess, self.cw_model, targeted=False)

    def FGSM(self, input, labels):
        new_labels = np.argmax(self.logits.eval(session=self.sess, feed_dict={
            self.images: input
        }), axis=1)
        return self.adv_image.eval(session=self.sess, feed_dict={
            self.images: input,
            self.eps: FLAGS.eps,
            self.labels: new_labels
        })

    def FGSM_rand(self, input, labels):
        perturbed_input = np.copy(input)
        perturbed_input = self.randomize(perturbed_input, eps=FLAGS.eps // 2)
        new_labels = np.argmax(self.logits.eval(session=self.sess, feed_dict={
            self.images: perturbed_input
        }), axis=1)
        return self.adv_image.eval(session=self.sess, feed_dict={
            self.images: perturbed_input,
            self.eps: FLAGS.eps // 2,
            self.labels: new_labels
        })

    def BIM(self, input, labels):
        '''
        new_labels = np.argmax(self.logits.eval(session=self.sess, feed_dict={
            self.images: input
        }), axis=1)
        '''
        iter = int(min(FLAGS.eps + 4, 1.25 * FLAGS.eps))
        image = np.copy(input)
        for i in range(iter):
            image = self.adv_image.eval(session=self.sess, feed_dict={
                self.images: image,
                self.eps: 1,
                self.labels: labels
            })
            image = np.clip(image, input - FLAGS.eps, input + FLAGS.eps)

        return image


    def CW(self, input, labels):
        input = np.copy(input)
        labels = np.copy(labels)
        scaled_input = (input / 255.) - 0.5
        one_hots = np.zeros((labels.shape[0], self.cw_model.num_labels))
        one_hots[np.arange(0, labels.shape[0]), labels] = 1.
        scaled_adv_image = self.cw_attacker.attack(scaled_input, one_hots)
        adv_image = (scaled_adv_image + 0.5) * 255.0
        res_image = np.clip(adv_image, input - FLAGS.eps, input + FLAGS.eps)
        res_image = np.clip(res_image, 0.0, 255.0)
        return res_image


    def deepfool(self, input):
        input = np.copy(input)

        def f(x):
            return self.logits.eval(session=self.sess, feed_dict={
                self.images: x
            })

        def grads(x, inds):
            return np.array([
                self.grad.eval(session=self.sess, feed_dict={
                    self.images: x,
                    self.index: i
                }) for i in inds
            ])

        res = []
        for i, img in enumerate(input):
            img = img[None, ...]
            print("Fooling the {}th image".format(i + 1))
            r_tot, loop_i, k_i, pert_img = deepfool(img, f, grads, num_classes=self.cw_model.num_labels)
            pert_img = stretch(pert_img)
            pert_img = np.clip(pert_img, img - FLAGS.eps, img + FLAGS.eps)
            pert_img = np.clip(pert_img, 0.0, 255.0)
            res.extend(pert_img)

        return np.array(res)
