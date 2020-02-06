# Copyright (c) Microsoft Corporation.  Licensed under the MIT license.
import time
import json
import shutil
import models.resnet_model as resnet_model
from data import cifar10_data, f_mnist_data
from utils import *
from attacks.attacks import Attacker
from cleverhans.attacks_tf import fgsm, fgm
from models.vgg16 import vgg_16
from scipy.stats import truncnorm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'cifar', 'cifar or f_mnist.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval or attack.')
tf.app.flags.DEFINE_boolean('adversarial', False,
                            'For training models, turning on this flag means adversarial training.'
                            'For evaluating and attacking models, turning on this flag means loading adversarially trained model')
tf.app.flags.DEFINE_string('model', 'resnet', 'resnet | vgg')
tf.app.flags.DEFINE_string('input_data_path', None,
                           'Path for input data. Will override the default one')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('seed', 1234, 'random seeds')
tf.app.flags.DEFINE_integer('maxiter', 80000, 'maximum number of training iterations')
tf.app.flags.DEFINE_integer('eps', 8, 'eps for attacking')
tf.app.flags.DEFINE_boolean('random', False, 'Turning it on means we want to evaluate randomly perturbed images')
tf.app.flags.DEFINE_boolean('top5', False, 'evaluate top-5 accuracy')
tf.app.flags.DEFINE_boolean('label_smooth', False, 'do label smoothing during training')
tf.app.flags.DEFINE_boolean('feature_squeeze', False, 'do feature squeezing during evaluation or attack')
tf.app.flags.DEFINE_boolean('adversarial_BIM', False, 'do adversarial training with BIM adversarial examples')
tf.app.flags.DEFINE_integer('adv_std', 8, 'standard deviation of eps used for adversarial training')
tf.app.flags.DEFINE_string('attack_type', None,
                           'Specify a list of attacks rather than the whole attack.'
                           'If None, do all attacks.')
tf.app.flags.DEFINE_string('noise_type', None, 'NOISE TYPE is some folder name in results/data')

def resnet_template(images, training, hps):
    # Do per image standardization
    images_standardized = per_image_standardization(images)
    model = resnet_model.ResNet(hps, images_standardized, training)
    model.build_graph()
    return model.logits


def vgg_template(images, training, hps):
    images_standardized = per_image_standardization(images)
    logits, _ = vgg_16(images_standardized, num_classes=hps.num_classes, is_training=training, dataset=hps.dataset)
    return logits


def train(hps, data):
    """Training loop."""
    images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.channels), name="images")
    labels = tf.placeholder(tf.int64, shape=(None), name="labels")
    labels_onehot = tf.one_hot(labels, depth=hps.num_classes, dtype=tf.float32, name="labels_onehot")
    if FLAGS.label_smooth:
        labels_onehot = label_smooth(labels_onehot)

    lrn_rate = tf.placeholder(tf.float32, shape=(), name="lrn_rate")
    tf.logging.info(json.dumps(vars(FLAGS)))
    tf.logging.info(json.dumps(hps._asdict()))

    flipped_images = random_flip_left_right(images)

    net = tf.make_template('net', resnet_template, hps=hps) if FLAGS.model == 'resnet' else \
        tf.make_template('net', vgg_template, hps=hps)

    truth = labels
    if FLAGS.adversarial or FLAGS.adversarial_BIM:
        logits = net(flipped_images, training=False)
    else:
        logits = net(flipped_images, training=True)
    probs = tf.nn.softmax(logits)

    predictions = tf.argmax(logits, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_onehot))

    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net')
    weight_norm = tf.add_n([tf.nn.l2_loss(v) for v in weights])
    cost = cost + 0.0005 * weight_norm

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = tf.train.MomentumOptimizer(learning_rate=lrn_rate, momentum=0.9).minimize(cost)

    if FLAGS.adversarial or FLAGS.adversarial_BIM:
        eps = tf.abs(tf.truncated_normal(shape=(tf.shape(images)[0],), mean=0, stddev=FLAGS.adv_std))
        eps = eps[:, None, None, None]
        adv_x = fgsm(flipped_images, probs, eps=eps, clip_min=0.0, clip_max=255.0)
        adv_x_leak = fgm(flipped_images, probs, y=labels_onehot, eps=np.asarray([1])[:, None, None, None],
                         clip_min=0.0, clip_max=255.0)

        adv_logits = net(adv_x, training=False)
        adv_pred = tf.argmax(adv_logits, axis=1)
        adv_precision = tf.reduce_mean(tf.to_float(tf.equal(adv_pred, truth)))

        adv_logits_leak = net(adv_x_leak, training=False)
        adv_pred_leak = tf.argmax(adv_logits_leak, axis=1)
        adv_precision_leak = tf.reduce_mean(tf.to_float(tf.equal(adv_pred_leak, truth)))

        num_normal = hps.batch_size // 2
        combined_images = tf.concat([flipped_images[:num_normal], images[num_normal:]], axis=0)
        com_logits = net(combined_images, training=True)

        normal_cost = 2.0 / 1.3 * tf.nn.softmax_cross_entropy_with_logits(logits=com_logits[:num_normal],
                                                                          labels=labels_onehot[:num_normal])
        adv_cost = 0.6 / 1.3 * tf.nn.softmax_cross_entropy_with_logits(logits=com_logits[num_normal:],
                                                                       labels=labels_onehot[num_normal:])

        combined_cost = tf.reduce_mean(tf.concat([normal_cost, adv_cost], axis=0)) + 0.0005 * weight_norm

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op_adv = tf.train.MomentumOptimizer(learning_rate=lrn_rate, momentum=0.9).minimize(combined_cost)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(max_to_keep=3)
        save_path, save_path_ckpt = get_weights_path()
        state = tf.train.get_checkpoint_state(save_path)
        if state and state.model_checkpoint_path:
            ans = verify("Warning: model already trained. Delete files and re-train? (y/n)")
            if ans:
                shutil.rmtree(save_path)
                os.makedirs(save_path)
            else:
                saver_state = tf.train.get_checkpoint_state(save_path)
                saver.restore(sess, saver_state.model_checkpoint_path)
                # raise FileExistsError("Model weight already exists")
        else:
            os.makedirs(save_path, exist_ok=True)

        hps_path = os.path.join(save_path, 'hps.txt')
        with open(hps_path, 'w') as fout:
            fout.write(json.dumps(vars(FLAGS)))
            fout.write(json.dumps(hps._asdict()))

        for iter in range(FLAGS.maxiter):
            try:
                x, y = data.next(hps.batch_size)
            except StopIteration:
                tf.logging.info("New epoch!")

            if iter < 40000:
                lr = 0.1
            elif iter < 60000:
                lr = 0.01
            elif iter < 80000:
                lr = 0.001
            else:
                lr = 0.0001

            if not FLAGS.adversarial and not FLAGS.adversarial_BIM:
                _, acc = sess.run([train_op, precision], feed_dict={
                    images: x,
                    labels: y,
                    lrn_rate: lr
                })
                tf.logging.info("Iter: {}, Precision: {:.6f}".format(iter + 1, acc))
            elif FLAGS.adversarial:
                adv_images, acc, acc_adv = sess.run([adv_x, precision, adv_precision], feed_dict={
                    images: x,
                    labels: y,
                })
                combined_batch = np.concatenate([x[:num_normal], adv_images[num_normal:]], axis=0)
                _, com_loss = sess.run([train_op_adv, combined_cost], feed_dict={
                    images: combined_batch,
                    labels: y,
                    lrn_rate: lr
                })
                tf.logging.info("Iter: {}, Precision: {:.6f}, Adv precision: {:.6f}, Combined loss: {:.6f}"
                                .format(iter + 1, acc, acc_adv, com_loss))

            elif FLAGS.adversarial_BIM:
                BIM_eps = np.abs(truncnorm.rvs(a=-2., b=2.) * FLAGS.adv_std)
                attack_iter = int(min(BIM_eps + 4, 1.25 * BIM_eps))
                adv_images = np.copy(x)
                for i in range(attack_iter):
                    adv_images, acc, acc_adv = sess.run([adv_x_leak, precision, adv_precision_leak], feed_dict={
                        images: adv_images,
                        labels: y,
                    })

                combined_batch = np.concatenate([x[:num_normal], adv_images[num_normal:]], axis=0)
                _, com_loss = sess.run([train_op_adv, combined_cost], feed_dict={
                    images: combined_batch,
                    labels: y,
                    lrn_rate: lr
                })
                tf.logging.info("Iter: {}, Precision: {:.6f}, Adv precision: {:.6f}, Combined loss: {:.6f}"
                                .format(iter + 1, acc, acc_adv, com_loss))

            if (iter + 1) % 5000 == 0:
                saver.save(sess, save_path_ckpt, global_step=iter + 1)
                tf.logging.info("Model saved! Path: " + save_path)


def evaluate(hps, data):
    """Eval loop."""
    images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.channels))
    # Do feature squeezing
    if FLAGS.feature_squeeze:
        smoothed_images = feature_squeeze(images, dataset=FLAGS.dataset)

    labels = tf.placeholder(tf.int64, shape=(None))
    labels_onehot = tf.one_hot(labels, depth=hps.num_classes, dtype=tf.int32)

    net = tf.make_template('net', resnet_template, hps=hps) if FLAGS.model == 'resnet' else \
        tf.make_template('net', vgg_template, hps=hps)

    logits = net(images, training=False) if not FLAGS.feature_squeeze else net(smoothed_images, training=False)
    pred = tf.argmax(logits, axis=1)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_onehot)
    top_5 = tf.nn.in_top_k(predictions=logits, targets=labels, k=5)

    saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    best_precision = 0.0
    save_path, save_path_ckpt = get_weights_path()
    try:
        ckpt_state = tf.train.get_checkpoint_state(save_path)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)
        return
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', save_path)
        return
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0
    total_loss = 0
    all_preds = []
    for x, y in data:
        x = x.astype(np.float32)
        if FLAGS.random:
            x = Attacker.randomize(x, eps=FLAGS.eps)
        if not FLAGS.top5:
            (loss, predictions) = sess.run(
                [cost, pred], feed_dict={
                    images: x,
                    labels: y
                })
            all_preds.extend(predictions)
        else:
            (loss, in_top5) = sess.run(
                [cost, top_5], feed_dict={
                    images: x,
                    labels: y
                }
            )
        total_loss += np.sum(loss)
        correct_prediction += np.sum(y == predictions) if not FLAGS.top5 else np.sum(in_top5)
        total_prediction += loss.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    loss = 1.0 * total_loss / total_prediction
    best_precision = max(precision, best_precision)

    if not FLAGS.top5:
        tf.logging.info('loss: %.6f, precision: %.6f, best precision: %.6f' %
                        (loss, precision, best_precision))
    else:
        tf.logging.info('loss: %.6f, top 5 accuracy: %.6f, best top 5 accuracy: %.6f' %
                        (loss, precision, best_precision))

    all_preds = np.asarray(all_preds)
    return np.stack([data.labels, all_preds], axis=1)


def attack(hps, data):
    tf.logging.info(json.dumps(vars(FLAGS)))
    tf.logging.info(json.dumps(hps._asdict()))

    net = tf.make_template('net', resnet_template, hps=hps, training=False) if FLAGS.model == "resnet" else \
        tf.make_template('net', vgg_template, hps=hps, training=False)

    # The type of all attacks
    if FLAGS.attack_type is None:
        types = ['random', 'FGSM', 'BIM', 'deepfool', 'CW', 'FGSM_rand']
    else:
        types = FLAGS.attack_type.split(' ')
    # types = ['random', 'FGSM', 'BIM', 'deepfool']
    # types = ['random', 'FGSM', 'FGSM_rand', 'BIM']
    for type in types:
        prefix, full_path = get_attack_output_name(type)
        if os.path.exists(prefix):
            ans = verify("Perturbed data {} already exists. Rewrite them? (y/n)".format(os.path.basename(prefix)))
            if ans:
                shutil.rmtree(prefix)
                os.makedirs(prefix)
            else:
                raise FileExistsError("{} data already exists.".format(os.path.basename(prefix)))
        else:
            os.makedirs(prefix)
        with open(os.path.join(prefix, "hps.txt"), 'w') as fout:
            json.dump(vars(FLAGS), fout)
            json.dump(hps._asdict(), fout)

    gen_data = {name: [] for name in types}
    check_input = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.channels))
    check_logits = net(check_input)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    attacker = Attacker(sess, net)

    def accuracy(x, y):
        return np.sum(np.argmax(x, axis=1) == y) / x.shape[0]

    for i, datum in enumerate(data):
        x, y = datum
        x = x.astype(np.float32)
        tf.logging.info("Generating the {}th batch of perturbed data".format(i + 1))
        print("Accuracy of clear image: {:.6f}".format(
            accuracy(check_logits.eval(session=sess, feed_dict={check_input: x}), y)))
        # generate random data
        if 'random' in types:
            res = Attacker.randomize(x, eps=FLAGS.eps)
            gen_data['random'].extend(res)
            print("Accuracy of random: {:.6f}".format(
                accuracy(check_logits.eval(session=sess, feed_dict={check_input: res}), y)))

        # generate fgsm random data
        if 'FGSM_rand' in types:
            res = attacker.FGSM_rand(x, y)
            gen_data['FGSM_rand'].extend(res)
            print("Accuracy of FGSM_rand: {:.6f}".format(
                accuracy(check_logits.eval(session=sess, feed_dict={check_input: res}), y)))

        # generate fgsm data
        if 'FGSM' in types:
            res = attacker.FGSM(x, y)
            gen_data['FGSM'].extend(res)
            print("Accuracy of FGSM: {:.6f}".format(
                accuracy(check_logits.eval(session=sess, feed_dict={check_input: res}), y)))

        # generate BIM data
        if 'BIM' in types:
            res = attacker.BIM(x, y)
            gen_data['BIM'].extend(res)
            print("Accuracy of BIM: {:.6f}".format(
                accuracy(check_logits.eval(session=sess, feed_dict={check_input: res}), y)))

        # generate CW data
        if 'CW' in types:
            tf.logging.info("CW attack for {}th batch".format(i + 1))
            res = attacker.CW(x, y)
            gen_data['CW'].extend(res)
            print("Accuracy of CW: {:.6f}".format(
                accuracy(check_logits.eval(session=sess, feed_dict={check_input: res}), y)))

        # generate DeepFool data
        if 'deepfool' in types:
            tf.logging.info("DeepFool attack for {}th batch".format(i + 1))
            res = attacker.deepfool(x)
            gen_data['deepfool'].extend(res)
            print("Accuracy of deepfool: {:.6f}".format(
                accuracy(check_logits.eval(session=sess, feed_dict={check_input: res}), y)))

    for type in types:
        gen_data[type] = np.array(gen_data[type])
        save_output_data(gen_data[type], data.labels, type=type)


def main(_):
    rng = np.random.RandomState(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    if FLAGS.dataset == 'cifar':
        DataLoader = cifar10_data.DataLoader
        num_classes = 10
        FLAGS.num_classes = 10
        FLAGS.image_size = 32
        FLAGS.channels = 3

    elif FLAGS.dataset == 'f_mnist':
        DataLoader = f_mnist_data.DataLoader
        num_classes = 10
        FLAGS.num_classes = 10
        FLAGS.image_size = 28
        FLAGS.channels = 1

    else:
        raise NotImplementedError()

    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    input_data_path = get_input_data_path() if FLAGS.input_data_path is None else \
        FLAGS.input_data_path
    if FLAGS.mode == 'train':
        batch_size = 128
        train_data = DataLoader(input_data_path, 'train', batch_size,
                                rng=rng, shuffle=True, return_labels=True)

    elif FLAGS.mode == 'eval':
        batch_size = 100
        eval_data = DataLoader(input_data_path, 'test', batch_size, shuffle=False, return_labels=True)
    elif FLAGS.mode == 'attack':
        batch_size = 100
        attack_data = DataLoader(input_data_path, 'test', batch_size, shuffle=False, return_labels=True)
    else:
        raise NotImplementedError()

    # num_residual_units = 10 if FLAGS.adversarial or FLAGS.adversarial_BIM else 5
    num_residual_units = 5
    hps = resnet_model.HParams(batch_size=batch_size,
                               num_classes=num_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.1,
                               num_residual_units=num_residual_units,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom',
                               dataset=FLAGS.dataset)

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps, train_data)
        elif FLAGS.mode == 'eval':
            savable = evaluate(hps, eval_data)
            np.save(os.path.join(input_data_path, FLAGS.model + '_preds.npy'), savable)
        elif FLAGS.mode == 'attack':
            attack(hps, attack_data)
        else:
            raise NotImplementedError("unrecognized mode " + FLAGS.mode)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
