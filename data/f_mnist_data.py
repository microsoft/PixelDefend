# Copyright (c) Microsoft Corporation.  Licensed under the MIT license.
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def load_data(data_dir, subset):
    if subset == 'train':
        mnist_data = input_data.read_data_sets(data_dir)
        data, labels = mnist_data.train.images, mnist_data.train.labels
        data = np.reshape(data, [-1, 28, 28, 1]) * 255.
        return data, labels
    elif subset == 'test':
        if os.path.exists(os.path.join(data_dir, 'test_batch')):
            import pickle
            with open(os.path.join(data_dir, 'test_batch'), 'rb') as fin:
                d = pickle.load(fin, encoding='latin1')
            data = d['data'].reshape((-1, 28, 28, 1))
            labels = np.array(d['labels']).astype(np.uint8)
            return data, labels
        else:
            raise FileNotFoundError(os.path.join(data_dir, 'test_batch') + " Not found!")

    else:
        raise NotImplementedError("subset {} is not supported!".format(subset))


class DataLoader(object):
    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False):
        """
        - data_dir is location where to store files
        - subset is train|test
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        self.data, self.labels = load_data(data_dir, subset)
        self.p = 0
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset()  # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        # make sure the dimension is (batch_size, 28, 28, 1)
        x = self.data[self.p: self.p + n]
        y = self.labels[self.p: self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x, y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)
