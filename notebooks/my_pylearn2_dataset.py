# -*- coding: utf-8 -*-
# I've started from github/pylearn2/pylearn2/datasets/csv_dataset.py
# load filtered .hkl files as features

import csv
import numpy as np
import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess

import sys
sys.path.append('..')

class MyPyLearn2Dataset(DenseDesignMatrix):

    """A  class for accessing seizure files

    Parameters
    ----------
    path : str
      base directory, location of directories of files

    target : str
      target is added bot to the path and as a prefix to each file name.
      one of 'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2'

    task : str
      The type of task in which the dataset will be used -- either
      "classification" or "regression".  The task determines the shape of the
      target variable.  For classification, it is a vector; for regression, a
      matrix.

    one_hot : bool
      Whether the target variable (i.e. "label") should be encoded as a one-hot
      vector.

    expect_labels : bool
      Whether to load positive (*preictal*) and negative (*interictal*) files or to load
      unlabeled test files (*test*)
    """

    def __init__(self,
                 path='../filtered-seizure-data', # base directory, location of directories of filtered hkl files
                 target='Dog_1', # target is added bot to the path and as a prefix to each file name
                 one_hot=False,
                 scale_option='usf',
                 nwindows=60,
                 skip=5,
                 window_size=None,
                 expect_labels = True):
        """
        .. todo::

            WRITEME
        """
        self.path = path
        self.target = target
        self.one_hot = one_hot
        self.scale_option = scale_option
        self.nwindows = nwindows
        self.expect_labels = expect_labels
        self.skip = skip

        self.view_converter = None
        self.Nsamples = 239766 # 10 min at 399.61 Hz
        if window_size is None:
            self.window_size = self.Nsamples // self.nwindows
        else:
            self.window_size = window_size

        # and go

        self.path = preprocess(self.path)
        X, y = self._load_data()

        super(MyPyLearn2Dataset, self).__init__(X=X, y=y)

    def _load_data(self):
        """
        .. todo::

            WRITEME
        """
        import common.time as time
        start = time.get_seconds()

        from seizure.tasks import load_mat_data, count_mat_data
        from seizure.transforms import UnitScaleFeat, UnitScale
        import seizure.tasks
        seizure.tasks.task_predict = True

        # data_type is one of ('preictal', 'interictal', 'test')
        # target is one of 'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2'
        data_dir = self.path
        data_types = ['preictal', 'interictal'] if self.expect_labels else ['test']

        N = 0
        for data_type in data_types:
            for i in count_mat_data(data_dir, self.target, data_type):
                N += 1
        print 'Number of segments', N

        Nf = None
        row = 0
        count = 0
        for data_type in data_types:
            mat_data = load_mat_data(data_dir, self.target, data_type)
            for segment in mat_data:
                for key in segment.keys():
                    if not key.startswith('_'):
                        break
                data = segment[key]['data'][0,0]

                assert data.shape[-1] == self.Nsamples

                istartend = np.linspace(0.,self.Nsamples - self.window_size, self.nwindows)

                for i in range(self.nwindows):
                    count += 1
                    if (count-1) % self.skip != 0:
                        continue
                    window = data[:,int(istartend[i]):int(istartend[i] + self.window_size)]
                    if Nf is None:
                        Nchannels = window.shape[0]
                        print 'Number of channels', Nchannels
                        N *= Nchannels * self.nwindows / self.skip
                        print 'Number of examples', N
                        Nf = window.shape[1]
                        print 'Number of features', Nf
                        X = np.empty((N, Nf))
                        y = np.empty(N)

                    if self.scale_option == 'usf':
                        window = UnitScaleFeat().apply(window)
                    elif self.scale_option == 'us':
                        window = UnitScale().apply(window)
                    X[row:row+Nchannels, :] = window
                    y[row:row+Nchannels] = (0 if data_type == 'interictal' else 1)
                    row += Nchannels

        if self.expect_labels:
            if self.one_hot:
                # get unique labels and map them to one-hot positions
                labels = np.unique(y)
                labels = dict((x, i) for (i, x) in enumerate(labels))

                one_hot = np.zeros((y.shape[0], len(labels)), dtype='float32')
                for i in xrange(y.shape[0]):
                    label = y[i]
                    label_position = labels[label]
                    one_hot[i, label_position] = 1.
                y = one_hot

        print X.shape, y.shape, y.mean(axis=-1)
        print 'time %ds' % (time.get_seconds() - start)
        return X, y

if __name__ == "__main__":
    x = MyPyLearn2Dataset(one_hot=True)