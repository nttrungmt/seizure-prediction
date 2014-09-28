#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
import os.path
import numpy as np
import scipy.io
import common.time as time
from sklearn import cross_validation, preprocessing
from sklearn.metrics import roc_curve, auc
from common.pipeline import Pipeline, UnionPipeline

task_predict = False # flag between the two kaggle competitons seizure detection (False) and seizure prediction (True)

TaskCore = namedtuple('TaskCore', ['cached_data_loader', 'data_dir', 'target', 'pipeline', 'classifier_name',
                                   'classifier', 'normalize', 'gen_ictal', 'cv_ratio'])

class Task(object):
    """
    A Task computes some work and outputs a dictionary which will be cached on disk.
    If the work has been computed before and is present in the cache, the data will
    simply be loaded from disk and will not be pre-computed.
    """
    def __init__(self, task_core):
        self.task_core = task_core

    def filename(self):
        raise NotImplementedError("Implement this")

    def run(self):
        if isinstance(self.task_core.pipeline, UnionPipeline):
            # dont cache the union (assume the components are already cached)
            tc = self.task_core
            transforms = tc.pipeline.transforms
            gen_ictal = tc.pipeline.gen_ictal
            features_union = None
            for transform in transforms:
                self.task_core = TaskCore(cached_data_loader=tc.cached_data_loader,
                                          data_dir=tc.data_dir, target=tc.target,
                                         pipeline=Pipeline(gen_ictal, [transform]),
                                         classifier_name=tc.classifier_name, classifier=tc.classifier,
                                         normalize=tc.normalize, gen_ictal=tc.gen_ictal, cv_ratio=tc.cv_ratio)
                features = self.task_core.cached_data_loader.load(self.filename(), self.load_data)
                if features_union is None:
                    features_union = features
                else:
                    assert np.all(features.y == features_union.y)
                    features_union.X = np.concatenate((features_union.X, features.X), axis=-1)
            self.task_core = tc
            return features_union
        else:
            return self.task_core.cached_data_loader.load(self.filename(), self.load_data)

class Task(object):
    """
    A Task computes some work and outputs a dictionary which will be cached on disk.
    If the work has been computed before and is present in the cache, the data will
    simply be loaded from disk and will not be pre-computed.
    """
    def __init__(self, task_core):
        self.task_core = task_core

    def filename(self):
        raise NotImplementedError("Implement this")

    def run(self):
        return self.task_core.cached_data_loader.load(self.filename(), self.load_data)

class LoadTask(Task):
    def filename(self):
        return 'data_%s_%s_%s' % (self.data_type, self.task_core.target, self.task_core.pipeline.get_name())

    def run(self):
        if isinstance(self.task_core.pipeline, UnionPipeline):
            # dont cache the union (assume the components are already cached)
            tc = self.task_core
            transforms = tc.pipeline.transforms
            gen_ictal = tc.pipeline.gen_ictal
            features_union = None
            for transform in transforms:
                self.task_core = TaskCore(cached_data_loader=tc.cached_data_loader,
                                          data_dir=tc.data_dir, target=tc.target,
                                         pipeline=Pipeline(gen_ictal, [transform]),
                                         classifier_name=tc.classifier_name, classifier=tc.classifier,
                                         normalize=tc.normalize, gen_ictal=tc.gen_ictal, cv_ratio=tc.cv_ratio)
                features = self.task_core.cached_data_loader.load(self.filename(), self.load_data)
                if features_union is None:
                    features_union = features
                else:
                    assert np.all(features.y == features_union.y)
                    features_union.X = np.concatenate((features_union.X, features.X), axis=-1)
            self.task_core = tc
            return features_union
        else:
            return self.task_core.cached_data_loader.load(self.filename(), self.load_data)

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, self.data_type, self.task_core.pipeline,
                           self.task_core.gen_ictal)

class LoadIctalDataTask(LoadTask):
    """
    Load the ictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y, 'latencies': latencies}
    """
    data_type = 'ictal'



class LoadInterictalDataTask(LoadTask):
    """
    Load the interictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y}
    """
    data_type = 'interictal'

class LoadPreictalDataTask(LoadTask):
    """
    Load the pre-ictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y}
    """
    data_type = 'preictal'

class LoadTestDataTask(LoadTask):
    """
    Load the test mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X}
    """
    data_type = 'test'

class TrainingDataTask(Task):
    """
    Creating a training set and cross-validation set from the transformed ictal and interictal data.
    """
    def filename(self):
        return None  # not cached, should be fast enough to not need caching

    def load_data(self):
        if task_predict:
            ictal_data = LoadPreictalDataTask(self.task_core).run()
        else:
            ictal_data = LoadIctalDataTask(self.task_core).run()
        interictal_data = LoadInterictalDataTask(self.task_core).run()
        return prepare_training_data(ictal_data, interictal_data, self.task_core.cv_ratio)


class CrossValidationScoreTask(Task):
    """
    Run a classifier over a training set, and give a cross-validation score.
    """
    def filename(self):
        return 'score_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        classifier_data = train_classifier(self.task_core.classifier, data, normalize=self.task_core.normalize)
        del classifier_data['classifier'] # save disk space
        return classifier_data


class TrainClassifierTask(Task):
    """
    Run a classifier over the complete data set (training data + cross-validation data combined)
    and save the trained models.
    """
    def filename(self):
        return 'classifier_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        return train_classifier(self.task_core.classifier, data, use_all_data=True, normalize=self.task_core.normalize)


class MakePredictionsTask(Task):
    """
    Make predictions on the test data.
    """
    def filename(self):
        return 'predictions_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        y_classes = data.y_classes
        del data

        classifier_data = TrainClassifierTask(self.task_core).run()
        test_data = LoadTestDataTask(self.task_core).run()
        X_test = flatten(test_data.X)

        return make_predictions(self.task_core.target, X_test, y_classes, classifier_data)

# a list of pairs indicating the slices of the data containing full seizures
# e.g. [(0, 5), (6, 10)] indicates two ranges of seizures
def seizure_ranges_for_latencies(latencies):
    indices = np.where(latencies == 0)[0]

    ranges = []
    for i in range(1, len(indices)):
        ranges.append((indices[i-1], indices[i]))
    ranges.append((indices[-1], len(latencies)))

    return ranges


#generator to iterate over competition mat data
def load_mat_data(data_dir, target, component):
    dir = os.path.join(data_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        if task_predict:
            filename = '%s/%s_%s_segment_%04d.mat' % (dir, target, component, i)
        else:
            filename = '%s/%s_%s_segment_%d.mat' % (dir, target, component, i)

        if os.path.exists(filename):
            data = scipy.io.loadmat(filename)
            yield(data)
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True


# process all of one type of the competition mat data
# data_type is one of ('ictal', 'interictal', 'test')
def parse_input_data(data_dir, target, data_type, pipeline, gen_ictal=False):
    ictal = data_type == 'ictal'
    preictal = data_type == 'preictal'
    interictal = data_type == 'interictal'

    # create an iterator
    mat_data = load_mat_data(data_dir, target, data_type)

    # for each data point in ictal, interictal and test,
    # generate (X, <y>, <latency>) per channel
    def process_raw_data(mat_data, with_latency):
        start = time.get_seconds()
        print 'Loading data',
        X = []
        y = []
        latencies = []

        prev_data = None
        prev_sequence = None
        prev_latency = None
        for segment in mat_data:
            if task_predict:
                for key in segment.keys():
                    if not key.startswith('_'):
                        break
                data = segment[key]['data'][0,0]
                if key.startswith('preictal') or key.startswith('interictal'):
                    sequence = segment[key]['sequence'][0,0][0,0]
                else:
                    sequence = None
            else:
                data = segment['data']
                sequence = None
            transformed_data = pipeline.apply(data)

            if with_latency:
                # this is ictal
                latency = segment['latency'][0]
                if latency <= 15:
                    y_value = 0 # ictal <= 15
                else:
                    y_value = 1 # ictal > 15

                # generate extra ictal training data by taking 2nd half of previous
                # 1-second segment and first half of current segment
                # 0.5-1.5, 1.5-2.5, ..., 13.5-14.5, ..., 15.5-16.5
                # cannot take half of 15 and half of 16 because it cannot be strictly labelled as early or late
                if gen_ictal and prev_data is not None and prev_latency + 1 == latency and prev_latency != 15:
                    # gen new data :)
                    axis = prev_data.ndim - 1
                    def split(d):
                        return np.split(d, 2, axis=axis)
                    new_data = np.concatenate((split(prev_data)[1], split(data)[0]), axis=axis)
                    X.append(pipeline.apply(new_data))
                    y.append(y_value)
                    latencies.append(latency - 0.5)

                y.append(y_value)
                latencies.append(latency)

                prev_latency = latency
            elif y is not None:
                # this is interictal
                label = 0 if key.startswith('preictal') else 2
                if key.startswith('preictal') or key.startswith('interictal'):
                    # generate extra training data by taking overlaps with previous
                    # segment
                    # negative gen_ictal indicates we want to correct for DC jump between segments
                    # non integer value indicates we want to generate overlaps also for negative examples
                    ng = abs(int(gen_ictal)) # number of overlapping windows
                    if (gen_ictal and
                            (key.startswith('preictal') or gen_ictal != int(gen_ictal)) and
                                prev_data is not None and prev_sequence+1 == sequence):
                        if isinstance(gen_ictal,bool) or gen_ictal > 0:
                            new_data = np.concatenate((prev_data, data), axis=-1)
                        else:
                            # see 140922-signal-crosscorelation
                            # it looks like each segment was scaled to have DC=0
                            # however different segments will be scaled differently
                            # as result you can't concatenate sequential segments
                            # without undoing the relative offset

                            # import scipy.signal
                            # # we want to filter the samples so as to not be sensitive to change in the signal itself
                            # # over the distance of one sample (1/Fs). Taking 100 samples sounds safe enough.
                            # normal_cutoff = 2./100. # 1/100*Fs in Hz
                            # order = 6
                            # b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
                            # # use filtfilt to get zero phase http://wiki.scipy.org/Cookbook/FiltFilt
                            # W1 = 5000
                            # x1 = scipy.signal.filtfilt(b, a, prev_data[:,-W1:])
                            # # we want the first sample of data after fitering so we will run it backward through
                            # # the filter
                            # x2 = scipy.signal.filtfilt(b, a, data[:,W1-1::-1])
                            # # the first sample of data should be about the same as the last sample of prev_data
                            # data_offset = x2[:,-1] - x1[:,-1]
                            if data.shape[1] > 5*60*5000: # only Patients need offset correction
                                data_offset = data[:,0:10].mean(axis=-1) - prev_data[:,-10:].mean(axis=-1)
                                data -= data_offset.reshape(-1,1)
                            new_data = np.concatenate((prev_data, data), axis=-1)

                        # jump = np.mean(np.abs(prev_data[:,-1]-data[:,0])*2./(np.std(prev_data[:,-4000:],axis=-1)+np.std(data[:,:4000],axis=-1)))
                        # if jump < 0.7:
                        # if ng==1:
                        #     # gen new data :)
                        #     axis = prev_data.ndim - 1
                        #     def split(d):
                        #         return np.split(d, 2, axis=axis)
                        #     new_data = np.concatenate((split(prev_data)[1], split(data)[0]), axis=axis)
                        #     X.append(pipeline.apply(new_data))
                        #     y.append(0) # seizure
                        #     latencies.append(sequence-0.5)
                        # else:
                        n = data.shape[1]
                        s = n / (ng + 1.)
                        # new_data = np.concatenate((prev_data, data), axis=-1)
                        for i in range(1,ng+1):
                            start = int(s*i)
                            X.append(pipeline.apply(new_data[:,start:(start+n)]))
                            y.append(label) # seizure
                            latencies.append(sequence-1.+i/(ng+1.))
                    y.append(label) # seizure
                    latencies.append(float(sequence))
                else:
                    y.append(label) # no seizure

            X.append(transformed_data)
            prev_data = data
            prev_sequence = sequence

        print '(%ds)' % (time.get_seconds() - start)

        X = np.array(X)
        y = np.array(y)
        latencies = np.array(latencies)

        if ictal or preictal or interictal:
            print 'X', X.shape, 'y', y.shape, 'latencies', latencies.shape
            return X, y, latencies
        # elif interictal:
        #     print 'X', X.shape, 'y', y.shape
        #     return X, y
        else:
            print 'X', X.shape
            return X

    data = process_raw_data(mat_data, with_latency=ictal)

    if len(data) == 3:
        X, y, latencies = data
        return {
            'X': X,
            'y': y,
            'latencies': latencies
        }
    elif len(data) == 2:
        X, y = data
        return {
            'X': X,
            'y': y
        }
    else:
        X = data
        return {
            'X': X
        }


# flatten data down to 2 dimensions for putting through a classifier
def flatten(data):
    if data.ndim > 2:
        return data.reshape((data.shape[0], np.product(data.shape[1:])))
    else:
        return data


# split up ictal and interictal data into training set and cross-validation set
def prepare_training_data(ictal_data, interictal_data, cv_ratio, withlatency=False):
    print 'Preparing training data ...',
    ictal_X, ictal_y = flatten(ictal_data.X), ictal_data.y
    interictal_X, interictal_y = flatten(interictal_data.X), interictal_data.y

    # split up data into training set and cross-validation set for both seizure and early sets
    if withlatency:
        ictal_X_train, ictal_y_train, ictal_X_cv, ictal_y_cv = split_train_ictal(ictal_X, ictal_y, ictal_data.latencies, cv_ratio)
    else:
        ictal_X_train, ictal_y_train, ictal_X_cv, ictal_y_cv = split_train_random(ictal_X, ictal_y, cv_ratio)
    interictal_X_train, interictal_y_train, interictal_X_cv, interictal_y_cv = split_train_random(interictal_X, interictal_y, cv_ratio)

    def concat(a, b):
        return np.concatenate((a, b), axis=0)

    X_train = concat(ictal_X_train, interictal_X_train)
    y_train = concat(ictal_y_train, interictal_y_train)
    X_cv = concat(ictal_X_cv, interictal_X_cv)
    y_cv = concat(ictal_y_cv, interictal_y_cv)

    y_classes = np.unique(concat(y_train, y_cv))

    start = time.get_seconds()
    elapsedSecs = time.get_seconds() - start
    print "%ds" % int(elapsedSecs)

    print 'X_train:', np.shape(X_train)
    print 'y_train:', np.shape(y_train)
    print 'X_cv:', np.shape(X_cv)
    print 'y_cv:', np.shape(y_cv)
    print 'y_classes:', y_classes

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_cv': X_cv,
        'y_cv': y_cv,
        'y_classes': y_classes
    }


# split interictal segments at random for training and cross-validation
def split_train_random(X, y, cv_ratio):
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=cv_ratio, random_state=0)
    return X_train, y_train, X_cv, y_cv


# split ictal segments for training and cross-validation by taking whole seizures at a time
def split_train_ictal(X, y, latencies, cv_ratio):
    seizure_ranges = seizure_ranges_for_latencies(latencies)
    seizure_durations = [r[1] - r[0] for r in seizure_ranges]

    num_seizures = len(seizure_ranges)
    num_cv_seizures = int(max(1.0, num_seizures * cv_ratio))

    # sort seizures by biggest duration first, then take the middle chunk for cross-validation
    # and take the left and right chunks for training
    tagged_durations = zip(range(len(seizure_durations)), seizure_durations)
    tagged_durations.sort(cmp=lambda x,y: cmp(y[1], x[1]))
    middle = num_seizures / 2
    half_cv_seizures = num_cv_seizures / 2
    start = middle - half_cv_seizures
    end = start + num_cv_seizures

    chosen = tagged_durations[start:end]
    chosen.sort(cmp=lambda x,y: cmp(x[0], y[0]))
    cv_ranges = [seizure_ranges[r[0]] for r in chosen]

    train_ranges = []
    prev_end = 0
    for start, end in cv_ranges:
        train_start = prev_end
        train_end = start

        if train_start != train_end:
            train_ranges.append((train_start, train_end))

        prev_end = end

    train_start = prev_end
    train_end = len(latencies)
    if train_start != train_end:
        train_ranges.append((train_start, train_end))

    X_train_chunks = [X[start:end] for start, end in train_ranges]
    y_train_chunks = [y[start:end] for start, end in train_ranges]

    X_cv_chunks = [X[start:end] for start, end in cv_ranges]
    y_cv_chunks = [y[start:end] for start, end in cv_ranges]

    X_train = np.concatenate(X_train_chunks)
    y_train = np.concatenate(y_train_chunks)
    X_cv = np.concatenate(X_cv_chunks)
    y_cv = np.concatenate(y_cv_chunks)

    return X_train, y_train, X_cv, y_cv


# train classifier for cross-validation
def train(classifier, X_train, y_train, X_cv, y_cv, y_classes):
    print "Training ..."

    print 'Dim', 'X', np.shape(X_train), 'y', np.shape(y_train), 'X_cv', np.shape(X_cv), 'y_cv', np.shape(y_cv)
    start = time.get_seconds()
    classifier.fit(X_train, y_train)
    print "Scoring..."
    S, E = score_classifier_auc(classifier, X_cv, y_cv, y_classes)
    score = 0.5 * (S + E)

    elapsedSecs = time.get_seconds() - start
    print "t=%ds score=%f" % (int(elapsedSecs), score)
    return score, S, E


# train classifier for predictions
def train_all_data(classifier, X_train, y_train, X_cv, y_cv):
    print "Training ..."
    X = np.concatenate((X_train, X_cv), axis=0)
    y = np.concatenate((y_train, y_cv), axis=0)
    print 'Dim', np.shape(X), np.shape(y)
    start = time.get_seconds()
    classifier.fit(X, y)
    elapsedSecs = time.get_seconds() - start
    print "t=%ds" % int(elapsedSecs)


# sub mean divide by standard deviation
def normalize_data(X_train, X_cv):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_cv = scaler.transform(X_cv)

    return X_train, X_cv

# depending on input train either for predictions or for cross-validation
def train_classifier(classifier, data, use_all_data=False, normalize=False):
    X_train = data.X_train
    y_train = data.y_train
    X_cv = data.X_cv
    y_cv = data.y_cv

    if normalize:
        X_train, X_cv = normalize_data(X_train, X_cv)

    if not use_all_data:
        score, S, E = train(classifier, X_train, y_train, X_cv, y_cv, data.y_classes)
        return {
            'classifier': classifier,
            'score': score,
            'S_auc': S,
            'E_auc': E
        }
    else:
        train_all_data(classifier, X_train, y_train, X_cv, y_cv)
        return {
            'classifier': classifier
        }


# convert the output of classifier predictions into (Seizure, Early) pair
def translate_prediction(prediction, y_classes):
    if len(prediction) == 3:
        # S is 1.0 when ictal <=15 or >15
        # S is 0.0 when interictal is highest
        ictalLTE15, ictalGT15, interictal = prediction
        S = ictalLTE15 + ictalGT15
        E = ictalLTE15
        return S, E
    elif len(prediction) == 2:
        # 1.0 doesn't exist for Patient_4, i.e. there is no late seizure data
        if not np.any(y_classes == 1.0):
            ictalLTE15, interictal = prediction
            S = ictalLTE15
            E = ictalLTE15
            # y[i] = 0 # ictal <= 15
            # y[i] = 1 # ictal > 15
            # y[i] = 2 # interictal
            return S, E
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


# use the classifier and make predictions on the test data
def make_predictions(target, X_test, y_classes, classifier_data):
    classifier = classifier_data.classifier
    predictions_proba = classifier.predict_proba(X_test)

    lines = []
    for i in range(len(predictions_proba)):
        p = predictions_proba[i]
        S, E = translate_prediction(p, y_classes)
        if task_predict:
            lines.append('%s_test_segment_%04d.mat,%.15f' % (target, i+1, S))
        else:
            lines.append('%s_test_segment_%d.mat,%.15f,%.15f' % (target, i+1, S, E))

    return {
        'data': '\n'.join(lines)
    }


# the scoring mechanism used by the competition leaderboard
def score_classifier_auc(classifier, X_cv, y_cv, y_classes):
    predictions = classifier.predict_proba(X_cv)
    S_predictions = []
    E_predictions = []
    S_y_cv = [1.0 if (x == 0.0 or x == 1.0) else 0.0 for x in y_cv]
    E_y_cv = [1.0 if x == 0.0 else 0.0 for x in y_cv]

    for i in range(len(predictions)):
        p = predictions[i]
        S, E = translate_prediction(p, y_classes)
        S_predictions.append(S)
        E_predictions.append(E)

    fpr, tpr, thresholds = roc_curve(S_y_cv, S_predictions)
    S_roc_auc = auc(fpr, tpr)
    fpr, tpr, thresholds = roc_curve(E_y_cv, E_predictions)
    E_roc_auc = auc(fpr, tpr)

    return S_roc_auc, E_roc_auc

