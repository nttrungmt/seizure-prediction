#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os.path
import numpy as np
from common import time
from common.data import CachedDataLoader, makedirs
from common.pipeline import *
from seizure.transforms import *
from seizure.tasks import TaskCore, CrossValidationScoreTask, MakePredictionsTask, TrainClassifierTask, \
    TrainingDataTask, LoadTestDataTask
import seizure.tasks
from seizure.scores import get_score_summary, print_results

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def run_seizure_detection(build_target, targets=None):
    """
    The main entry point for running seizure-detection cross-validation and predictions.
    Directories from settings file are configured, classifiers are chosen, pipelines are
    chosen, and the chosen build_target ('cv', 'predict', 'train_model') is run across
    all combinations of (targets, pipelines, classifiers)
    """

    with open('SETTINGS.json') as f:
        settings = json.load(f)

    data_dir = str(settings['competition-data-dir'])
    cache_dir = str(settings['data-cache-dir'])
    import seizure.transforms
    seizure.transforms.cache_dir = cache_dir
    submission_dir = str(settings['submission-dir'])
    seizure.tasks.task_predict = str(settings.get('task')) == 'predict'

    makedirs(submission_dir)

    cached_data_loader = CachedDataLoader(cache_dir)

    ts = time.get_millis()

    if not targets:
        if seizure.tasks.task_predict:
            # add leader-board weight to each target. I am using the number of test example as the weight assuming
            # all test examples are weighted equally on the leader-board
            targets = [
                ('Dog_1',502),
                ('Dog_2',1000),
                ('Dog_3',907),
                ('Dog_4',990),
                ('Dog_5',191),
                ('Patient_1',195),
                ('Patient_2',150),
            ]
        else:
            targets = [
                'Dog_1',
                'Dog_2',
                'Dog_3',
                'Dog_4',
                'Dog_5',
                'Patient_1',
                'Patient_2',
                'Patient_3',
                'Patient_4',
                'Patient_5',
                'Patient_6',
                'Patient_7',
                'Patient_8'
            ]

    pipelines = [
        # NOTE(mike): you can enable multiple pipelines to run them all and compare results
        #Pipeline(gen_ictal=False, pipeline=[FFT(), Slice(1, 48), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[FFT(), Slice(1, 64), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[FFT(), Slice(1, 96), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[FFT(), Slice(1, 128), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[FFT(), Slice(1, 160), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[FFT(), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[Stats()]),
        # Pipeline(gen_ictal=False, pipeline=[DaubWaveletStats(4)]),
        # Pipeline(gen_ictal=False, pipeline=[Resample(400), DaubWaveletStats(4)]),
        # Pipeline(gen_ictal=False, pipeline=[Resample(400), MFCC()]),
        # Pipeline(gen_ictal=False, pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'us')]),
        # Pipeline(gen_ictal=True,  pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'us')]),
        #Pipeline(gen_ictal=False, pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')]), # winning detection submission
        # Pipeline(gen_ictal=False, pipeline=[WindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600)]),
        #Pipeline(gen_ictal=False, pipeline=[StdWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600)]),
        #Pipeline(gen_ictal=False, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600)]),
        # Pipeline(gen_ictal=True, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600)]),
        # Pipeline(gen_ictal=2, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600)]),
        # Pipeline(gen_ictal=4, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600)]),
        #Pipeline(gen_ictal=8, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600)]),
        #Pipeline(gen_ictal=-8, pipeline=[MedianWindow1FFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600)]),
        # Pipeline(gen_ictal=-8, pipeline=[MedianWindowBands1('usf2', 60, p=2)]),
        # Pipeline(gen_ictal=-8, pipeline=[MedianWindowBands1('141022-PCA-model', 60, p=2)]),
        #Pipeline(gen_ictal=-8, pipeline=[MedianWindowBands1('141022-ICA-model-1', 60, p=2)]),
        # Pipeline(gen_ictal=-8, pipeline=[MedianWindowBands1('ica', 60, p=2, timecorr=True)]),
        #Pipeline(gen_ictal=-8.5, pipeline=[MedianWindowBands1('usf', 60, p=2)]),
        #Pipeline(gen_ictal=-8, pipeline=[MedianWindowBands('usf', 10, p=2, window='hammingP2')]),
        #Pipeline(gen_ictal=-8, pipeline=[AllBands('usf', 60)]),
        Pipeline(gen_ictal=-8, pipeline=[AllTimeCorrelation('usf', 60)]),
        #Pipeline(gen_ictal=-8, pipeline=[MaxDiff(60)]),
        #Pipeline(gen_ictal=-8, pipeline=[MedianWindowBandsTimeCorrelation('usf', 60)]),
        #Pipeline(gen_ictal=-8, pipeline=[MedianWindowBandsCorrelation('usf', 60)]),
        #Pipeline(gen_ictal=-8, pipeline=[MedianWindowTimeCorrelation('usf', 60)]),
        #Pipeline(gen_ictal=False, pipeline=[MedianWindow1FFTWithTimeFreqCorrelation(1, 49, 400, 'usf',600)]),
        #Pipeline(gen_ictal=8, pipeline=[MedianWindowFFTWithTimeFreqCov2(1, 48, 400, 'usf',600)]),
        #Pipeline(gen_ictal=8, pipeline=[CleanMedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600, window='hammingP2')]),
        #Pipeline(gen_ictal=8, pipeline=[CleanCorMedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600, window='hammingP2')]),
        #Pipeline(gen_ictal=8, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600,subsample=2)]),
        # Pipeline(gen_ictal=16, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600)]),
        #Pipeline(gen_ictal=16, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 96, 400, 'usf',600, window='hamming')]),
        #Pipeline(gen_ictal=16, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600, window='hamming2')]),
        #Pipeline(gen_ictal=8, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600, window='hamming0')]),
        # Pipeline(gen_ictal=8, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600, window='square0')]),
        # Pipeline(gen_ictal=2, pipeline=[Variance(nwindows=600)]),
        # UnionPipeline(gen_ictal=2, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600),Variance(nwindows=600)]),
        #Pipeline(gen_ictal=True, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600, nunits=4)]),
        #Pipeline(gen_ictal=True, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 50, 400, 'usf',600)]),
        #Pipeline(gen_ictal=False, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600,[0.5,0.9])]),
        # Pipeline(gen_ictal=False, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600,[0.1,0.9])]),
        # Pipeline(gen_ictal=False, pipeline=[MedianWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600,[0.05,0.5,0.95])]),
        # Pipeline(gen_ictal=False, pipeline=[BoxWindowFFTWithTimeFreqCorrelation(1, 48, 400, 'usf',600)]),
        # Pipeline(gen_ictal=True,  pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')]), # higher score than winning submission
        # Pipeline(gen_ictal=False, pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'none')]),
        # Pipeline(gen_ictal=True,  pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'none')]),
        # Pipeline(gen_ictal=False, pipeline=[TimeCorrelation(400, 'usf', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[TimeCorrelation(400, 'us', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[TimeCorrelation(400, 'us', with_corr=True, with_eigen=False)]),
        # Pipeline(gen_ictal=False, pipeline=[TimeCorrelation(400, 'us', with_corr=False, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[TimeCorrelation(400, 'none', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[FreqCorrelation(1, 48, 'usf', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[FreqCorrelation(1, 48, 'us', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[FreqCorrelation(1, 48, 'us', with_corr=True, with_eigen=False)]),
        # Pipeline(gen_ictal=False, pipeline=[FreqCorrelation(1, 48, 'us', with_corr=False, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[FreqCorrelation(1, 48, 'none', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[TimeFreqCorrelation(1, 48, 400, 'us')]),
        # Pipeline(gen_ictal=False, pipeline=[TimeFreqCorrelation(1, 48, 400, 'usf')]),
        # Pipeline(gen_ictal=False, pipeline=[TimeFreqCorrelation(1, 48, 400, 'none')]),
    ]
    classifiers = [
        # NOTE(mike): you can enable multiple classifiers to run them all and compare results
        # (RandomForestClassifier(n_estimators=50, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf50mss1Bfrs0'),
        # (RandomForestClassifier(n_estimators=150, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf150mss1Bfrs0'),
        # (RandomForestClassifier(n_estimators=300, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf300mss1Bfrs0'),
        # (RandomForestClassifier(n_estimators=3000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf3000mss1Bfrs0'),
        # (RandomForestClassifier(n_estimators=3000, min_samples_split=1, max_depth=10, bootstrap=True, n_jobs=-1, random_state=0), 'rf3000mss1md10Bt'),
        # (RandomForestClassifier(n_estimators=1000, min_samples_split=1, max_depth=10, bootstrap=False, n_jobs=-1, random_state=0), 'rf3000mss1md10Bf'),
        (RandomForestClassifier(n_estimators=3000, min_samples_split=1, max_depth=10, bootstrap=False, n_jobs=-1, random_state=0), 'rf3000mss1md10Bf'),
        # (RandomForestClassifier(n_estimators=10000, min_samples_split=1, max_depth=10, bootstrap=False, n_jobs=-1, random_state=0), 'rf10000mss1md10Bf'),
        # (RandomForestClassifier(n_estimators=3000, min_samples_split=1, max_depth=3, bootstrap=False, n_jobs=-1, random_state=0), 'rf3000mss1md3Bf'),
        # (RandomForestClassifier(n_estimators=3000, min_samples_split=1, max_depth=30, bootstrap=False, n_jobs=-1, random_state=0), 'rf3000mss1md30Bf'),
        # (RandomForestClassifier(n_estimators=3000, min_samples_split=1, max_depth=10, max_features='log2', bootstrap=False, n_jobs=-1, random_state=0), 'rf3000mss1md10BfmfL2'),
        # (RandomForestClassifier(n_estimators=3000, min_samples_split=1, max_depth=10, max_features=200, bootstrap=False, n_jobs=-1, random_state=0), 'rf3000mss1md10Bfmf200'),
        # (GradientBoostingClassifier(n_estimators=500,min_samples_split=1,),'gbc500mss1'),
        # (GradientBoostingClassifier(n_estimators=1000,min_samples_split=1, random_state=0),'gbc1000mss1'),
        # (GradientBoostingClassifier(n_estimators=1000,min_samples_split=1, random_state=0, learning_rate=0.03),'gbc1000mss1lr03'),
        # (GradientBoostingClassifier(n_estimators=1000,min_samples_split=1, random_state=0, learning_rate=0.01),'gbc1000mss1lr01'),
        # (GradientBoostingClassifier(n_estimators=1000,min_samples_split=1, random_state=0, learning_rate=0.01, max_depth=1000),'gbc1000mss1lr01md1000'),
   ]
    cv_ratio = 0.5

    def should_normalize(classifier):
        clazzes = [LogisticRegression]
        return np.any(np.array([isinstance(classifier, clazz) for clazz in clazzes]) == True)

    def train_full_model(make_predictions):
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                if seizure.tasks.task_predict:
                    guesses = ['clip,preictal']
                else:
                    guesses = ['clip,seizure,early']
                classifier_filenames = []
                for target in targets:
                    if isinstance(target,tuple):
                        target, leaderboard_weight = target
                    else:
                        leaderboard_weight = 1
                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_ictal=pipeline.gen_ictal,
                                         cv_ratio=cv_ratio)

                    if make_predictions:
                        predictions = MakePredictionsTask(task_core).run()
                        guesses.append(predictions.data)
                    else:
                        task = TrainClassifierTask(task_core)
                        task.run()
                        classifier_filenames.append(task.filename())

                if make_predictions:
                    filename = 'submission%d-%s_%s.csv' % (ts, classifier_name, pipeline.get_name())
                    filename = os.path.join(submission_dir, filename)
                    with open(filename, 'w') as f:
                        print >> f, '\n'.join(guesses)
                    print 'wrote', filename
                else:
                    print 'Trained classifiers ready in %s' % cache_dir
                    for filename in classifier_filenames:
                        print os.path.join(cache_dir, filename + '.pickle')

    def do_cross_validation():
        summaries = []
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                scores = []
                S_scores = []
                E_scores = []
                leaderboard_weights = []
                for target in targets:
                    if isinstance(target,tuple):
                        target, leaderboard_weight = target
                    else:
                        leaderboard_weight = 1
                    leaderboard_weights.append(leaderboard_weight)
                    print 'Processing %s (classifier %s)' % (target, classifier_name)

                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_ictal=pipeline.gen_ictal,
                                         cv_ratio=cv_ratio)

                    data = CrossValidationScoreTask(task_core).run()
                    score = data.score

                    scores.append(score)

                    print '%.3f' % score, 'S=%.4f' % data.S_auc, 'E=%.4f' % data.E_auc
                    S_scores.append(data.S_auc)
                    E_scores.append(data.E_auc)

                if len(scores) > 0:
                    name = pipeline.get_name() + '_' + classifier_name
                    weighted_average = np.average(scores, weights=leaderboard_weights)
                    summary = get_score_summary(name, scores, weighted_average)
                    summaries.append((summary, weighted_average))
                    print summary
                if len(S_scores) > 0:
                    name = pipeline.get_name() + '_' + classifier_name
                    summary = get_score_summary(name, S_scores, np.mean(S_scores))
                    print 'S', summary
                if len(E_scores) > 0:
                    name = pipeline.get_name() + '_' + classifier_name
                    summary = get_score_summary(name, E_scores, np.mean(E_scores))
                    print 'E', summary

            print_results(summaries)

    def do_train_data():
        for pipeline in pipelines:
            print 'Using pipeline %s' % (pipeline.get_name())
            for target in targets:
                if isinstance(target,tuple):
                    target, leaderboard_weight = target

                task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                     target=target, pipeline=pipeline,
                                         classifier_name=None, classifier=None,
                                         normalize=None, gen_ictal=pipeline.gen_ictal,
                                         cv_ratio=None)
                # call the load data tasks for positive and negative examples (ignore the merge of the two.)
                TrainingDataTask(task_core).run()

    def do_test_data():
        for pipeline in pipelines:
            print 'Using pipeline %s' % (pipeline.get_name())
            for target in targets:
                if isinstance(target,tuple):
                    target, leaderboard_weight = target

                task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                     target=target, pipeline=pipeline,
                                         classifier_name=None, classifier=None,
                                         normalize=None, gen_ictal=pipeline.gen_ictal,
                                         cv_ratio=None)

                LoadTestDataTask(task_core).run()

    if build_target == 'train_data':
        do_train_data()
    elif build_target == 'test_data':
        do_test_data()
    elif build_target == 'cv':
        do_cross_validation()
    elif build_target == 'train_model':
        train_full_model(make_predictions=False)
    elif build_target == 'make_predictions':
        train_full_model(make_predictions=True)
    else:
        raise Exception("unknown build target %s" % build_target)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Seizure prediction or detection')
    parser.add_argument('-b','--build',
                       help='select what we want to build: train_data, test_data, cv, train_model, make_predictions')
    parser.add_argument('-t', '--targets', nargs='+',
                       help='Target file(s) we want to process')

    args = parser.parse_args()
    if args.build == 'predict':
        args.build = 'make_predictions'
    elif args.build == 'train':
        args.build = 'train_model'
    elif args.build == 'td':
        args.build = 'train_data'
    elif args.build == 'tt':
        args.build = 'test_data'

    assert args.build in ['data', 'train_data', 'test_data', 'cv', 'train_model', 'make_predictions'], \
        'Illegal/missing build command %s'%args.build
    assert len(args.targets), 'No target(s) were given'

    if args.build == 'data':
        run_seizure_detection('train_data', targets=args.targets)
        run_seizure_detection('test_data', targets=args.targets)
    else:
        run_seizure_detection(args.build, targets=args.targets)
