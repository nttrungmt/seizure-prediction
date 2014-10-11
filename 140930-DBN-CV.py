
# coding: utf-8

# In[1]:

import cPickle as pickle
import numpy as np
import random
from collections import defaultdict

import sys
# sys.path.insert(0,'/Users/udi/Documents/MyProjects/github/nolearn/')
## Read precomputed features

# uncommoent the relevant pipeline in `../seizure_detection.py` and run
# ```bash
# cd ..
# ./doall data
# ```

# In[42]:

FEATURES = 'gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9'


# In[43]:

from common.data import CachedDataLoader
cached_data_loader = CachedDataLoader('data-cache')


# In[44]:

def read_data(target, data_type):
    return cached_data_loader.load('data_%s_%s_%s'%(data_type,target,FEATURES),None)


## Predict

# In[45]:

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10,
                             n_jobs=-1) #min_samples_leaf=4


# In[63]:

from sklearn import preprocessing
from nolearn.dbn import DBN
from sklearn.pipeline import Pipeline

min_max_scaler = preprocessing.MinMaxScaler() # scale features to be [0..1] which is DBN requirement

dbn = DBN(
    [-1, 300, -1], # first layer has size X.shape[1], hidden layer(s), last layer will have number of classes in y (2))
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=0,
    )

clf = Pipeline([('min_max_scaler', min_max_scaler), ('dbn', dbn)])


# In[64]:

with_weights = False
PWEIGHT = 0.
LWEIGHT = 0.
suffix = 'DBN300.'


# split examples into segments, each from the same event
# in each CV-split we will take all examples from the same segment to either train or validate

# In[65]:

def getsegments(pdata):
    segments = []
    start = 0
    last_l = 0
    for i,l in enumerate(pdata.latencies):
        if l<last_l:
            segments.append(np.arange(start,i))
            start = i
        last_l = l
    segments.append(np.arange(start,i+1))
    return np.array(segments)


# Compute AUC for each target separatly

# In[68]:

import itertools
from sklearn.metrics import roc_auc_score

target2iter2ys = {}
for target in ['Patient_2']:#['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']:
    # positive examples
    pdata = read_data(target, 'preictal')
    Np, NF = pdata.X.shape
    
    clf.set_params(dbn__layer_sizes=[NF,300,2]) # we need to reset each time because NF is different

    psegments = getsegments(pdata)
    Nps = len(psegments)

    # negative examples
    ndata = read_data(target, 'interictal')
    Nn, NF = ndata.X.shape
    nsegments = getsegments(ndata)
    Nns = len(nsegments)
    
    npratio = float(Nn)/Np
    print target,1/(1+npratio),Np,Nn
    npsratio = float(Nns)/Nps
    print target,1/(1+npsratio),Nps,Nns
    Ntrainps = 1
    Ntrainns = int(Ntrainps*npsratio)

    iter2ys = defaultdict(list) # {niter: Ns *[(ytest,y_proba)]
    for s in psegments:
        Xtestp = pdata.X[s,:]
        weightstest = pdata.latencies[s] # latency for first segment is 1
        
        Ntrainp = len(s)
        Ntrainn = int(Ntrainp*npratio)
        
        good_iter = 0
        for niter in range(3):
#             n = np.array(random.sample(xrange(Nn),Ntrainn)) # segment based
            ns = np.array(random.sample(xrange(Nns),Ntrainns)) # sequence based
            n = np.array(list(itertools.chain(*nsegments[ns]))) # .ravel does not work - elements of nsegments are not of equal length
            Xtestn = ndata.X[n,:]

            Xtrainp = pdata.X[-s,:]
            Xtrainn = ndata.X[-n,:]

            Xtrain = np.concatenate((Xtrainp,Xtrainn))
            ytrain = np.concatenate((np.ones(Xtrainp.shape[0]),np.zeros(Xtrainn.shape[0])))
            perm = np.random.permutation(len(ytrain))
            ytrain = ytrain[perm]
            Xtrain = Xtrain[perm,:]

            Xtest = np.concatenate((Xtestp,Xtestn))
            ytest = np.concatenate((np.ones(Xtestp.shape[0]),np.zeros(Xtestn.shape[0])))

            if with_weights:
                weightsp = PWEIGHT*np.ones(Xtrainp.shape[0])
                weightsp += LWEIGHT * (pdata.latencies[-s]-1.) # latency for first segment is 1
                weightsn = np.ones(Xtrainn.shape[0]) 
                weights = np.concatenate((weightsp,weightsn))
                weights = weights[perm]

            minibatch_size = 64
            if Xtrain.shape[0] < minibatch_size:
                minibatch_size = Xtrain.shape[0]
            clf.set_params(dbn__minibatch_size=minibatch_size)


            if with_weights:
                clf.fit(Xtrain, ytrain, sample_weight=weights)
            else:
                clf.fit(Xtrain, ytrain)
            good_iter += 1

            y_proba = clf.predict_proba(Xtest)[:,1]
            iter2ys[good_iter].append((ytest, y_proba))
            
            auc = roc_auc_score(ytest, y_proba)
            print '%.3f'%auc,Ntrainp,np.mean(weightstest)
            if good_iter >= 2:
                break
    target2iter2ys[target] = iter2ys
    print


# In[52]:

fname = '../data-cache/test.pkl'
with open(fname,'wb') as fp:
    pickle.dump(target2iter2ys,fp,-1)


# In[53]:

fname


# Generate a single AUC score

# In[55]:

from sklearn.metrics import roc_auc_score
def p(a,b):
    return '%d E%d'%(1000*a,1000*b)

all_ytest = all_y_proba =None
all_aucs = []
for target, iter2ys in target2iter2ys.iteritems():
    target_ytest = target_y_proba =None
    target_aucs = []
    print target,
    for ys in iter2ys.itervalues():
        ytest = y_proba =None
        aucs = []
        for y in ys:
            yt, yp = y
            ytest = yt if ytest is None else np.concatenate((ytest,yt))
            y_proba = yp if y_proba is None else np.concatenate((y_proba,yp))
            aucs.append(roc_auc_score(yt, yp))
        print p(roc_auc_score(ytest, y_proba), np.mean(aucs)),
        target_aucs += aucs
        target_ytest = ytest if target_ytest is None else np.concatenate((target_ytest,ytest))
        target_y_proba = y_proba if target_y_proba is None else np.concatenate((target_y_proba,y_proba))
    print target,p(roc_auc_score(target_ytest, target_y_proba),np.mean(target_aucs))
    all_aucs += target_aucs
    all_ytest = target_ytest if all_ytest is None else np.concatenate((all_ytest,target_ytest))
    all_y_proba = target_y_proba if all_y_proba is None else np.concatenate((all_y_proba,target_y_proba))
#         if target == 'Dog_3':
#             pl.hist(target_aucs,alpha=0.5)
print p(roc_auc_score(all_ytest, all_y_proba),np.mean(all_aucs))


