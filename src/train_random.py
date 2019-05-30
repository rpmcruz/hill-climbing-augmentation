import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--transformations', nargs='*', default=[])
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--child-epochs', type=int, default=25)  # 50
parser.add_argument('--final-epochs', type=int, default=250)
parser.add_argument('--children', type=int, default=50)
parser.add_argument('--second-children', type=int, default=0)  # 0 to disable
parser.add_argument('--bayesian', action='store_true')
parser.add_argument('--bayesian-seed', type=int, default=10)
parser.add_argument('--out', default='out')
args = parser.parse_args()
assert args.transformations != []
print(args)

import mydatasets, mymodels, mymetrics
from sklearn.metrics import jaccard_similarity_score, balanced_accuracy_score
from time import time
import numpy as np
import uuid
import sys
import os

def segmentation_metric(Y, Yhat):
    return jaccard_similarity_score((Y >= 0.5).ravel(), (Yhat >= 0.5).ravel())

def classification_metric(Y, Yhat):
    return balanced_accuracy_score(Y.argmax(1), Yhat.argmax(1))

def create_transformation(t, v):
    if t == 'rotation':  # [0, 180]
        return {'rotation_range': int(v*180)}
    if t == 'shift':  # [0, 1]
        return {'width_shift_range': v, 'height_shift_range': v}
    if t == 'brightness':  # [0.5, 2]
        if v == 0:
            return {'brightness_range': None}
        return {'brightness_range': (1-v*0.5, 1+v*1)}
    if t == 'shear':  # [0, 60]
        return {'shear_range': v*60}
    if t == 'channel':  # [0, 50]
        return {'channel_shift_range': v*50}
    sys.error('Unknown transformation: %s' % t)

def todict(values):
    d = {'fill_mode': 'constant', 'rescale': 1/255}
    zoom_range = [1, 1]  # zoom is a special case
    for t, v in zip(args.transformations, values):
        if t == 'zoomin':  # [0.333, 1]
            zoom_range[0] = 1-(2*v/3)
        elif t == 'zoomout':  # [1, 3]
            zoom_range[1] = 1+(v*2)
        else:
            d.update(create_transformation(t, v))
    if zoom_range != [1, 1]:
        d['zoom_range'] = zoom_range
    return d

from tensorflow.keras import models, backend, utils
from mydatagen import ImageGenerator, SegmentationGenerator

dataname = args.dataset.strip('/').split('/')[-1]
filename = '%s/%s-%s-random' % (args.out, dataname, '-'.join(args.transformations))

(Xtr, Ytr), (Xval, Yval), (Xts, Yts) = mydatasets.load(args.dataset, False)

if len(Ytr.shape) < 3:  # classification
    print('-> Classification dataset')
    W = len(Ytr) / ((Ytr.max()+1)*np.bincount(Ytr.ravel()))
    class_weight = {i: v for i, v in enumerate(W)}
    print('class weight:', class_weight)

    Ytr = utils.to_categorical(Ytr)
    Yval = utils.to_categorical(Yval)
    Yts = utils.to_categorical(Yts)
    Ytr2 = Ytr
    create_model = mymodels.create_model
    evaluate_metric = classification_metric
else:  # segmentation
    print('-> Segmentation dataset')
    class_weight = None

    Ytr2 = (Ytr  / 255).astype(np.float32)
    Yval = (Yval / 255).astype(np.float32)
    Yts = (Yts / 255).astype(np.float32)
    create_model = mymodels.create_unet
    evaluate_metric = segmentation_metric

steps = 2 * int(np.ceil(len(Xtr) / args.batchsize))
#steps = 1000 // args.batchsize
history_transformations = []
state = np.random.RandomState(123)

def choose_next_random(H, S):
    return state.rand(len(args.transformations))

# There is a cool package called BayesianOptimization. It looks very good - but
# it's not very flexible. We cannot give our own sampling function, nor does it
# save the results from the seed points or all models scores (it keeps only the
# target).

def ei(x, gp, ymax, xi=0):
    mean, std = gp.predict(x, True)
    z = (mean-ymax-xi)/(std+1e-8)
    return (mean-ymax-xi)*norm.cdf(z) + std*norm.pdf(z)

def acq_max(gp, ymax, n_warmup=100000):
    xtries = state.rand(n_warmup, len(args.transformations))
    i = ei(xtries, gp, ymax).argmax()
    xmax = xtries[i]
    # the BayesianOptimization package does some further tweaking by using L-BFGS-B
    return xmax

from sklearn import gaussian_process
from scipy.stats import norm

def choose_next_bayesian(H, S):
    if len(H) < args.bayesian_seed:
        return state.rand(len(args.transformations))

    gp = gaussian_process.GaussianProcessRegressor(
        #gaussian_process.kernels.Matern(nu=2.5),
        alpha=1e-4, n_restarts_optimizer=25)
    gp.fit(np.array(H), np.array(S))
    return acq_max(gp, np.max(S))

choose_next = choose_next_bayesian if args.bayesian else choose_next_random

def train_children(children, child_epochs, use_children_values=None):
    children_val_metrics = []
    children_val_losses = []
    children_values = []
    children_lines = []
    historic = []
    
    for childi in range(children):
        print('Child %d/%d' % (childi+1, children))
        seed = 1234

        backend.clear_session()
        model = create_model(Xtr.shape[1:], Ytr.shape[1])

        if use_children_values is None:
            values = choose_next(historic, children_val_metrics)
            D = todict(values)
        else:
            D = use_children_values[childi]

        print(D)
        if len(Ytr.shape) < 3:  # classification
            gen = ImageGenerator(D, Xtr, Ytr, args.batchsize, seed)
        else:  # segmentation
            gen = SegmentationGenerator(D, Xtr, Ytr, args.batchsize, seed)

        model.fit_generator(gen, steps, child_epochs, verbose=2, class_weight=class_weight)

        Yhat = model.predict(Xval/255, args.batchsize)
        val_metric = evaluate_metric(Yval, Yhat)
        val_loss = model.evaluate(Xval/255, Yval, args.batchsize, 0)[0]
        print('Evaluation:', val_metric, val_loss)
        line = values.tolist() + [val_loss, val_metric]

        children_val_metrics.append(val_metric)
        children_val_losses.append(val_loss)
        children_values.append(D)
        children_lines.append(line)
        historic.append(values)

    return children_val_metrics, children_val_losses, children_values, children_lines

## First phase: train 50 children for 20 epochs (cost=1000)
print('# First phase')
children_val_metrics, children_val_losses, children_values, children_lines = \
    train_children(args.children, args.child_epochs)
children_val_metrics = [-m for m in children_val_metrics]  # less is better
lines = children_lines

# sort with primary/secondary key
ix = np.lexsort((children_val_losses, children_val_metrics))

## Second phase: train 10 children for 100 epochs (cost=1000)
if args.second_children > 1:
    print('# Second phase: train 10 models')
    best_values = [children_values[i] for i in ix[:args.second_children]]

    children_val_metrics, children_val_losses, children_values, children_lines = \
        train_children(args.second_children, args.final_epochs)
    children_val_metrics = [-m for m in children_val_metrics]  # less is better
    lines += children_lines

    # sort with primary/secondary key
    ix = np.lexsort((children_val_losses, children_val_metrics))

## Third phase: train final child for 100 epochs
print('# Final phase: final model...')
D = children_values[ix[0]]
print(D)
backend.clear_session()
model = create_model(Xtr.shape[1:], Ytr.shape[1])

# for the final model, train for more time...
steps = 2 * int(np.ceil(len(Xtr) / args.batchsize))

seed = 1234
if len(Ytr.shape) < 3:  # classification
    gen = ImageGenerator(D, Xtr, Ytr, args.batchsize, seed)
else:  # segmentation
    gen = SegmentationGenerator(D, Xtr, Ytr, args.batchsize, seed)
model.fit_generator(gen, steps, args.final_epochs, verbose=2, class_weight=class_weight)

# Save results

header = 'epoch,' + ','.join(args.transformations) + ',val_loss,val_metric'
np.savetxt(filename + '.csv', lines, header=header, comments='')

print(filename)
Yhat = model.predict(Xts/255, args.batchsize)
print('final score:', evaluate_metric(Yts, Yhat))
print()

with open('evaluation.txt', 'a') as f:
    print(args, file=f)

    Yhat = model.predict(Xtr/255, args.batchsize)
    print('train:', evaluate_metric(Ytr2, Yhat), file=f)
    Yhat = model.predict(Xval/255, args.batchsize)
    print('val:', evaluate_metric(Yval, Yhat), file=f)
    Yhat = model.predict(Xts/255, args.batchsize)
    print('test:', evaluate_metric(Yts, Yhat), file=f)
    print('', file=f)
