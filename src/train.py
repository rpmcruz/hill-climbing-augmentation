import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--transformations', nargs='*', default=[])
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--burn-epochs', type=int, default=0)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--step', type=float, default=0.1)
parser.add_argument('--out', default='out')
args = parser.parse_args()
print(args)

import mydatasets, mymodels, mymetrics
from sklearn.metrics import jaccard_similarity_score, balanced_accuracy_score
from time import time
import numpy as np
import uuid
import sys
import os

if args.transformations == ['all']:
    print('Using all transformations!')
    args.transformations = [
        'rotation', 'shift', 'brightness', 'shear', 'zoomin', 'zoomout', 'channel']

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

class ModelAug:
    # Each value within the transformation is within [0, 1]
    def __init__(self, with_noise):
        # we may want to force noise because it ensures some data augmentation is
        # always applied. This is a hack to ensure that the random seed is consistent.
        self.min_value = 1e-8 if with_noise else 0
        self.values = [self.min_value] * len(args.transformations)
        self.filename = 'temp-%s.h5' % uuid.uuid4()

    def __del__(self):
        os.remove(self.filename)

    def load(self):
        backend.clear_session()
        self.model = models.load_model(self.filename, custom_objects={'jaccard_distance': mymetrics.jaccard_distance, 'balanced_binary_crossentropy': mymodels.balanced_binary_crossentropy})
        return self.model

    def save(self):
        self.model.save(self.filename)

    def inc(self, i):
        self.values[i] = min(1, self.values[i] + args.step)

    def dec(self, i):
        self.values[i] = max(self.min_value, self.values[i] - args.step)

    def todict(self):
        d = {'fill_mode': 'constant', 'rescale': 1/255}
        zoom_range = [1, 1]  # zoom is a special case
        for t, v in zip(args.transformations, self.values):
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
filename = '%s/%s-%s' % (args.out, dataname, '-'.join(args.transformations))

(Xtr, Ytr), (Xval, Yval), (Xts, Yts) = mydatasets.load(args.dataset, False)
print('Data sizes:')
print('Train:     ', len(Xtr))
print('Validation:', len(Xval))
print('Test:      ', len(Xts))
assert Xtr.max() > 1

if len(Ytr.shape) < 3:  # classification
    print('-> Classification dataset')
    W = len(Ytr) / ((Ytr.max()+1)*np.bincount(Ytr.ravel()))
    class_weight = {i: v for i, v in enumerate(W)}
    print('class weight:', class_weight)

    Ytr = utils.to_categorical(Ytr)
    Yval = utils.to_categorical(Yval)
    Yts = utils.to_categorical(Yts)
    Ytr2 = Ytr
    print('Number of classes:', Ytr.shape[1])
    model = mymodels.create_model(Xtr.shape[1:], Ytr.shape[1])
    evaluate_metric = classification_metric
else:  # segmentation
    print('-> Segmentation dataset')
    class_weight = None

    Ytr2 = (Ytr / 255).astype(np.float32)
    Yval = (Yval / 255).astype(np.float32)
    Yts = (Yts / 255).astype(np.float32)
    model = mymodels.create_unet(Xtr.shape[1:], Ytr.shape[1])
    evaluate_metric = segmentation_metric

model.summary()
steps = 2 * int(np.ceil(len(Xtr) / args.batchsize))

model.fit(Xtr/255, Ytr2, args.batchsize, args.burn_epochs, 2, class_weight=class_weight)

model1 = ModelAug(args.transformations != [])
model.save(model1.filename)

if args.transformations:
    model2 = ModelAug(True)
    model.save(model2.filename)

pmodels = [model1, model2] if args.transformations else [model1]
history_transformations = []
state = np.random.RandomState(123)

for epoch in range(args.burn_epochs, args.epochs):
    print('Epoch %d/%d' % (epoch+1, args.epochs))
    tic = time()
    seed = state.randint(2**32-1, dtype=np.uint32)
    if args.transformations:
        t = state.randint(len(args.transformations))
        pmodels[0].dec(t)
        pmodels[1].inc(t)

    val_metric = [0, 0]  # maximize
    val_loss = [0, 0]  # minimize (if tied)

    for i, model in enumerate(pmodels):
        #print('train model:', i)
        #print(model.values)
        #print(model.todict())

        D = model.todict()
        last_policy = D
        if len(Ytr.shape) < 3:  # classification
            gen = ImageGenerator(D, Xtr, Ytr, args.batchsize, seed)
        else:  # segmentation
            gen = SegmentationGenerator(D, Xtr, Ytr, args.batchsize, seed)

        model.load()
        model.model.fit_generator(gen, steps, epoch+1, verbose=0, initial_epoch=epoch,
            class_weight=class_weight)

        Yhat = model.model.predict(Xval/255, args.batchsize)
        val_metric[i] = evaluate_metric(Yval, Yhat)
        val_loss[i] = model.model.evaluate(Xval/255, Yval, args.batchsize, 0)[0]
        model.save()

    best = 0
    if args.transformations:
        # check which metric is higher; if tied, use loss
        if val_metric[0] == val_metric[1]:
            print('TIE!')
            best = np.argmin(val_loss)
        else:
            best = np.argmax(val_metric)
        worst = 1 - best
        print(' - %d beats %d | %s' % (best, worst, ' - '.join(map(str, pmodels[best].values))))

        if best == 0:
            pmodels[best].load()
        pmodels[best].model.save(pmodels[worst].filename)
        pmodels[worst].values = pmodels[best].values.copy()

        line = [epoch] + pmodels[best].values + [val_loss[best], val_metric[best]]
    else:
        line = [epoch, val_loss[0], val_metric[0]]
    history_transformations.append(line)
    toc = time()
    print(' - Elapsed time: %ds - Val metric: %f' % (toc-tic, val_metric[best]))

header = 'epoch,' + ','.join(args.transformations) + ',val_loss,val_metric'
np.savetxt(filename + '.csv', history_transformations, delimiter=',', header=header, comments='')
model = pmodels[0].load()
model.save(filename + '.h5')

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

## TRAIN again with only the final policy
# just to see if the impact is of the final or not

if len(Ytr.shape) < 3:  # classification
    model = mymodels.create_model(Xtr.shape[1:], Ytr.shape[1])
else:  # segmentation
    model = mymodels.create_unet(Xtr.shape[1:], Ytr.shape[1])

state = np.random.RandomState(123)
seed = state.randint(2**32-1, dtype=np.uint32)
D = last_policy
if len(Ytr.shape) < 3:  # classification
    gen = ImageGenerator(D, Xtr, Ytr, args.batchsize, seed)
else:  # segmentation
    gen = SegmentationGenerator(D, Xtr, Ytr, args.batchsize, seed)

model.fit_generator(gen, steps, args.epochs, verbose=2, class_weight=class_weight)

print('MODEL WITH ONLY the last policy')
Yhat = model.predict(Xts/255, args.batchsize)
print('final score:', evaluate_metric(Yts, Yhat))
print()

with open('evaluation.txt', 'a') as f:
    print(args, file=f)

    Yhat = model.predict(Xtr/255, args.batchsize)
    print('last policy train:', evaluate_metric(Ytr2, Yhat), file=f)
    Yhat = model.predict(Xval/255, args.batchsize)
    print('last policy val:', evaluate_metric(Yval, Yhat), file=f)
    Yhat = model.predict(Xts/255, args.batchsize)
    print('last policy test:', evaluate_metric(Yts, Yhat), file=f)
    print('', file=f)
