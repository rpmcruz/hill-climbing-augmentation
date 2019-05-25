from tensorflow.keras import utils, datasets
import numpy as np
import os

def prepare(X, Y):
    if X.max() > 1:
        X = (X / 255).astype(np.float32)  # normalization
    if len(Y.shape) >= 3:  # segmentations
        Y = (Y / 255).astype(np.float32)
    else:  # classes
        Y = utils.to_categorical(Y)
    return X, Y

def load(path, prepare_data):
    state = np.random.RandomState(124)
    val = None
    if hasattr(datasets, path):
        module = getattr(datasets, path)
        tr, ts = module.load_data()
        if len(tr[0].shape) == 3:  # monochrome
            tr = tr[0][:, :, :, np.newaxis], tr[1]
            ts = ts[0][:, :, :, np.newaxis], ts[1]
        if len(tr[1].shape) == 2:  # it contains 2 axis, remove one
            tr = tr[0], tr[1][:, 0]
            ts = ts[0], ts[1][:, 0]
    else:
        if os.path.exists(os.path.join(path, 'X.npy')):
            X = np.load(os.path.join(path, 'X.npy'))
            Y = np.load(os.path.join(path, 'Y.npy'))
            # split 60-20-20 train-val-test
            ix = state.choice(len(X), len(X), False)
            ix = np.split(ix, [int(0.6*len(X)), int(0.8*len(X))])
            tr = X[ix[0]], Y[ix[0]]
            val = X[ix[1]], Y[ix[1]]
            ts = X[ix[2]], Y[ix[2]]
        else:
            X = np.load(os.path.join(path, 'Xtr.npy'))
            Y = np.load(os.path.join(path, 'Ytr.npy'))
            tr = X, Y
            X = np.load(os.path.join(path, 'Xts.npy'))
            Y = np.load(os.path.join(path, 'Yts.npy'))
            ts = X, Y
            if os.path.exists(os.path.join(path, 'Xval.npy')):
                X = np.load(os.path.join(path, 'Xval.npy'))
                Y = np.load(os.path.join(path, 'Yval.npy'))
                val = X, Y

    if val is None:
        # if there is no validation set, use 50-50 of the test images
        X, Y = ts
        ix = state.choice(len(X), len(X), False)
        ix = np.split(ix, [int(0.5*len(X))])
        ts = X[ix[0]], Y[ix[0]]
        val = X[ix[1]], Y[ix[1]]

    if prepare_data:
        tr = prepare(*tr)
        val = prepare(*val)
        ts = prepare(*ts)
    return tr, val, ts
