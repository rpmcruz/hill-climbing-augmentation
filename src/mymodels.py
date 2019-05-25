import tensorflow as tf
import mymetrics
from tensorflow.keras import models, layers, backend
import numpy as np

def balanced_binary_crossentropy(Y, Yhat):
    Yhat = backend.clip(Yhat, backend.epsilon(), 1-backend.epsilon())
    N = tf.to_float(tf.size(Y))
    #N = 100*128*128  # average
    W0 = N / (2 * backend.sum(1-Y))
    W1 = N / (2 * backend.sum(Y))
    return -backend.mean(W0*(1-Y)*backend.log(1-Yhat) + W1*Y*backend.log(Yhat))

def create_unet(input_shape, output_shape):
    # all children should be initialized similarly
    tf.set_random_seed(123)

    x = input_layer = layers.Input(input_shape)

    # add convolutional layers until reaching final_width
    final_width = 6
    nlayers = int(np.log2(input_shape[0]/final_width))
    nfilters = 32
    reps = 1

    # encoding layers
    skip_layers = []
    for _ in range(nlayers):
        for _ in range(reps):
            x = layers.Conv2D(nfilters, 3, padding='same', activation='relu')(x)
        skip_layers.append(x)
        x = layers.MaxPooling2D()(x)

    # bottleneck
    for _ in range(reps):
        x = layers.Conv2D(nfilters, 3, padding='same', activation='relu')(x)

    # decoder
    for i in range(nlayers):
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([skip_layers[-i-1], x])
        for _ in range(reps):
            x = layers.Conv2D(nfilters, 3, padding='same', activation='relu')(x)

    x = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    model = models.Model(input_layer, x)
    model.compile('adam', balanced_binary_crossentropy, ['accuracy', mymetrics.jaccard_distance])
    return model

def create_model(input_shape, output_shape):
    # all children should be initialized similarly
    tf.set_random_seed(123)

    x = input_layer = layers.Input(input_shape)

    # add convolutional layers until reaching final_width
    final_width = 6
    nlayers = int(np.log2(input_shape[0]/final_width))
    nfilters = 32
    reps = 1

    for _ in range(nlayers):
        for _ in range(reps):
            x = layers.Conv2D(nfilters, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    for _ in range(2):
        x = layers.Dense(nfilters, activation='relu')(x)

    x = layers.Dense(output_shape, activation='softmax')(x)
    model = models.Model(input_layer, x)
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    return model
