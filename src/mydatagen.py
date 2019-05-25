from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

SAVE = None

def ImageGenerator(D, X, Y, batchsize, seed, prefix=''):
    # it's important to change type because some Keras transformations like channel_shift_range
    # can commit overflow if the image is uint8
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    return ImageDataGenerator(**D).flow(X, Y, batchsize, seed=seed, save_to_dir=SAVE, save_prefix=prefix)

def SegmentationGenerator(D, X, Y, batchsize, seed):
    # HACK: some transformations like channel_shift_range, have different behaviors
    # for images and masks because some are RGB and the others are Gray
    hack_rgb = X.shape[3] == 3 and 'channel_shift_range' in D and D['channel_shift_range']
    if hack_rgb:
        Y = np.repeat(Y, 3, 3)

    #seed = np.random.randint(2**32-1, dtype=np.uint32)
    gen1 = ImageGenerator(D, X, Y, batchsize, seed, 'img')
    gen2 = ImageGenerator(D, Y, X, batchsize, seed, 'mask')
    while True:
        X, _ = gen1.next()
        Y, _ = gen2.next()
        if hack_rgb:
            Y = Y[:, :, :, 0][:, :, :, np.newaxis]
        # HACK: Transformations like brightness change colors.
        # For masks, we need to convert them back to BW.
        Y = Y >= 0.5
        yield X, Y

if __name__ == '__main__':
    import mydatasets
    import sys
    import matplotlib.pyplot as plt
    (Xtr, Ytr), _, _ = mydatasets.load(sys.argv[1], False)
    #Xtr = np.mean(Xtr, axis=3, keepdims=True)  # to gray-scale
    #Xtr = Xtr[:, :, :, np.newaxis]
    print(Xtr.shape, Ytr.shape)
    D = {
        'rotation_range': 180,
        'width_shift_range': 0.5, 'height_shift_range': 0.5,
        'channel_shift_range': 50,
        'brightness_range': (0.5, 2),
        'rescale': 1/255, 'fill_mode': 'constant'}
    gen = SegmentationGenerator(D, Xtr, Ytr, 5)
    X, Y = next(gen)
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(X[i])
        plt.subplot(2, 5, i+6)
        plt.imshow(Y[i, :, :, 0])
    plt.show()
