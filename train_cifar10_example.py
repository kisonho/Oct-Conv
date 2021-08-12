# import system modules
import os, numpy as np
from typing import List

# import tensorflow modules
import tensorflow as tf
from tensorflow import keras as K

# import core modules
from applications.mobilenetv2 import Cifar10OctMobileNetV2 as Model

# main function
def train(max_oct_ratio=0.25, oct_mode='constant', root_dir='Data', epochs=10, initial_epoch=0, batch_size=5, experiment_name="test"):
    # initialize model
    model = Model(max_oct_ratio=max_oct_ratio, mode=oct_mode) # Octave Model
    model.is_compressible = False

    # initialize dataset
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255 
    x_test = x_test.astype("float32") / 255 
    x_train_mean: list = np.mean(x_train, axis=0)
    x_train -= np.mean(x_train_mean, axis=0)
    x_test -= np.mean(x_train_mean, axis=0)
    num_classes: int = 10
    y_train = K.utils.to_categorical(y_train, num_classes, dtype="int32")
    y_test = K.utils.to_categorical(y_test, num_classes, dtype="int32")

    # set up learning rate
    lr = 1e-3
    decay_steps = 10000
    decay_rate = 0.98

    # initialize weight optimizer
    scheduled_lr = K.optimizers.schedules.ExponentialDecay(lr, decay_steps, decay_rate, staircase=True)
    weights_optimizer = K.optimizers.Adam(learning_rate=scheduled_lr)

    # initialize loss and accuracy
    loss_object = K.losses.CategoricalCrossentropy(name='loss')
    accuracy = K.metrics.Accuracy(name='accuracy')

    # initialize model
    model.compile(optimizer=weights_optimizer, loss=loss_object, metrics=[accuracy])

    # restore checkpoint
    if initial_epoch > 0:
        model.evaluate(x_test, y_test, steps=1, verbose=0)
        latest_ckpt = tf.train.latest_checkpoint('checkpoints')
        model.load_weights(latest_ckpt)
        model.evaluate(x_test, y_test)

    # initialize directory
    ratio_dir = os.path.join(root_dir, experiment_name, 'non_compress', 'ratio_{}_{}'.format(str(max_oct_ratio), oct_mode))
    tensorboard_dir = os.path.join(ratio_dir, 'logs')
    last_ckpt_dir = os.path.join(ratio_dir, "ckpt", "last")
    best_ckpt_dir = os.path.join(ratio_dir, "ckpt", "best")
        
    # training callbacks
    tensorboard_callback = K.callbacks.TensorBoard(log_dir=tensorboard_dir)
    last_ckpt_callback = K.callbacks.ModelCheckpoint(filepath=last_ckpt_dir, save_weights_only=True)
    best_ckpt_callback = K.callbacks.ModelCheckpoint(filepath=best_ckpt_dir, monitor='val_accuracy', save_weights_only=True, save_best_only=True, mode='max')
    callbacks: List[K.callbacks.Callback] = [tensorboard_callback, last_ckpt_callback, best_ckpt_callback]
    
    # preprocessing generator
    data_generator = K.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=False, validation_split=0.1)

    # train model
    model.fit(data_generator.flow(x_train, y_train, batch_size=batch_size, subset='training'), epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks, validation_data=data_generator.flow(x_train, y_train, batch_size=batch_size, subset='validation'))
    model.save_weights("cifar10.h5")
