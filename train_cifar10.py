# import system modules
import os, numpy as np
from typing import List

# import tensorflow modules
import tensorflow as tf
from tensorflow import keras as K

# import core modules
from applications.mobilenetv2 import Cifar10OctMobileNetV2 as Model

# main function
def train(max_oct_ratio=0.25, oct_mode: str='constant', root_dir: str='Data', epochs: int=10, initial_epoch: int=0, batch_size: int=5, experiment_name: str="test", output_dir: str='models'):
    # initialize directory
    ratio_dir = os.path.join(root_dir, experiment_name, 'non_compress', 'ratio_{}_{}'.format(str(max_oct_ratio), oct_mode))
    tensorboard_dir = os.path.join(ratio_dir, 'logs')
    last_ckpt_dir = os.path.join(ratio_dir, "ckpt", "last")
    best_ckpt_dir = os.path.join(ratio_dir, "ckpt", "best")
    
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
    # steps_per_epoch: int = int(len(x_train) / batch_size)

    # set up learning rate
    lr = 0.1

    # set up lr schedule
    def lr_schedule(epoch: int) -> float:
        if epoch >= 250:
            return lr * 0.01
        elif epoch >= 150:
            return lr * 0.1
        else:
            return lr

    # initialize mirrored strategy
    mirrored_strategy: tf.distribute.MirroredStrategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    # mirrored strategy scope
    with mirrored_strategy.scope():
        # initialize model
        model = Model(max_oct_ratio=max_oct_ratio, mode=oct_mode) # Octave Model
        model.is_compressible = False

        # initialize weight optimizer
        optimizer: K.optimizers.optimizer = K.optimizers.SGD(learning_rate=lr, momentum=0.9)

        # initialize model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # restore checkpoint
        if initial_epoch > 0:
#             model.evaluate(x_test, y_test, steps=1, verbose=0)
            model.load_weights(last_ckpt_dir)
            model.evaluate(x_test, y_test, batch_size=batch_size)
        
    # training callbacks
    tensorboard_callback = K.callbacks.TensorBoard(log_dir=tensorboard_dir)
    last_ckpt_callback = K.callbacks.ModelCheckpoint(filepath=last_ckpt_dir, save_weights_only=True)
    best_ckpt_callback = K.callbacks.ModelCheckpoint(filepath=best_ckpt_dir, monitor='val_accuracy', save_weights_only=True, save_best_only=True, mode='max')
    lr_schedule_callback = K.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    callbacks: List[K.callbacks.Callback] = [tensorboard_callback, last_ckpt_callback, best_ckpt_callback, lr_schedule_callback]
    
    # preprocessing generator
    data_generator = K.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=False, validation_split=0.1)

    # train model
    model.fit(data_generator.flow(x_train, y_train, batch_size=batch_size, subset='training'), epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks, validation_data=data_generator.flow(x_train, y_train, batch_size=batch_size, subset='validation'))
    
    # evaluate model
    model.evaluate(x_test, y_test, batch_size=batch_size)
    model.save_weights(output_dir)
