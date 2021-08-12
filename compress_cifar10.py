# import system modules
import os, numpy as np

# import tensorflow modules
import tensorflow as tf
from tensorflow import keras as K

# import core modules
from lib.optimizers import OctaveCompressionOptimizer, FineTuneOptimizer
from applications.mobilenetv2 import Cifar10OctMobileNetV2 as Model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# main function
def compress(model_dir: str='cifar10.h5', max_oct_ratio=0.25, oct_mode: str='constant', root_dir: str='Data/', epochs: int=5, batch_size: int=64, lr: float=5e-4, compression_lambda: float=5, bayesian_steps: int=50, frequency_multiplier: float=1.5, experiment_name: str="test", output_dir: str='cifar10.h5'):
    # initialize model
    model = Model(max_oct_ratio=max_oct_ratio, depth_decay=1/18, mode=oct_mode) # Octave Model
    model.is_compressing = True

    # set up compressing optimizer
    weights_optimizer = K.optimizers.SGD(learning_rate=lr, momentum=0.9)
    optimizer = FineTuneOptimizer(model, weights_optimizer)
    
    # compile model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(model_dir)

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

    # restore checkpoint
    model.evaluate(x_test, y_test, batch_size=batch_size)

    # initialize compression optimizer
    compression_optimizer = OctaveCompressionOptimizer(compression_lambda, model, frequency_multiplier=frequency_multiplier, optimize_iterations=bayesian_steps)

    # training callbacks
    ratio_dir = os.path.join(root_dir, experiment_name, 'compress', 'ratio_' + str(max_oct_ratio) + '_' + str(oct_mode))
    tensorboard_dir = os.path.join(ratio_dir, 'logs')
    last_ckpt_dir = os.path.join(ratio_dir, "ckpt", "last")
    best_ckpt_dir = os.path.join(ratio_dir, "ckpt", "best")
    tensorboard_callback = K.callbacks.TensorBoard(log_dir=tensorboard_dir)
    checkpoint_callback = K.callbacks.ModelCheckpoint(filepath=last_ckpt_dir, save_weights_only=True)
    best_checkpoint_callback = K.callbacks.ModelCheckpoint(filepath=best_ckpt_dir, save_weights_only=True, save_best_only=True)
    callbacks = [tensorboard_callback, checkpoint_callback, best_checkpoint_callback]

    # initialize compression
    training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    compression_optimizer.minimize(training_dataset)

    # preprocessing generator
    data_generator = K.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=False, validation_split=0.1)

    # compress model
    model.fit(data_generator.flow(x_train, y_train, batch_size=batch_size, subset='training'), epochs=epochs, validation_data=data_generator.flow(x_train, y_train, batch_size=batch_size, subset='validation'), callbacks=callbacks)

    # load best model
    model.load_weights(best_ckpt_dir)
    model.save_weights(output_dir)
    model.evaluate(x_test, y_test, batch_size=batch_size)

    # evaluate size
    uncompressed_size, compressed_size = model.evaluate_size()
    print("Uncompressed size: %d" % uncompressed_size)
    print("Compressed size: %d" % compressed_size)    
