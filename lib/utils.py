from typing import Tuple, Optional
import numpy as np

# import tensorflow
import tensorflow as tf
from tensorflow import keras as K

# class data preprocessor
class ImagePreprocessor(object):
    # properties
    _input_shape: Tuple[int, int, int]
    _is_crop_enabled: bool
    _is_random_flip: bool
    _is_random_crop: bool
    _mu: tf.Tensor

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    # constructor
    def __init__(self, input_shape: Tuple[int, int, int], is_crop_enabled: bool=True, is_random_flip: bool=True, is_random_crop: bool=True):
        # initialize input shape
        self._input_shape = input_shape
        self._is_crop_enabled = is_crop_enabled
        self._is_random_flip = is_random_flip
        self._is_random_crop = is_random_crop
        self._mu = tf.constant(np.load("mean.npy").mean(1).mean(1))
        
    # preprocessing imagenet v2 dataset
    def preprocess(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # convert to tensor
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)
        
        # cast images to float
        image = tf.cast(image, tf.float32)

        # preprocess
        image -= self._mu
        image /= 256
        
        # random flip
        if self._is_random_flip is True:
            image = tf.image.random_flip_left_right(image)
        
        # check crop enabled
        if self._is_crop_enabled is True:
            assert self._input_shape[0] is not None and self._input_shape[1] is not None
            image = self.scale_image(image)
            
            # crop
            if self._is_random_crop is True:
                image = tf.image.random_crop(image, self._input_shape)
            else:
                image = tf.image.resize_with_crop_or_pad(image, self._input_shape[0], self._input_shape[1])
            
        # set shape
        image.set_shape(self.input_shape)
        return image, label
    
    # scale image
    def scale_image(self, image: tf.Tensor) -> tf.Tensor:
        # get image shape
        image_shape: tf.Tensor = tf.shape(image)
        size: tf.Tensor = tf.random.uniform([], minval=256, maxval=481, dtype=tf.int32)
        
        # scale image
        scale: tf.Tensor = tf.cond(tf.greater(image_shape[0], image_shape[1]), lambda: size / image_shape[1], lambda: size / image_shape[0])
        h = tf.cast(tf.cast(image_shape[0], tf.float64) * scale, tf.int32)
        w = tf.cast(tf.cast(image_shape[1], tf.float64) * scale, tf.int32)
        return tf.image.resize(image, (h, w))