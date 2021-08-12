from typing import List, Tuple, Protocol

# import tensorflow
import tensorflow as tf
from tensorflow import keras as K

# compression utils
class Utils(object):
    # flatten a non-square list to a tensor
    @staticmethod
    def flatten(list_to_flatten: list) -> tf.Tensor:
        # flatten weights
        flatten_list = list()

        # loop for each item
        for l in list_to_flatten:
            l = tf.reshape(l, (-1,))
            flatten_list.append(l)
        return tf.concat(flatten_list, axis=0)
    
    # calculate the clip c+ and c- with weights and pruning ratio
    @staticmethod
    def calculate_clip(weights: list, p: float) -> Tuple[float, float]:
        # flatten weights
        flatten_weights = Utils.flatten(weights)

        # get and sort positive and negative weights
        positive_mask = tf.greater_equal(flatten_weights, 0)
        positive_weight = tf.boolean_mask(flatten_weights, positive_mask)
        positive_weight = tf.sort(positive_weight)
        negative_mask = tf.less_equal(flatten_weights, 0)
        negative_weight = tf.boolean_mask(flatten_weights, negative_mask)
        negative_weight = tf.sort(negative_weight, direction='DESCENDING')

        # get clip for positive weights
        if positive_weight.shape[0] > 0:
            positive_clip_index = tf.minimum(tf.cast(positive_weight.shape[0] * p, tf.int32), positive_weight.shape[0] - 1)
            positive_clip = positive_weight[positive_clip_index]
        else:
            positive_clip_index = None
            positive_clip = 0.

        # get clip for positive weights
        if negative_weight.shape[0] > 0:
            negative_clip_index = tf.minimum(tf.cast(negative_weight.shape[0] * p, tf.int32), negative_weight.shape[0] - 1)
            negative_clip = negative_weight[negative_clip_index]
        else:
            negative_clip_index = None
            negative_clip = 0.
        return positive_clip, negative_clip

    # clip
    @staticmethod
    def clip(w: list, positive_clip: float, negative_clip: float) -> tf.Tensor:
        # calculate mask
        positive_mask = tf.cast(tf.greater_equal(w, positive_clip), tf.float32)
        negative_mask = tf.cast(tf.less_equal(w, negative_clip), tf.float32)
        mask = positive_mask + negative_mask

        # apply mask
        clipped_weights = w * mask
        return clipped_weights
        
    # fake quantize a weight but keep type to be float32
    # current tensorflow framework does not support for specific bits quantize
    @staticmethod
    def fake_quantize(weights: tf.Tensor, num_bits: int) -> tf.Tensor:
        # initialize quantization
        max_weight: float = tf.cast(tf.reduce_max(weights), tf.float32).numpy()
        min_weight: float = tf.cast(tf.reduce_min(weights), tf.float32).numpy()
        max_weight = max_weight if max_weight > min_weight else min_weight + 1

        # quantize function
        quantized_weights = tf.quantization.fake_quant_with_min_max_args(weights, max=max_weight, min=min_weight, num_bits=num_bits)
        return quantized_weights
        
    # real quantize weight
    @staticmethod
    def quantize(weights: list) -> list:
        # initialize quantization
        max_weight = float(tf.reduce_max(weights))
        min_weight = float(tf.reduce_min(weights))
        
        # initialize quantized weights
        quantized_weights = list()
        
        # loop weights
        for w in weights:
            w = tf.quantization.quantize(w, min_weight, max_weight, T=tf.int8)
            quantized_weights.append(w)
        
        # return quantized weight
        return quantized_weights

# layer compressing protocol
class LayerCompressing(Protocol):
    # compress method
    def compress(self, parameters: tuple=None):
        return NotImplemented

    # set compression parameters
    def set_parameters(self, parameters: tuple=None):
        return NotImplemented

    # evaluate size
    def evaluate_size(self) -> int:
        return NotImplemented

    # print summary
    def print_summary(self):
        return NotImplemented

# weights compressing protocol
class WeightsCompressing(Protocol):
    # clip
    def clip(self, weights: list, positive_clip: float, negative_clip: float) -> list:
        return NotImplemented

    # quantize
    def quantize(self, weights: list, b: int) -> list:
        return NotImplemented

    # get compressible weights
    def get_compressible_weights(self) -> list:
        return NotImplemented

    # set compressible weights
    def set_compressible_weights(self, weights: list):
        return NotImplemented

# compressible layer
class CompressibleLayer(K.layers.Layer, LayerCompressing):
    '''
    A compressible layer
    '''
    # properties
    _pruning_ratio: tf.Variable
    _quantization_bit: tf.Variable

    @property
    def pruning_ratio(self) -> tf.Variable:
        return self._pruning_ratio
        
    @property
    def quantization_bit(self) -> tf.Variable:
        return self._quantization_bit

    # constructor
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize compression parameters
        self._pruning_ratio = 0
        self._quantization_bit = 8
    
    # compress model
    def compress(self, parameters: Tuple[float, int]=None):
        # initialize parameters
        if parameters is not None:
            # when parameters are given, used in bayesian optimization
            (p, b) = parameters
        else:
            # when parameters are not given, used in fine-tune
            p = self.pruning_ratio
            b = self.quantization_bit
            
        # gpu accelerated method
        # initialize compressed weights
        weights: List[tf.Variable] = self.trainable_weights
        positive_clip, negative_clip = Utils.calculate_clip(weights, p)

        # compress each trainable weight
        for w in weights:
            compressed_weights: tf.Tensor = Utils.clip(w, positive_clip, negative_clip)
            compressed_weights = tf.py_function(Utils.fake_quantize, (compressed_weights, b), Tout=tf.float32, name='quantization')
            w.assign(compressed_weights)

    # set compression parameters
    def set_parameters(self, parameters: tuple):
        # unpack parameters
        (p, b) = parameters
        
        # update compression parameters
        self._pruning_ratio = p
        self._quantization_bit = b
    
    # evaluate the size cost by this layer
    # TensorFlow does not support pruning and fixed bits quantization so that this function is used to estimated the compression size
    def evaluate_size(self) -> int:
        # initialize weights
        weights = self.get_weights()
        weight_size = 0

        # calculate weight size
        for w in weights:
            weight_size += int(tf.cast(tf.size(w), tf.float64) * (1 - self.pruning_ratio))

        # calculate size for compressed weights
        size = weight_size * self.quantization_bit
        return size

    # print compression summary
    def print_summary(self):
        print("Layer %s:" % self.name, (self.pruning_ratio, self.quantization_bit))