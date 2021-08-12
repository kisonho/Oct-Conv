# import typing
from typing import *
from enum import Enum

# import tensorflow
import tensorflow as tf
from tensorflow import keras as K

# import core module
from .compression import WeightsCompressing, CompressibleLayer

# residual block layer
class SeparableResBlock(K.layers.Layer):
    '''
    A Residual Block of several layers
    extends K.layers.Layer
    parameters:
        output_dim - an int for the output dim of the block
        expansion_ratio - a float for the expansion ratio of the bottleneck
        strides - an int for the strides of the block
        padding - a string of padding type of the block
        is_shortcut - a boolean to control if the block contains shortcut
        regularizer - a regularizer of the layer kernel regularizer
        convolutional_layers - a dictionary of convolutional layers
    '''
    # properties
    _output_dim: Optional[int]
    _expansion_ratio: float
    _strides: int
    _is_shortcut: bool
    _padding: str
    _regularizer: Optional[K.regularizers.Regularizer]
    convolutional_layers: Dict[str, K.layers.Layer]

    @property
    def output_dim(self) -> Optional[int]:
        return self._output_dim

    @property
    def expansion_ratio(self) -> float:
        return self._expansion_ratio

    @property
    def strides(self) -> int:
        return self._strides

    @property
    def is_shortcut(self) -> bool:
        return self._is_shortcut

    @property
    def padding(self) -> str:
        return self._padding

    @property
    def regularizer(self) -> Optional[K.regularizers.Regularizer]:
        return self._regularizer
    
    # constructor
    def __init__(self, output_dim: Optional[int]=None, expansion_ratio: float=1, strides: int=1, is_shortcut: bool=True, padding: str='SAME', regularizer: Optional[K.regularizers.Regularizer]=None, trainable: bool=True, name: Optional[str]=None, dtype: tf.DType=None, dynamic: bool=False, **kwargs):
        # call super
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic)

        # set parameters
        self._output_dim = output_dim
        self._expansion_ratio = expansion_ratio
        self._strides = strides
        self._is_shortcut = is_shortcut
        self._padding = padding
        self._regularizer = regularizer
        self.convolutional_layers = dict()

    # get config
    def get_config(self):
        return dict()

# residual block for mobilenet v2
class CompressibleResidualBlock(SeparableResBlock, WeightsCompressing):
    '''
    The Residual Block of MobileNetV2
    extends ResidualBlock
    parameters:
        input_dim - an int for the input dim of the block
        convolutional_layers - a list of convolutional layers in current block
    '''
    # constructor
    def __init__(self, *args, **kwargs):
        # call super constructor
        super().__init__(*args, **kwargs)
        
        # initialize parameters
        self.input_dim = 0
        self.convolutional_layers = dict()
        
    # build layer
    def build(self, input_shape):
        # set up input dim
        self.input_dim = input_shape[-1]
        bottleneck_dim = round(self.expansion_ratio * self.input_dim)
        
        # pw1_conv
        self.convolutional_layers['pw1_conv'] = K.layers.Conv2D(bottleneck_dim, (1,1), kernel_regularizer=self.regularizer, padding='SAME', name='pw1_conv')
        self.convolutional_layers['pw1_batchnorm'] = K.layers.BatchNormalization(name='pw1_batchnorm')
        self.convolutional_layers['dw2_conv'] = K.layers.DepthwiseConv2D(kernel_size=(3,3), kernel_regularizer=self.regularizer, strides=(self.strides, self.strides), padding=self.padding, name='dw2_conv')
        self.convolutional_layers['dw2_batchnorm'] = K.layers.BatchNormalization(name='dw2_batchnorm')
        self.convolutional_layers['pw3_conv'] = K.layers.Conv2D(self.output_dim, (1,1), kernel_regularizer=self.regularizer, padding='SAME', name='pw3_conv')
        self.convolutional_layers['pw3_batchnorm'] = K.layers.BatchNormalization(name='pw3_batchnorm')

        # call super build
        super().build(input_shape)

    # call function
    def call(self, input_data: tf.Tensor) -> tf.Tensor:
        # point wise convolution
        block = self.convolutional_layers['pw1_conv'](input_data)
        block = self.convolutional_layers['pw1_batchnorm'](block)
        block = tf.nn.relu6(block, name='pw1_relu')

        # depthwise convolution
        block = self.convolutional_layers['dw2_conv'](block)
        block = self.convolutional_layers['dw2_batchnorm'](block)
        block = tf.nn.relu6(block, name='dw2_relu')

        # linear pointwise convolution
        block = self.convolutional_layers['pw3_conv'](block)
        block = self.convolutional_layers['pw3_batchnorm'](block)

        # element wise add
        if self.is_shortcut is True and self.strides == 1 and self.input_dim == self.output_dim:
            # extra pointwise layer to output_dim when not equal
            block += input_data
        
        # set input to block
        output_data = block
        return output_data

# Compressible Dense
class CompressibleDense(CompressibleLayer):
    # properties
    dense: K.layers.Dense

    # constructor
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name=name)
        self.dense = K.layers.Dense(*args, name=name, **kwargs)

    # call method
    def call(self, input_data: Any) -> tf.Tensor:
        return self.dense(input_data)

# Compressible Conv2D
class CompressibleConv2D(CompressibleLayer):
    # properties
    conv: K.layers.Conv2D

    # constructor
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name=name)
        self.conv = K.layers.Conv2D(*args, name=name, **kwargs)

    # call method
    def call(self, input_data: Any) -> tf.Tensor:
        return self.conv(input_data)
