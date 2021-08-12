# import system module
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

# import tensorflow module
import tensorflow as tf
from tensorflow import keras as K

# import core module
from lib.layers import CompressibleConv2D
from lib.compression import CompressibleLayer, LayerCompressing

# enum padding
class Padding(Enum):
    VALID = 'VALID'
    SAME = 'SAME'

# enum octave type
class OctaveType(Enum):
    FIRST = 'FIRST'
    NORMAL = 'NORMAL'
    LAST = 'LAST'

# octave block
class OctaveLayer(K.layers.Layer, LayerCompressing):
    '''
    Main octave block
    '''
    # properties
    _oct_type: OctaveType
    _frequency_blocks: Dict[str, Optional[CompressibleLayer]]
    _regularizer: Callable
    input_oct_ratio: float
    output_oct_ratio: float

    @property
    def oct_type(self) -> OctaveType:
        return self._oct_type

    @property
    def oct_ratio(self) -> float:
        return self.output_oct_ratio

    @property
    def frequency_blocks(self) -> Dict[str, Optional[CompressibleLayer]]:
        return self._frequency_blocks

    # constructor
    def __init__(self, *args, input_oct_ratio: float=0, output_oct_ratio: float=0, oct_type: str='NORMAL', regularizer=None, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        '''
        input_oct_ratio - a float for the input oct ratio
        output_oct_ratio - a float for the output oct ratio
        oct_type - a string for the type of oct block
        '''
        # call super constructor
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic)

        # initialize variables
        self.input_oct_ratio = input_oct_ratio  
        self.output_oct_ratio = output_oct_ratio
        self._oct_type = OctaveType[oct_type]
        self._frequency_blocks = dict()
        self._regularizer = regularizer
        
    # get config
    def get_config(self):
        return dict()

    # call layer method
    def call(self, input_data: Any) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:
        # run hf layer
        if self._frequency_blocks['hf'] is not None:
            hf_data = self._frequency_blocks['hf'](input_data)
        else:
            hf_data = None

        # run lf layer
        if self._frequency_blocks['lf'] is not None:
            lf_data = self._frequency_blocks['lf'](input_data)
        else:
            lf_data = None

        # combine data
        return hf_data, lf_data
        
    # compress method
    def compress(self, parameters: tuple=None):
        # unpack parameters
        if parameters is not None:
            (hf_parameters, lf_parameters) = parameters
        else:
            hf_parameters, lf_parameters = None, None
        
        # compress hf block
        if self._frequency_blocks['hf'] is not None:
            self._frequency_blocks['hf'].compress(hf_parameters)
            
        # compress lf block
        if self._frequency_blocks['lf'] is not None:
            self._frequency_blocks['lf'].compress(lf_parameters)
            
    # evaluate size
    def evaluate_size(self) -> int:
        # initialize total size
        total_size = 0
        
        # get size for hf layer
        if self._frequency_blocks['hf'] is not None:
            total_size += self._frequency_blocks['hf'].evaluate_size()
        
        # get size of lf layer
        if self._frequency_blocks['lf'] is not None:
            total_size += self._frequency_blocks['lf'].evaluate_size()
        return total_size
            
    # set parameters
    def set_parameters(self, parameters: tuple):
        # unpack parameters
        (hf_parameters, lf_parameters) = parameters
        
        # compress hf block
        if self._frequency_blocks['hf'] is not None:
            self._frequency_blocks['hf'].set_parameters(hf_parameters)
            
        # compress lf block
        if self._frequency_blocks['lf'] is not None:
            self._frequency_blocks['lf'].set_parameters(lf_parameters)
            
    # reset weights
    def reset_weights(self, is_compressed: bool=False):
        # reset hf block weights
        if self._frequency_blocks['hf'] is not None:
            self._frequency_blocks['hf'].reset_weights(is_compressed)
        
        # reset lf block weights
        if self._frequency_blocks['lf'] is not None:
            self._frequency_blocks['lf'].reset_weights(is_compressed)

    # print compression summary
    def print_summary(self):
        # check frequency blocks
        if self._frequency_blocks['hf'] is not None and self._frequency_blocks['lf'] is not None:
            print("Layer %s:" % self.name, "hf=(%f, %d)" % (self._frequency_blocks['hf'].pruning_ratio, self._frequency_blocks['hf'].quantization_bit), "lf=(%f, %d)" % (self._frequency_blocks['lf'].pruning_ratio, self._frequency_blocks['lf'].quantization_bit))
        elif self._frequency_blocks['hf'] is not None:
            print("Layer %s:" % self.name, "hf=(%f, %d)" % (self._frequency_blocks['hf'].pruning_ratio, self._frequency_blocks['hf'].quantization_bit))
        elif self._frequency_blocks['lf'] is not None:
            print("Layer %s:" % self.name, "lf=(%f, %d)" % (self._frequency_blocks['lf'].pruning_ratio, self._frequency_blocks['lf'].quantization_bit))

# octave block
class OctaveConv2D(K.layers.Conv2D, OctaveLayer):
    '''
    Main octave block
    '''
    # properties
    _oct_type: OctaveType
    _frequency_blocks: Dict[str, Optional[CompressibleLayer]]
    _regularizer: Callable
    input_oct_ratio: float
    output_oct_ratio: float

    @property
    def oct_type(self) -> OctaveType:
        return self._oct_type

    @property
    def oct_ratio(self) -> float:
        return self.output_oct_ratio

    @property
    def frequency_blocks(self) -> Dict[str, Optional[CompressibleLayer]]:
        return self._frequency_blocks

    # constructor
    def __init__(self, *args, **kwargs):
        # call super constructor
        super().__init__(self, *args, **kwargs)
        
    # get config
    def get_config(self):
        return dict()

    # build
    def build(self, input_shape: tuple=None):        
        # calculate octave ratio
        alpha_out = self.output_oct_ratio

        # calculate dims
        hf_output_dim = int(self.filters * (1 - alpha_out)) if self._oct_type != OctaveType.LAST else self.filters
        lf_output_dim = self.filters - hf_output_dim

        # hf block
        if hf_output_dim > 0:
            self._frequency_blocks['hf'] = HFBlock(hf_output_dim, self.kernel_size, oct_type=self._oct_type, strides=self.strides, padding=self.padding, kernel_regularizer=self.kernel_regularizer, activation=self.activation, name='hf')
        else:
            self._frequency_blocks['hf'] = None
        
        # lf block
        if lf_output_dim > 0:
            self._frequency_blocks['lf'] = LFBlock(lf_output_dim, self.kernel_size, oct_type=self._oct_type, strides=self.strides, padding=self.padding, kernel_regularizer=self.kernel_regularizer, activation=self.activation, name='lf')
        else:
            self._frequency_blocks['lf'] = None

    # call layer method
    def call(self, input_data: Any) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:
        # run hf layer
        if self._frequency_blocks['hf'] is not None:
            hf_data = self._frequency_blocks['hf'](input_data)
        else:
            hf_data = None

        # run lf layer
        if self._frequency_blocks['lf'] is not None:
            lf_data = self._frequency_blocks['lf'](input_data)
        else:
            lf_data = None

        # combine data
        return hf_data, lf_data
        
    # compress method
    def compress(self, parameters: tuple=None):
        # unpack parameters
        if parameters is not None:
            (hf_parameters, lf_parameters) = parameters
        else:
            hf_parameters, lf_parameters = None, None
        
        # compress hf block
        if self._frequency_blocks['hf'] is not None:
            self._frequency_blocks['hf'].compress(hf_parameters)
            
        # compress lf block
        if self._frequency_blocks['lf'] is not None:
            self._frequency_blocks['lf'].compress(lf_parameters)
            
    # evaluate size
    def evaluate_size(self) -> int:
        # initialize total size
        total_size = 0
        
        # get size for hf layer
        if self._frequency_blocks['hf'] is not None:
            total_size += self._frequency_blocks['hf'].evaluate_size()
        
        # get size of lf layer
        if self._frequency_blocks['lf'] is not None:
            total_size += self._frequency_blocks['lf'].evaluate_size()
        return total_size
            
    # set parameters
    def set_parameters(self, parameters: tuple):
        # unpack parameters
        (hf_parameters, lf_parameters) = parameters
        
        # compress hf block
        if self._frequency_blocks['hf'] is not None:
            self._frequency_blocks['hf'].set_parameters(hf_parameters)
            
        # compress lf block
        if self._frequency_blocks['lf'] is not None:
            self._frequency_blocks['lf'].set_parameters(lf_parameters)
            
    # reset weights
    def reset_weights(self, is_compressed: bool=False):
        # reset hf block weights
        if self._frequency_blocks['hf'] is not None:
            self._frequency_blocks['hf'].reset_weights(is_compressed)
        
        # reset lf block weights
        if self._frequency_blocks['lf'] is not None:
            self._frequency_blocks['lf'].reset_weights(is_compressed)

    # print compression summary
    def print_summary(self):
        # check frequency blocks
        if self._frequency_blocks['hf'] is not None and self._frequency_blocks['lf'] is not None:
            print("Layer %s:" % self.name, "hf=(%f, %d)" % (self._frequency_blocks['hf'].pruning_ratio, self._frequency_blocks['hf'].quantization_bit), "lf=(%f, %d)" % (self._frequency_blocks['lf'].pruning_ratio, self._frequency_blocks['lf'].quantization_bit))
        elif self._frequency_blocks['hf'] is not None:
            print("Layer %s:" % self.name, "hf=(%f, %d)" % (self._frequency_blocks['hf'].pruning_ratio, self._frequency_blocks['hf'].quantization_bit))
        elif self._frequency_blocks['lf'] is not None:
            print("Layer %s:" % self.name, "lf=(%f, %d)" % (self._frequency_blocks['lf'].pruning_ratio, self._frequency_blocks['lf'].quantization_bit))

# frequency block
class FrequencyBlock(K.layers.Conv2D, CompressibleLayer):
    '''
    Frequency block in octave block
    extends ResidualBlock
    implements Compressing
    Parameters:
        oct_type - an octave type of octave type
        residual_blocks - a dictionary of residual blocks
    '''
    # properties
    _oct_type: OctaveType
    residual_blocks: Dict[str, Optional[K.layers.Layer]]
    batchnorm_layers: Dict[str, Optional[K.layers.BatchNormalization]]
    activation_layers: Dict[str, Optional[K.layers.ReLU]]

    # constructor
    def __init__(self, oct_type: OctaveType=OctaveType.NORMAL, regularizer=None, **kwargs):
        # call super constructor
        super().__init__(**kwargs)
        
        # initialize variables
        self._oct_type = oct_type
        self.residual_blocks = dict()
        self.batchnorm_layers = dict()
        self.activation_layers = dict()
        self._regularizer = regularizer

    # get config
    def get_config(self) -> dict:
        return dict()

    # build layer
    def build(self, input_shape: tuple):
        # switch octave type
        if self._oct_type == OctaveType.NORMAL:
            # normal oct type
            self._conv_blocks['hf_conv'] = K.layers.Conv2D(self.filters, self.kernel_size, kernel_regularizer=self.kernel_regularizer, padding=self.padding.value, name='hf_conv')
            
            self._conv_blocks['lf_conv'] = K.layers.Conv2D(self.filters, self.kernel_size, kernel_regularizer=self.kernel_regularizer, padding=self.padding.value, name='lf_conv')
        elif self._oct_type == OctaveType.FIRST:
            # first oct type
            self._conv_blocks['hf_conv'] = K.layers.Conv2D(self.filters, self.kernel_size, kernel_regularizer=self.kernel_regularizer, padding=self.padding.value, name='hf_conv')
            self._conv_blocks['lf_conv'] = None
        elif self._oct_type == OctaveType.LAST:
            # last oct type
            self._conv_blocks['hf_conv'] = K.layers.Conv2D(self.filters, self.kernel_size, kernel_regularizer=self.kernel_regularizer, padding=self.padding.value, name='hf_conv')
            self._conv_blocks['lf_conv'] = K.layers.Conv2D(self.filters, self.kernel_size, kernel_regularizer=self.kernel_regularizer, padding=self.padding.value, name='lf_conv')
        else:
            # unknown type
            raise TypeError('Unknown octave type %s' % self._oct_type)
            
    # call layer
    def call(self, input_data: Any) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:
        # unpack input data
        hf_data = input_data[0]
        lf_data = input_data[1]

        # hf convolution
        if hf_data is not None:
            hf_data = self.residual_blocks['hf_conv'](hf_data)

            # batch norm
            if self.batchnorm_layers['hf'] is not None:
                hf_data = self.batchnorm_layers['hf'](hf_data)

            # activation
            if self.activation_layers['hf'] is not None:
                hf_data = self.activation_layers['hf'](hf_data)
        
        # lf convolution
        if lf_data is not None:
            lf_data = self.residual_blocks['lf_conv'](lf_data)

            # batch norm
            if self.batchnorm_layers['lf'] is not None:
                lf_data = self.batchnorm_layers['lf'](lf_data)

            # activation
            if self.activation_layers['lf'] is not None:
                lf_data = self.activation_layers['lf'](lf_data)
        
        # return data
        return hf_data, lf_data

# high frequency block
class HFBlock(FrequencyBlock):
    '''
    High Frequency Block
    extends FrequencyBlock
    '''
    # call layer
    def call(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        # unpack input data
        hf_data, lf_data = input_data

        # pooling for strides 2
        if self.strides == 2 and hf_data is not None:
            hf_data = tf.nn.avg_pool(hf_data, ksize=[1,2,2,1], strides=(2,2), padding='SAME', name='data_pool')

        # call super layer
        hf_data, lf_data = super().call([hf_data, lf_data])

        # upsampling lf data
        if self.strides != 2 and lf_data is not None:
            lf_data = K.layers.UpSampling2D()(lf_data)

        # check frequency feature maps
        if hf_data is not None and lf_data is not None:
            # add
            hf_data += lf_data
        elif lf_data is not None:
            hf_data = lf_data

        # return high frequency data
        return hf_data

# low frequency block
class LFBlock(FrequencyBlock):
    '''
    Low Frequency Block
    extends FrequencyBlock
    '''
    # call layer
    def call(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        # unpack input data
        hf_data, lf_data = input_data
        
        # pooling for strides = 2
        if self.strides == 2:
            # pool hf data
            if hf_data is not None:
                hf_data = tf.nn.avg_pool(hf_data, ksize=[1,2,2,1], strides=(2,2), padding=self.padding, name='data_pool')
            
            # pool lf data
            if lf_data is not None:
                lf_data = tf.nn.avg_pool(lf_data, ksize=[1,2,2,1], strides=(2,2), padding=self.padding, name='data_pool')

        # high frequency to low frequency pooling
        if hf_data is not None:
            hf_data = tf.nn.avg_pool(hf_data, ksize=[1,2,2,1], strides=(2,2), padding=self.padding, name='lf_pool')

        # call super layer
        hf_data, lf_data = super().call([hf_data, lf_data])

        # add
        if hf_data is not None and lf_data is not None:
            lf_data += hf_data
        elif hf_data is not None:
            lf_data = hf_data

        # return high frequency data
        return lf_data

# octave maxpool
class OctaveMaxPool2D(K.layers.Layer):
    # properties
    hf_pool: K.layers.MaxPool2D
    lf_pool: K.layers.MaxPool2D
    _pool_size: Tuple[int, int]
    _strides: Optional[Tuple[int, int]]
    _padding: Padding

    @property
    def pool_size(self) -> Tuple[int, int]:
        return self._pool_size

    @property
    def strides(self) -> Optional[Tuple[int, int]]:
        return self._strides

    @property
    def padding(self) -> str:
        return self._padding.value

    # constructor
    def __init__(self, pool_size: Tuple[int, int]=(2,2), strides: Optional[Tuple[int, int]]=None, padding: Padding=Padding.VALID, **kwargs):
        super().__init__(**kwargs)

        # initialize properties
        self._pool_size = pool_size
        self._strides = strides
        self._padding = padding

    # build method
    def build(self, input_shape: tuple):
        self.hf_pool = K.layers.MaxPool2D(self.pool_size, strides=self.strides, padding=self.padding, name='hf_pool')
        self.lf_pool = K.layers.MaxPool2D(self.pool_size, strides=self.strides, padding=self.padding, name='lf_pool')

    # call method
    def call(self, input_data: Any) -> Tuple[tf.Tensor, tf.Tensor]:
        # unpack input data
        hf_data = input_data[0]
        lf_data = input_data[1]

        # hf maxpool
        if hf_data is not None:
            hf_data = self.hf_pool(hf_data)

        # lf maxpool
        if lf_data is not None:
            lf_data = self.lf_pool(lf_data)

        # return data
        return hf_data, lf_data

# Octave batchnorm
class OctaveBatchNormalization(K.layers.Layer):
    # properties
    hf_batchnorm: K.layers.BatchNormalization
    lf_batchnorm: K.layers.BatchNormalization

    # build
    def build(self, input_shape: tuple):
        self.hf_batchnorm = K.layers.BatchNormalization(name='hf')
        self.lf_batchnorm = K.layers.BatchNormalization(name='lf')

    # call
    def call(self, input_data: Any) -> Any:
        # unpack input data
        hf_data = input_data[0]
        lf_data = input_data[1]

        # hf batchnorm
        if hf_data is not None:
            hf_data = self.hf_batchnorm(hf_data)

        # lf batchnorm
        if lf_data is not None:
            lf_data = self.lf_batchnorm(lf_data)

        # return data
        return hf_data, lf_data

# Octave relu
class OctaveRelu(K.layers.Layer):
    # properties
    hf_relu: K.layers.ReLU
    lf_relu: K.layers.ReLU

    # build
    def build(self, input_shape: tuple):
        self.hf_relu = K.layers.ReLU(name='hf')
        self.lf_relu = K.layers.ReLU(name='lf')

    # call
    def call(self, input_data: Any) -> Any:
        # unpack input data
        hf_data = input_data[0]
        lf_data = input_data[1]

        # hf relu
        if hf_data is not None:
            hf_data = self.hf_relu(hf_data)

        # lf relu
        if lf_data is not None:
            lf_data = self.lf_relu(lf_data)

        # return data
        return hf_data, lf_data