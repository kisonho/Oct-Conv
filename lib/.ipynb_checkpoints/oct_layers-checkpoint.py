# import system module
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

# import tensorflow module
import tensorflow as tf
from tensorflow import keras as K

# import core module
from lib.layers import SeparableResBlock, CompressibleResidualBlock
from lib.compression import CompressibleLayer, LayerCompressing

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
class OctaveSeparableResBlock(SeparableResBlock, OctaveLayer):
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
        SeparableResBlock.__init__(self, *args, **kwargs)
        OctaveLayer.__init__(self, *args, **kwargs)
        
    # get config
    def get_config(self):
        return dict()

    # build
    def build(self, input_dim: tuple=None):
        # initialize input dim
        self.input_dim = input_dim
        
        # calculate octave ratio
        alpha_out = self.output_oct_ratio

        # calculate dims
        hf_output_dim = int(self.output_dim * (1 - alpha_out)) if self._oct_type != OctaveType.LAST else self.output_dim
        lf_output_dim = self.output_dim - hf_output_dim

        # hf block
        if hf_output_dim > 0:
            self._frequency_blocks['hf'] = HFSeparableResBlock(oct_type=self._oct_type, output_dim=hf_output_dim, strides=self.strides, padding=self.padding, is_shortcut=self.is_shortcut, regularizer=self._regularizer, name='hf')
        else:
            self._frequency_blocks['hf'] = None
        
        # lf block
        if lf_output_dim > 0:
            self._frequency_blocks['lf'] = LFSeparableResBlock(oct_type=self._oct_type, output_dim=lf_output_dim, strides=self.strides, padding=self.padding, is_shortcut=self.is_shortcut, regularizer=self._regularizer, name='lf')
        else:
            self._frequency_blocks['lf'] = None

        # print details
        print('Layer %s: alpha=%s' % (self.name, self.oct_ratio))

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
class FrequencyBlock(SeparableResBlock, CompressibleLayer):
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

# high frequency residual block for mobilenet v2
class HFSeparableResBlock(HFBlock):
    # build layer
    def build(self, input_dim: tuple):
        # switch octave type
        if self._oct_type == OctaveType.NORMAL:
            # normal oct type
            self.residual_blocks['hf_conv'] = CompressibleResidualBlock(self.output_dim, expansion_ratio=self.expansion_ratio, regularizer=self.regularizer, is_shortcut=self.is_shortcut, name='hf_conv')
            self.batchnorm_layers['hf'] = None
            self.activation_layers['hf'] = None
            self.residual_blocks['lf_conv'] = CompressibleResidualBlock(self.output_dim, expansion_ratio=self.expansion_ratio, regularizer=self.regularizer, is_shortcut=self.is_shortcut, name='lf_conv')
            self.batchnorm_layers['lf'] = None
            self.activation_layers['lf'] = None
        elif self._oct_type == OctaveType.FIRST:
            # first oct type
            self.residual_blocks['hf_conv'] = K.layers.Conv2D(self.output_dim, (3,3), kernel_regularizer=self.regularizer, padding=self.padding, name='hf_conv')
            self.batchnorm_layers['hf'] = K.layers.BatchNormalization(name='hf_batchnorm')
            self.activation_layers['hf'] =K.layers.ReLU(name='hf_relu')
            self.residual_blocks['lf_conv'] = None
            self.batchnorm_layers['lf'] = None
            self.activation_layers['lf'] = None
        elif self._oct_type == OctaveType.LAST:
            # last oct type
            self.residual_blocks['hf_conv'] = K.layers.Conv2D(self.output_dim, (1,1), kernel_regularizer=self.regularizer, padding=self.padding, name='hf_conv')
            self.batchnorm_layers['hf'] = K.layers.BatchNormalization(name='hf_batchnorm')
            self.activation_layers['hf'] =K.layers.ReLU(name='hf_relu')
            self.residual_blocks['lf_conv'] = K.layers.Conv2D(self.output_dim, (1,1), kernel_regularizer=self.regularizer, padding=self.padding,  name='lf_conv')
            self.batchnorm_layers['lf'] = K.layers.BatchNormalization(name='lf_batchnorm')
            self.activation_layers['lf'] =K.layers.ReLU(name='lf_relu')
        else:
            # unknown type
            raise TypeError('Unknown octave type %s' % self._oct_type)

    # call method
    def call(self, input_data: Any) -> tf.Tensor:
        return super().call(input_data)

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

# high frequency residual block for mobilenet v2
class LFSeparableResBlock(LFBlock):
    # build layer
    def build(self, input_dim: tuple):
        # switch octave type
        if self._oct_type == OctaveType.NORMAL:
            # normal oct type
            self.residual_blocks['hf_conv'] = CompressibleResidualBlock(self.output_dim, expansion_ratio=self.expansion_ratio, regularizer=self.regularizer, is_shortcut=self.is_shortcut, name='hf_conv')
            self.batchnorm_layers['hf'] = None
            self.activation_layers['hf'] = None
            self.residual_blocks['lf_conv'] = CompressibleResidualBlock(self.output_dim, expansion_ratio=self.expansion_ratio, regularizer=self.regularizer, is_shortcut=self.is_shortcut, name='lf_conv')
            self.batchnorm_layers['lf'] = None
            self.activation_layers['lf'] = None
        elif self._oct_type == OctaveType.FIRST:
            # first oct type
            self.residual_blocks['hf_conv'] = K.layers.Conv2D(self.output_dim, (3,3), kernel_regularizer=self.regularizer, padding=self.padding, name='hf_conv')
            self.batchnorm_layers['hf'] = K.layers.BatchNormalization(name='hf_batchnorm')
            self.activation_layers['hf'] =K.layers.ReLU(name='hf_relu')
            self.residual_blocks['lf_conv'] = None
            self.batchnorm_layers['lf'] = None
            self.activation_layers['lf'] = None
        elif self._oct_type == OctaveType.LAST:
            # last oct type
            self.residual_blocks['hf_conv'] = K.layers.Conv2D(self.output_dim, (1,1), kernel_regularizer=self.regularizer, padding=self.padding, name='hf_conv')
            self.batchnorm_layers['hf'] = K.layers.BatchNormalization(name='hf_batchnorm')
            self.activation_layers['hf'] =K.layers.ReLU(name='hf_relu')
            self.residual_blocks['lf_conv'] = K.layers.Conv2D(self.output_dim, (1,1), kernel_regularizer=self.regularizer, padding=self.padding,  name='lf_conv')
            self.batchnorm_layers['lf'] = K.layers.BatchNormalization(name='lf_batchnorm')
            self.activation_layers['lf'] =K.layers.ReLU(name='lf_relu')
        else:
            # unknown type
            raise TypeError('Unknown octave type %s' % self._oct_type)

    # call method
    def call(self, input_data: Any) -> tf.Tensor:
        return super().call(input_data)