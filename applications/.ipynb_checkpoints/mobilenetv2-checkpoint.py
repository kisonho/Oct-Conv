# import system modules
from typing import Tuple

# import tensorflow
import tensorflow as tf
from tensorflow import keras as K

# import core modules
from lib.model import OctaveModel
from lib.oct_layers import OctaveSeparableResBlock

# octave model for mobilenet v2
class OctMobileNetV2(OctaveModel):
    '''
    Octave model for mobilenet v2
    extends CompressibleModel
    '''
    # build a single octave residual block
    def build_single_oct_residual_block(self, output_dim: int, expansion_ratio: float=1, strides: int=1, padding: str='SAME', oct_ratio: Tuple[float, float]=None, oct_type='NORMAL', name=None):
        # get input alpha
        alpha_in = self._oct_ratio
        
        # calculate octave ratio
        if oct_ratio is None:
            alpha_out = self.calculate_oct_ratio()
        else:
            [_, alpha_out] = oct_ratio

        # octave block
        oct_block = OctaveSeparableResBlock(output_dim=output_dim, input_oct_ratio=alpha_in, output_oct_ratio=alpha_out, oct_type=oct_type, expansion_ratio=expansion_ratio, strides=strides, padding=padding, regularizer=self._regularizer, name=name)
        self.oct_layers.append(oct_block)
        self._compressible_oct_layers.append(oct_block)

    # build octave residual blocks with given repeated times
    def build_oct_residual_block(self, output_dim: int, expansion_ratio: float=1, strides: int=1, padding: str='SAME', oct_ratio: Tuple[float, float]=None, repeat: int=1, name: str=None):  
        # repeat blocks
        for i in range(0,repeat):
            # name with index
            name_with_index = name + "_block" + str(i+1)

            # get input alpha
            alpha_in = self._oct_ratio
            
            # calculate octave ratio
            if oct_ratio is None:
                alpha_out = self.calculate_oct_ratio()
            else:
                [_, alpha_out] = oct_ratio
            
            # octave block
            oct_block = OctaveSeparableResBlock(output_dim=output_dim, input_oct_ratio=alpha_in, output_oct_ratio=alpha_out, expansion_ratio=expansion_ratio, strides=strides, padding=padding, regularizer=self._regularizer, name=name_with_index)
            self.oct_layers.append(oct_block)
            self._compressible_oct_layers.append(oct_block)

            # reset strides
            strides = 1

# image net classficiation model
class Cifar10OctMobileNetV2(OctMobileNetV2):
    # constructor
    def __init__(self, **kwargs):
        # call super init
        super().__init__(**kwargs)

    # build model
    def build(self, input_shape: Tuple[int, int, int]=(32,32,3)):
        # first octave block
        self.build_single_oct_residual_block(32, padding='SAME', oct_type='FIRST', name='conv1')

        # octave blocks
        self.build_oct_residual_block(16, name='bn2')
        self.build_oct_residual_block(24, expansion_ratio=6, repeat=2, name='bn3')
        self.build_oct_residual_block(32, expansion_ratio=6, strides=2, repeat=3, name='bn4')
        self.build_oct_residual_block(64, expansion_ratio=6, strides=2, repeat=4, name='bn5')
        self.build_oct_residual_block(96, expansion_ratio=6, repeat=3, name='bn6')
        self.build_oct_residual_block(160, expansion_ratio=6, strides=2, repeat=3, name='bn7')
        self.build_oct_residual_block(320, expansion_ratio=6, name='bn8')

        # last octave block
        self.build_single_oct_residual_block(320, 1280, oct_type='LAST', name='pw9')

        # define classfication layer
        self.classification_layer = K.layers.Conv2D(10, kernel_size=(1,1), kernel_regularizer=self._regularizer, padding='SAME', name='classification')

        # print oct ratio
        print(self.oct_ratio_list)

    # call model
    def call(self, input_data: tf.Tensor) -> tf.Tensor:
        # call oct model
        x = super().call(input_data)

        # average pool 10
        x = tf.keras.layers.AveragePooling2D((4,4), (1,1), name='avgpool10')(x)

        # classification scope
        x = self.classification_layer(x)
        x = tf.squeeze(x, [1,2], name='squeeze')
        y = tf.nn.softmax(x)
        return y

# image net classficiation model
class ImageNetOctMobileNetV2(OctMobileNetV2):
    # constructor
    def __init__(self, **kwargs):
        # call super init
        super().__init__(**kwargs)

        # classification layer
        self.classfication_layer = None
        self.regularizer = K.regularizers.l2(4e-5)

    # build model method
    def build(self, input_shape: Tuple[int, int, int]=(224,224,3)):
        # first octave block
        self.build_single_oct_residual_block(32, padding='SAME', strides=2, oct_type='FIRST', name='conv1')

        # octave blocks
        self.build_oct_residual_block(16, name='bn2')
        self.build_oct_residual_block(24, expansion_ratio=6, strides=2, repeat=2, name='bn3')
        self.build_oct_residual_block(32, expansion_ratio=6, strides=2, repeat=3, name='bn4')
        self.build_oct_residual_block(64, expansion_ratio=6, strides=2, repeat=4, name='bn5')
        self.build_oct_residual_block(96, expansion_ratio=6, repeat=3, name='bn6')
        self.build_oct_residual_block(160, expansion_ratio=6, strides=2, repeat=3, name='bn7')
        self.build_oct_residual_block(320, expansion_ratio=6, name='bn8')

        # last octave block
        self.build_single_oct_residual_block(1280, oct_type='LAST', name='pw9_conv')

        # define classfication layer
        self.classification_layer = K.layers.Conv2D(1000, kernel_size=(1,1), kernel_regularizer=self.regularizer, padding='SAME', name='classification')

        # print oct ratio
        print(self.oct_ratio_list)

    # call model
    def call(self, input_data: tf.Tensor) -> tf.Tensor:
        # call oct model
        x = super().call(input_data)

        # average pool 10
        x = tf.keras.layers.AveragePooling2D(x.get_shape()[1:-1], (1,1), name='avgpool10')(x)

        # classification scope
        x = self.classification_layer(x)
        x = tf.squeeze(x, [1,2], name='squeeze')
        y = tf.nn.softmax(x)
        return y