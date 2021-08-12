# import typing
from typing import Any, Tuple

# import tensorflow
import tensorflow as tf
from tensorflow import keras as K

# import core module
from lib.model import OctaveModel
from lib.layers import CompressibleDense
from lib.oct_layers import Padding, OctaveConv2D, OctaveBatchNormalization, OctaveRelu, OctaveMaxPool2D

# image net classficiation model
class Cifar10Octave(OctaveModel):
    '''
    extends OctaveModel
    '''
    # properties
    flatten: K.layers.Flatten
    fc6: CompressibleDense
    classification_layer: CompressibleDense

    # constructor
    def __init__(self, **kwargs):
        # call super init
        super().__init__(**kwargs)

    # build model
    def build(self, input_shape: tuple=(32,32,3)):
        # conv 1_1
        self.build_oct_conv(64, (3,3), padding=Padding.SAME, oct_type='FIRST', kernel_regularizer=self.regularizer, name='conv1_1')
        self.build_oct_batchnorm(name='batchnorm1_1')
        self.build_oct_relu(name='relu1_1')

        # conv 1_2
        self.build_oct_conv(64, (3,3), padding=Padding.SAME, kernel_regularizer=self.regularizer, name='conv1_2')
        self.build_oct_batchnorm(name='batchnorm1_2')
        self.build_oct_relu(name='relu1_2')

        # maxpool 1
        self.build_oct_maxpool(name='maxpool1')

        # conv 2_1
        self.build_oct_conv(128, (3,3), padding=Padding.SAME, kernel_regularizer=self.regularizer, name='conv2_1')
        self.build_oct_batchnorm(name='batchnorm2_1')
        self.build_oct_relu(name='relu2_1')

        # conv 2_2
        self.build_oct_conv(128, (3,3), padding=Padding.SAME, kernel_regularizer=self.regularizer, name='conv2_2')
        self.build_oct_batchnorm(name='batchnorm2_2')
        self.build_oct_relu(name='relu2_2')

        # maxpool 2
        self.build_oct_maxpool(name='maxpool2')

        # conv 3_1
        self.build_oct_conv(256, (3,3), padding=Padding.SAME, kernel_regularizer=self.regularizer, name='conv3_1')
        self.build_oct_batchnorm(name='batchnorm3_1')
        self.build_oct_relu(name='relu3_1')

        # conv 3_2
        self.build_oct_conv(256, (3,3), padding=Padding.SAME, kernel_regularizer=self.regularizer, name='conv3_2')
        self.build_oct_batchnorm(name='batchnorm3_2')
        self.build_oct_relu(name='relu3_2')

        # conv 3_3
        self.build_oct_conv(256, (3,3), padding=Padding.SAME, kernel_regularizer=self.regularizer, name='conv3_3')
        self.build_oct_batchnorm(name='batchnorm3_3')
        self.build_oct_relu(name='relu3_3')

        # maxpool 3
        self.build_oct_maxpool(name='maxpool3')

        # conv 4_1
        self.build_oct_conv(512, (3,3), padding=Padding.SAME, kernel_regularizer=self.regularizer, name='conv4_1')
        self.build_oct_batchnorm(name='batchnorm4_1')
        self.build_oct_relu(name='relu4_1')

        # conv 4_2
        self.build_oct_conv(512, (3,3), padding=Padding.SAME, kernel_regularizer=self.regularizer, name='conv4_2')
        self.build_oct_batchnorm(name='batchnorm4_2')
        self.build_oct_relu(name='relu4_2')

        # conv 4_3
        self.build_oct_conv(512, (3,3), padding=Padding.SAME, kernel_regularizer=self.regularizer, activation=tf.nn.relu, name='conv4_3')
        self.build_oct_batchnorm(name='batchnorm4_3')
        self.build_oct_relu(name='relu4_3')

        # maxpool 4
        self.build_oct_maxpool(name='maxpool4')

        # conv 5_1
        self.build_oct_conv(512, (3,3), padding=Padding.SAME, kernel_regularizer=self.regularizer, name='conv5_1')
        self.build_oct_batchnorm(name='batchnorm5_1')
        self.build_oct_relu(name='relu5_1')

        # conv 5_2
        self.build_oct_conv(512, (3,3), padding=Padding.SAME, kernel_regularizer=self.regularizer, name='conv5_2')
        self.build_oct_batchnorm(name='batchnorm5_2')
        self.build_oct_relu(name='relu5_2')

        # conv 5_3
        self.build_oct_conv(512, (3,3), padding=Padding.SAME, oct_type='LAST', kernel_regularizer=self.regularizer, name='conv5_3')
        self.batchnorm5_3 = K.layers.BatchNormalization(name='batchnorm5_3')

        # maxpool 5
        self.maxpool5 = K.layers.MaxPool2D(name='maxpool5')

        # flatten
        self.flatten = K.layers.Flatten()

        # fc6
        self.fc6 = CompressibleDense(512, kernel_regularizer=self.regularizer, name='fc6')
        self.batchnorm6 = K.layers.BatchNormalization(name='batchnorm6')
        self._compressible_layers.append(self.fc6)
        self.drop6 = K.layers.Dropout(0.5)

        # define classfication layer
        self.classification_layer = CompressibleDense(10, activation=tf.nn.softmax)
        self._compressible_layers.append(self.classification_layer)

        # print oct ratio
        print(self.oct_ratio_list)

    # build an octave convolutional layer
    def build_oct_conv(self, *args, oct_ratio: Tuple[float, float]=None, **kwargs):
        # get input alpha
        alpha_in = self._oct_ratio
        
        # calculate octave ratio
        if oct_ratio is None:
            alpha_out = self.calculate_oct_ratio()
        else:
            [_, alpha_out] = oct_ratio

        # octave block
        oct_block = OctaveConv2D(*args, input_oct_ratio=alpha_in, output_oct_ratio=alpha_out, **kwargs)
        self._oct_layers.append(oct_block)
        self._compressible_oct_layers.append(oct_block)

    # build an octave maxpool layer
    def build_oct_maxpool(self, *args, **kwargs):
        # maxpool
        maxpool = OctaveMaxPool2D(*args, **kwargs)
        self._oct_layers.append(maxpool)

    # build batch normalization layer
    def build_oct_batchnorm(self, *args, **kwargs):
        # batchnorm
        batchnorm = OctaveBatchNormalization(*args, **kwargs)
        self._oct_layers.append(batchnorm)

    # build relu
    def build_oct_relu(self, *args, **kwargs):
        # relu
        relu = OctaveRelu(*args, **kwargs)
        self._oct_layers.append(relu)

    # call method
    def call(self, input_data: Any) -> Any:
        # call octave structure
        model = super().call(input_data)
        model = self.batchnorm5_3(model)
        model = tf.nn.relu(model, name='relu5_3')

        # maxpool 5
        model = self.maxpool5(model)

        # logits
        model = self.flatten(model)

        # fc6 - drop6
        model = self.fc6(model)
        model = self.batchnorm6(model)
        model = tf.nn.relu(model, name='relu6')
        model = self.drop6(model)

        # classification layer
        model = self.classification_layer(model)
        return model