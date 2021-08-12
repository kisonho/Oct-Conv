# import system module
from typing import Any, Callable, List, Optional, Tuple
from enum import Enum
import threading, time

# tensorflow import
import tensorflow as tf
from tensorflow import keras as K

# import core modules
from .compression import CompressibleLayer
from .oct_layers import OctaveLayer

# compressible model
class CompressibleModel(K.Model):
    # properties
    __is_compressing: bool
    _compressible_layers: List[CompressibleLayer]

    @property
    def is_compressing(self) -> bool:
        return self.__is_compressing
        
    @is_compressing.setter
    def is_compressing(self, is_compressing: bool):
        self.__is_compressing = is_compressing

    @property
    def compressible_layers(self) -> List[CompressibleLayer]:
        return self._compressible_layers

    # constructor
    def __init__(self, is_compressing: bool=False, **kwargs):
        '''
        is_compressing - a boolean to control if the model is compressible
        '''
        # call super constructor
        super().__init__(**kwargs)
        
        # initialize parameters
        self.is_compressing = is_compressing
        self._compressible_layers = list()

    # compress the model
    def compress(self):       
        assert self.is_compressing is True

        # initialize thread list
        compressing_threads: List[threading.Thread] = list()

        # loop layers
        for layer in self.compressible_layers:             
            t = threading.Thread(target=layer.compress)
            compressing_threads.append(t)
            t.start()

        # join threads
        for t in compressing_threads:
            t.join()
            
    # evaluate
    def evaluate_size(self, **kwargs) -> Tuple[int, Optional[int]]:
        # check if model is compressible
        if self.is_compressing is True:  
            # initial size
            compressed_size = 0          

            # loop layers
            for layer in self.compressible_layers:
                compressed_size += layer.evaluate_size()

            # classification layer
            for w in self.classification_layer.get_weights():
                compressed_size += tf.cast(tf.size(w), tf.int64) * 32
        else:
            compressed_size = None
            
        # get total trainable weights
        uncompressed_size = 0
        for w in self.get_weights():
            uncompressed_size += tf.size(w) * 32
        return uncompressed_size, compressed_size

    # fit with compression, use this to test speed
    def fit_with_compression(self, dataset: tf.data.Dataset, epochs=1, initial_epoch: int=0):
        # epoch loop
        for _ in range(initial_epoch, epochs):            
            # initialize batch
            batch: int = 0
            t: float = 0

            # batch loop 
            for example in dataset:
                # unpack dataset
                x_train, y_train = example

                # forward pass
                def forward_pass() -> tf.Tensor:
                    # calculate loss
                    y = self(x_train)
                    l: tf.Tensor = self.loss(y_train, y)
                    return l

                # minimize
                start_time: float = time.time()
                self.optimizer.minimize(forward_pass, self.trainable_variables)
                end_time: float = time.time()

                # calculate time
                t += end_time - start_time

                # on batch end
                batch += 1

            # evaluate time
            avg_time: float = t / batch
            tf.print('Average time: %f' % avg_time)

# enum octave mode
class OctaveMode(Enum):
    CONSTANT = 'constant'
    LINEAR = 'linear'
    CONVERTED_LINEAR = 'converted_linear'

# base octave model
class OctaveModel(CompressibleModel):
    # properties
    _max_oct_ratio: float
    _depth_decay: float
    _oct_ratio: float
    _oct_depth: int
    _oct_layers: List[OctaveLayer]
    _regularizer: Callable
    _compressible_oct_layers: List[OctaveLayer]
    
    @property
    def oct_mode(self) -> OctaveMode:
        return self._oct_mode

    @property
    def oct_layers(self) -> List[OctaveLayer]:
        return self._oct_layers

    @property
    def oct_ratio_list(self) -> List[float]:
        return self._oct_ratio_list

    @property
    def compressible_oct_layers(self) -> List[OctaveLayer]:
        return self._compressible_oct_layers

    # constructor
    def __init__(self, max_oct_ratio: float, mode: str='constant', depth_decay: float=1/18, weight_decay: float=4e-5, **kwargs):
        # call super constructor
        super().__init__(**kwargs)

        # initialize variables
        self._oct_mode = OctaveMode[mode.upper()]
        self._max_oct_ratio = max_oct_ratio
        self._depth_decay = depth_decay
        self._oct_ratio = 0.
        self._oct_depth = 1
        self._oct_ratio_list = list()
        self._oct_layers = list()
        self._regularizer = tf.keras.regularizers.l2(weight_decay)
        self._compressible_oct_layers = list()

    # calculate octave ratio by decay
    def calculate_oct_ratio(self) -> float:
        '''
        returns the octave ratio of current layer calculated by the decay rate and octave depth in specific octave mode
        '''
        if self.oct_mode == OctaveMode.CONSTANT:
            # constant oct ratio
            self._oct_ratio = self._max_oct_ratio
        elif self.oct_mode == OctaveMode.LINEAR:
            # linear decay
            self._oct_ratio = min(self._max_oct_ratio * self._depth_decay * self._oct_depth, self.max_oct_ratio)
            self._oct_depth += 1
        elif self.oct_mode == OctaveMode.CONVERTED_LINEAR:
            # converted linear decay
            self._oct_ratio = min(self._max_oct_ratio * (1 - self._depth_decay * (self._oct_depth - 1)), self.max_oct_ratio)
            self._oct_depth += 1
        else:
            raise TypeError('Mode %s not supported' % self.oct_mode.name)
        
        # return ratio
        self._oct_ratio_list.append(self._oct_ratio)
        return self._oct_ratio

    # calculate mean oct ratio
    def calculate_mean_oct_ratio(self) -> float:
        '''
        returns the mean of oct ratio
        '''
        return tf.reduce_mean(self.oct_ratio_list)

    # call model
    def call(self, input_data: Any) -> tf.Tensor:
        # set up input data
        x = [input_data, None]

        # call octave layers
        for l in self.oct_layers:
            x = l(x)

        # get final feature maps
        y = x[0]
        return y

    # compress the model
    def compress(self):       
        assert self.is_compressing is True

        # initialize thread list
        compressing_threads: List[threading.Thread] = list()

        # loop layers
        for layer in self.compressible_oct_layers:             
            t = threading.Thread(target=layer.compress)
            compressing_threads.append(t)
            t.start()

        # call super compress method
        super().compress()

        # join threads
        for t in compressing_threads:
            t.join()