# import system module
from typing import Tuple, Optional

# import tensorflow
import tensorflow as tf
from tensorflow import keras as K

# import core module
from .compression import CompressibleLayer
from .BayesianOptimization.bayes_opt.bayesian_optimization import BayesianOptimization
from .model import CompressibleModel, OctaveModel

# fine tune optimizer for clip-q
class FineTuneOptimizer(K.optimizers.Optimizer):
    # properties
    _model: CompressibleModel
    _weights_optimizer: K.optimizers.Optimizer
    _uncompressed_weights: Optional[list]

    @property
    def model(self) -> CompressibleModel:
        return self._model

    @property
    def weights_optimizer(self) -> K.optimizers.Optimizer:
        return self._weights_optimizer

    @property
    def uncompressed_weights(self) -> Optional[list]:
        return self._uncompressed_weights

    @uncompressed_weights.setter
    def uncompressed_weights(self, w: list):
        self._uncompressed_weights = w

    # constructor
    def __init__(self, model: CompressibleModel, weights_optimizer: K.optimizers.Optimizer, name='fine_tune', **kwargs):
        super().__init__(name, **kwargs)

        # initialize variables
        self._model = model
        self._weights_optimizer = weights_optimizer

        # record variables
        self._uncompressed_weights = list()

        # compress base model layers
        with tf.name_scope('uncompressed'):
            for l in self.model.compressible_layers:
                # initialize layer count
                count: int = 1

                # loop for weights
                for w in l.trainable_variables:   
                    uncompressed_weight = tf.Variable(w)             
                    self.uncompressed_weights.append(uncompressed_weight)
                    count += 1

    '''
    # minimize, used for non GPU-accelerated multithreading compression test
    @tf.function
    def minimize(self, loss, var_list, grad_loss=None, name=None, **kwargs):
        # forward pass
        with tf.GradientTape() as tape:
            # calculate loss
            tape.watch(var_list)
            loss_value = loss()

        # backpropagation
        self._model.set_weights(self.uncompressed_weights)
        var_list = self._model.trainable_variables
        gradients = tape.gradients(loss_value, var_list, grad_loss)

        # apply gradients and compress
        self.weights_optimizer.apply_gradients(list(zip(gradients, var_list)))
        self._uncompressed_weights = self.model.get_weights()
        self.model.compress()
    '''

    # apply gradients, used for final GPU-accelerated compression fine-tuning
    @tf.function
    def apply_gradients(self, *args, **kwargs):
        # reset uncompressed weights
        for l in self.model.compressible_layers:
            # initialize layer count
            count: int = 1

            # loop for weights
            for w in l.trainable_variables:            
                w.assign(self.uncompressed_weights[count])
                count += 1
                
        # update procedure
        self.weights_optimizer.apply_gradients(*args, **kwargs)
        
        # initialize layer count
        count: int = 1

        # compress base model layers
        for l in self.model.compressible_layers:
            # loop for weights
            for w in l.trainable_variables:                
                self.uncompressed_weights[count].assign(w)
                count += 1

            # compress conv2d and dense layer
            l.compress()

    # get config
    def get_config(self) -> dict:
        return self.weights_optimizer.get_config()

# normal bayesian optimizer for clip-q
class CompressionOptimizer(K.optimizers.Optimizer):
    # properties
    _compression_lambda: float
    _optimize_iterations: int
    _model: CompressibleModel

    @property
    def compression_lambda(self) -> float:
        return self._compression_lambda
        
    @property
    def optimize_iterations(self) -> int:
        return self._optimize_iterations

    # constructor
    def __init__(self, compression_lambda: float, model: CompressibleModel, optimize_iterations: int=50, name: str='compression_optimizer', **kwargs):
        # call super constructor
        super().__init__(name, **kwargs)

        # initialize parameters
        self._compression_lambda = tf.constant(compression_lambda, dtype=tf.float32)
        self._optimize_iterations = int(optimize_iterations)
        self._model = model
        self._total_size = -1

    # initialize model size
    def initialize_model_size(self):
        # initialize size
        self._total_size = 0

        # get weights
        weights = self._model.get_weights()
        
        # calculate total size
        for w in weights:
            self._total_size += tf.size(w) * 32

    # run clip q
    def minimize(self, compression_dataset: tf.data.Dataset):
        # initialize model size
        if self._total_size < 0:
            self.initialize_model_size()

        # clip q loop for layers
        for block in self._model.compressible_layers:
            # compress block
            training_dataset = compression_dataset

            # bayesian optimize
            parameters = self.optimize_parameters(block, training_dataset)
            block.set_parameters(parameters)
            print("Layer %s: p=%f, b=%d" % (block.name, parameters[0], parameters[1]))

    # bayesian optimize compression parameters
    def optimize_parameters(self, block: CompressibleLayer, training_dataset: tf.data.Dataset) -> Tuple[float, int]: 
        # calculate benifit function
        def f(optimized_p: float, optimized_b: float) -> float:
            # initialize compression parameter
            parameters = (tf.constant(optimized_p), tf.constant(int(optimized_b)))

            # get weights
            uncompressed_weights = block.get_weights()

            # calculate weight size
            weight_size = 0
            for w in uncompressed_weights:
                weight_size += tf.size(w)
            
            # calculate hf benifit
            benifit = self.calculate_benifit(tf.constant(parameters[0]), tf.constant(parameters[1]), self.compression_lambda, tf.constant(weight_size))
            
            # compress layer
            block.compress(parameters=parameters)
            
            # initialize dataset
            example = list(training_dataset.take(1))[0]
            (x_train, y_train) = example
            
            # evaluate model
            y = self._model(x_train)
            loss = 1 - self._model.metrics[0](y_train, y)
        
            # reset weights
            block.set_weights(uncompressed_weights)
            
            # calculate bayesian optimization loss
            loss -= tf.cast(benifit, tf.float32)
            return loss
        
        # initialize bayesian optimizer
        bounds = {'optimized_p': (0,0.99), 'optimized_b': (2,8)}
        optimizer = BayesianOptimization(f=f, pbounds=bounds)

        # optimize compression parameters
        optimizer.maximize(n_iter=self.optimize_iterations)
        optimized_max_params = optimizer.max['params']
        optimized_p, optimized_b = optimized_max_params['optimized_p'], int(optimized_max_params['optimized_b'])
        
        # return optimized parameters
        return (optimized_p, optimized_b)

    # calculate benifit
    @tf.function
    def calculate_benifit(self, p: float, b: int, compression_lambda: float, weight_size: int) -> float:
        # calculate benifit
        pruned_size = tf.cast((1 - p) * tf.cast(weight_size, tf.float64), tf.int32) * b
        benifit = weight_size * 32 - tf.cast(pruned_size, tf.int32)
        benifit = tf.cast(compression_lambda, tf.float64) * tf.cast(benifit, tf.float64) / tf.cast(self._total_size, tf.float64)

        # bayesian minimize pruning ratio and quantization bits
        return benifit

# octave bayesian optimizer for clip-q
class OctaveCompressionOptimizer(CompressionOptimizer):
    # properties
    _frequency_multiplier: float
    _model: OctaveModel
        
    @property
    def frequency_multiplier(self) -> float:
        return self._frequency_multiplier
        
    @frequency_multiplier.setter
    def frequency_multiplier(self, k: float):
        self._frequency_multiplier = k

    # constructor
    def __init__(self, *args, frequency_multiplier: float, **kwargs):
        # call super constructor
        super().__init__(*args, **kwargs)

        # initialize parameters
        self.frequency_multiplier = tf.constant(frequency_multiplier, dtype=tf.float32)

    # run clip q
    def minimize(self, compression_dataset: tf.data.Dataset):
        # initialize model size
        if self._total_size < 0:
            self.initialize_model_size()

        # clip q loop for layers
        for block in self._model.compressible_oct_layers:
            # compress block
            training_dataset = compression_dataset

            # record overall lambda
            overall_lambda = self._compression_lambda

            # optimize hf
            if block.frequency_blocks['hf'] is not None:
                # when hf layer is not none
                self._compression_lambda = overall_lambda * (self._frequency_multiplier + (1 - self._frequency_multiplier) / (1 - block.oct_ratio))
                hf_parameters = self.optimize_parameters(block.frequency_blocks['hf'], training_dataset) # type: ignore None
            else:
                hf_parameters = None

            # optimize lf
            if block.frequency_blocks['lf'] is not None:
                # when lf layer is not none
                self._compression_lambda = overall_lambda * self._frequency_multiplier
                lf_parameters = self.optimize_parameters(block.frequency_blocks['lf'], training_dataset) # type: ignore None
            else:
                lf_parameters = None

            # set parameters
            block.set_parameters((hf_parameters, lf_parameters))
            self._compression_lambda = overall_lambda

            # print result
            block.print_summary()
        
        # minimize non-oct block
        super().minimize(compression_dataset)

    # override get config method
    def get_config(self) -> dict:
        config = dict()
        return config
