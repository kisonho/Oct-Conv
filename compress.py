# import argparse
import argparse

# main
if __name__=="__main__":
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset name, either cifar10 or imagenet.")
    parser.add_argument("-alpha", "--oct_ratio", default=0.25, type=float, help="Octave ratio of the model. Default is 0.25.")
    parser.add_argument("--log_dir", default="Data/", help="The directory of log files.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size of the dataset. Default is 128.")

    # compression optimizers
    parser.add_argument("-e", "--epochs", default=10, type=int, help="epochs to compress. Default is 10.")
    parser.add_argument("-lambda", "--compression_lambda", default=5, type=float, help="The compression lambda to compress the model. Default is 5.")
    parser.add_argument("-k", "--frequency_multiplier", default=1, type=float, help="The frequency multiplier of compression lambda. Default is 1.")
    parser.add_argument("--experiment", default="test", type=str, help="Experiment name")
    parser.add_argument("--model_dir", default='cifar10.h5', type=str, help="Directory of model to compress, default is 'cifar10.h5'.")
    parser.add_argument("--output_dir", default='cifar10.h5', type=str, help="Directory of compressed model, default is 'cifar10.h5'.")

    # read arguments
    args = parser.parse_args()
    dataset = args.dataset
    oct_ratio = args.oct_ratio
    # oct_mode = args.oct_mode
    root_dir = args.log_dir
    batch_size = args.batch_size
    experiment_name = args.experiment
    model_dir = args.model_dir
    output_dir = args.output_dir

    # read compression arguments
    epochs = args.epochs
    compression_lambda = args.compression_lambda
    frequency_multiplier = args.frequency_multiplier

    # call main function
    main_module = __import__("compress_"+dataset)
    main_module.compress(model_dir, oct_ratio, root_dir=root_dir, batch_size=batch_size, epochs=epochs, compression_lambda=compression_lambda, frequency_multiplier=frequency_multiplier, experiment_name=experiment_name, output_dir=output_dir)