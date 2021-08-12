# import argparse
import argparse

# main
if __name__=="__main__":
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset name, either cifar10 or imagenet.")
    parser.add_argument("-alpha", "--oct_ratio", default=0.25, type=float, help="Octave ratio of the model. Default is 0.25.")
    parser.add_argument("--oct_mode", default="constant", help="Octave mode of the model, either constant, linear, or converted_linear. Default is \'constant\'.")
    parser.add_argument("--log_dir", default="Data/", help="The directory of log files.")

    # training parameters
    parser.add_argument("-e", "--epochs", default=10, type=int, help="Epochs to train. Default is 10.")
    parser.add_argument("--initial_epoch", default=0, type=int, help="Initial epoch to train. Default is 0.")
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="Batch size of the dataset. Default is 64.")
    parser.add_argument("--experiment", default="test", type=str, help="Experiment name")
    parser.add_argument("--output_dir", default="models.h5")

    # read arguments
    args = parser.parse_args()
    dataset = args.dataset
    oct_ratio = args.oct_ratio
    oct_mode = args.oct_mode
    root_dir = args.log_dir
    epochs = args.epochs
    initial_epoch = args.initial_epoch
    batch_size = args.batch_size
    experiment_name = args.experiment
    output_dir = args.output_dir

    # call main function
    main_module = __import__("train_"+dataset)
    main_module.train(oct_ratio, oct_mode, root_dir, epochs=epochs, initial_epoch=initial_epoch, batch_size=batch_size, experiment_name=experiment_name, output_dir=output_dir)
