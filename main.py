import os
import tensorflow as tf
import numpy as np
import scipy.io as io
import shutil
import argparse

from model_run import Runner


np.random.seed(1234)
tf_data_type = tf.float64
tf.config.list_physical_devices("GPU")
tf.keras.backend.clear_session()


def main(save_index, load_unpruned, load_pruned):
    tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
    # Create directories
    current_directory = os.getcwd()
    case = "Case_"
    folder_index = str(save_index)
    results_dir = "/" + case + folder_index + "/Results"
    variable_dir = "/" + case + folder_index + "/Variables"
    save_results_to = current_directory + results_dir
    save_variables_to = current_directory + variable_dir

    if not load_pruned and not load_unpruned:
        # Remove existing results
        if os.path.exists(save_results_to) or os.path.exists(save_variables_to):
            shutil.rmtree(save_results_to)
            shutil.rmtree(save_variables_to)

        os.makedirs(save_results_to)
        os.makedirs(save_variables_to)

    # The size of the final layer of the branch network has to be the same 
    # as the size of the final layer of the trunk network. (For shape reasons)
    final = 80

    hyperparameters = {
        "n_channels": 2,
        "filter_size_1": 3,
        "filter_size_2": 3,
        "filter_size_3": 3,
        "filter_size_4": 3,
        "stride": 1,
        "num_filters_1": 40,
        "num_filters_2": 60,
        "num_filters_3": 100,
        "num_filters_4": 180,
        "B_net": [180, 80, final],
        "T_net": [2, 80, 80, final],
        "bs": 50,
        "tsbs": 20,
        "epochs": 200,
        "alpha_fc": 0.95,
        "alpha_conv": 0.90,
    }

    io.savemat(save_variables_to + "/hyperparameters.mat", mdict=hyperparameters)

    # Initialise and run the model
    network = Runner(tf_data_type)
    network.run(hyperparameters, save_results_to, save_variables_to, load_unpruned, load_pruned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Case_", default=1, type=int, help="Keeps track of previous runs")
    parser.add_argument("--load_unpruned", action="store_true", help="Load unpruned model from dictionary")
    parser.add_argument("--load_pruned", action="store_true", help="Load pruned model from dictionary")
    args = parser.parse_args()

    if args.load_pruned and args.load_unpruned:
        raise ValueError("Use either --load_pruned or --load_unpruned. Only one of the flags can be used at a time.")

    if args.load_unpruned:
        load_unpruned = True
        load_pruned = False
    elif args.load_pruned:
        load_unpruned = False
        load_pruned = True
    else:
        load_unpruned = False
        load_pruned = False
    Case = args.Case_
    main(Case, load_unpruned, load_pruned)
