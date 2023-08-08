import tensorflow as tf
import numpy as np
import time
import scipy.io as io

from dataset import DataSet
from DeepONet import DeepONet
from model_train import Train_Adam
from model_error import Error_Test
from model_plot import Plot
from NNRelief import pruning


class Runner:
    def __init__(self, tf_data_type):
        self.tf_data_type = tf_data_type

    def run(self, hyperparameters, save_results_to, save_variables_to, load_unpruned_bool, load_pruned_bool):
        """
        Runs the model.

        Args:
            hyperparameters (Dict): Dictionary of hyperparameters.
            save_results_to (str): Directory to save results to.
            save_variables_to (str): Directory to save variables to.
            load_unpruned_bool (bool): Boolean for whether or not the 
            unpruned model should be loaded.
            load_pruned_bool (bool): Boolean for whether or not the 
            pruned model should be loaded.
        """
        filter_size_1 = hyperparameters["filter_size_1"]
        filter_size_2 = hyperparameters["filter_size_2"]
        filter_size_3 = hyperparameters["filter_size_3"]
        filter_size_4 = hyperparameters["filter_size_4"]
        num_filters_1 = hyperparameters["num_filters_1"]
        num_filters_2 = hyperparameters["num_filters_2"]
        num_filters_3 = hyperparameters["num_filters_3"]
        num_filters_4 = hyperparameters["num_filters_4"]
        n_channels = hyperparameters["n_channels"]
        stride = hyperparameters["stride"]
        B_net = hyperparameters["B_net"]
        T_net = hyperparameters["T_net"]
        bs = hyperparameters["bs"]
        tsbs = hyperparameters["tsbs"]
        epochs = hyperparameters["epochs"]
        a_fc = hyperparameters["alpha_fc"]
        a_conv = hyperparameters["alpha_conv"]

        var = [8820] # Need to Automate this
        B_net = var + B_net
        

        param = DataSet(bs, tsbs)
        model = DeepONet(self.tf_data_type)

        if load_unpruned_bool:
            unpruned_dict = io.loadmat(save_variables_to + "/Weight_bias.mat")
            start_time = time.perf_counter()
            time_step_0 = time.perf_counter()
            x_train, f_train, u_train, f_norm_train, Xmin, Xmax = param.minibatch()
            percent_pruned, pruned_dict = self.prune_help(model, param, hyperparameters, save_results_to, save_variables_to, B_net, T_net, time_step_0, load_unpruned_bool, unpruned_dict, Xmin, Xmax, stride, a_conv, a_fc)
        
        elif load_pruned_bool:
            percent_pruned, io.loadmat(save_variables_to + "/Percent_pruned.mat")
            pruned_dict = io.loadmat(save_variables_to + "/Weight_bias_pruned.mat")
            start_time = time.perf_counter()
            time_step_0 = time.perf_counter()
            
        else:
            # Branch CNN initialisation
            W_br_1, b_br_1 = model.hyper_initial_cnn([filter_size_1, filter_size_1, n_channels, num_filters_1], num_filters_1)
            W_br_2, b_br_2 = model.hyper_initial_cnn([filter_size_2, filter_size_2, num_filters_1, num_filters_2], num_filters_2)
            W_br_3, b_br_3 = model.hyper_initial_cnn([filter_size_3, filter_size_3, num_filters_2, num_filters_3], num_filters_3)
            W_br_4, b_br_4 = model.hyper_initial_cnn([filter_size_4, filter_size_4, num_filters_3, num_filters_4], num_filters_4)

            # Branch FNN initialisation
            W_br_fnn, b_br_fnn = model.hyper_initial_fnn(B_net)

            # Trunk initialisation
            W_tr, b_tr = model.hyper_initial_fnn(T_net)

            W_b_dict = {
                "W_1 conv br": W_br_1,
                "b_1 conv br": b_br_1,
                "W_2 conv br": W_br_2,
                "b_2 conv br": b_br_2,
                "W_3 conv br": W_br_3,
                "b_3 conv br": b_br_3,
                "W_4 conv br": W_br_4,
                "b_4 conv br": b_br_4,
                "W_5 fc br": W_br_fnn,
                "b_5 fc br": b_br_fnn,
                "W_1 fc tr": W_tr,
                "b_1 fc tr": b_tr,
            }

            Train_Model_Adam = Train_Adam(model, self.tf_data_type, B_net, T_net, W_b_dict, hyperparameters)
            Test_error = Error_Test(W_b_dict, model, self.tf_data_type, save_results_to)
            # optimiser = tf.keras.optimizers.legacy.Adam() # if running on laptop running Apple Silicon (Debugging only)
            optimiser = tf.keras.optimizers.Adam()

            ######################
            ### Training Model ###
            ######################

            n = 0
            start_time = time.perf_counter()
            time_step_0 = time.perf_counter()

            train_loss = np.zeros((epochs + 1, 1))
            test_loss = np.zeros((epochs + 1, 1))
            while n <= epochs:
                x_train, f_train, u_train, f_norm_train, Xmin, Xmax = param.minibatch()
                lr = 0.0001
                optimiser.lr.assign(lr)
                train_dict, train_W_b_dict = Train_Model_Adam.nn_train(
                    optimiser, x_train, f_train, u_train, f_norm_train, Xmin, Xmax
                )

                loss = train_dict["loss"]

                if n % 50 == 0:
                    print("##########   WHILE   ############")
                    print("w shape", train_W_b_dict["W_1 conv br"].shape)
                    print("b shape", train_W_b_dict["b_1 conv br"].shape)
                    print("#################################")
                    x_test, f_test, u_test, f_norm_test, batch_id = param.testbatch()
                    u_pred = Train_Model_Adam.call(x_test, f_test, f_norm_test, Xmin, Xmax) # done to prevent reloading model
                    err = np.mean((u_test - u_pred) ** 2 / (u_test**2 + 1e-4))
                    err = np.reshape(err, (-1, 1))
                    time_step_1000 = time.perf_counter()
                    T = time_step_1000 - time_step_0
                    print(
                        "Step: %d, Loss: %.4e, Test L2 error: %.4f, Time (secs): %.4f"
                        % (n, loss, err, T)
                    )
                    time_step_0 = time.perf_counter()

                train_loss[n, 0] = loss
                test_loss[n, 0] = err
                n += 1

            x_print, f_print, u_print, f_norm_print, batch_id = param.printbatch()
            batch_id, f_print, u_print, u_pred = \
                Test_error.nn_error_test(x_print, f_print, u_print, f_norm_print, stride, Xmin, Xmax, batch_id)
            err = np.mean((u_print - u_pred) ** 2 / (u_print ** 2 + 1e-4))
            err = np.reshape(err, (-1, 1))
            np.savetxt(save_results_to + "/err", err, fmt="%e")
            io.savemat(
                save_results_to + "/Darcy.mat",
                mdict={
                    "test_id": batch_id,
                    "x_test": f_print,
                    "y_test": u_print,
                    "y_pred": u_pred,
                },
            )
            # stop_time = time.perf_counter()
            # print("Elapsed time (secs): %.3f" % (stop_time - start_time))

            ###############################
            ### Saving Unpruned Results ###
            ###############################

            print("##########   SAVING   ############")
            print("w shape", train_W_b_dict["W_1 conv br"].shape)
            print("b shape", train_W_b_dict["b_1 conv br"].shape)
            print("###################################")

            W_br_1_save, b_br_1_save = model.save_W_b(train_W_b_dict["W_1 conv br"], train_W_b_dict["b_1 conv br"])
            W_br_2_save, b_br_2_save = model.save_W_b(train_W_b_dict["W_2 conv br"], train_W_b_dict["b_2 conv br"])
            W_br_3_save, b_br_3_save = model.save_W_b(train_W_b_dict["W_3 conv br"], train_W_b_dict["b_3 conv br"])
            W_br_4_save, b_br_4_save = model.save_W_b(train_W_b_dict["W_4 conv br"], train_W_b_dict["b_4 conv br"])
            W_br_fnn_save, b_br_fnn_save = model.save_W_b(train_W_b_dict["W_5 fc br"], train_W_b_dict["b_5 fc br"])
            W_tr_save, b_tr_save = model.save_W_b(train_W_b_dict["W_1 fc tr"], train_W_b_dict["b_1 fc tr"])

            W_b_dict_save = {
                "W_1 conv br": W_br_1_save,
                "b_1 conv br": b_br_1_save,
                "W_2 conv br": W_br_2_save,
                "b_2 conv br": b_br_2_save,
                "W_3 conv br": W_br_3_save,
                "b_3 conv br": b_br_3_save,
                "W_4 conv br": W_br_4_save,
                "b_4 conv br": b_br_4_save,
                "W_5 fc br": W_br_fnn_save,
                "b_5 fc br": b_br_fnn_save,
                "W_1 fc tr": W_tr_save,
                "b_1 fc tr": b_tr_save,
            }

            io.savemat(save_variables_to + "/Weight_bias.mat", W_b_dict_save)
            print("Completed storing unpruned weights and biases")

            print("##########  ELSE RUN   ############")
            print("w shape", train_W_b_dict["W_1 conv br"].shape)
            print("b shape", train_W_b_dict["b_1 conv br"].shape)
            print("###################################")

            percent_pruned, pruned_dict = self.prune_help(model, param, hyperparameters, save_results_to, save_variables_to, B_net, T_net, time_step_0, load_unpruned_bool, train_W_b_dict, Xmin, Xmax, stride, a_conv, a_fc)

        stop_time = time.perf_counter()
        print("Elapsed time (secs): %.3f" % (stop_time - start_time))


        plot = Plot(save_results_to)
        plot.Plotting(train_loss, test_loss)

    def prune_help(self, model, param, hyperparameters, save_results_to, save_variables_to, B_net, T_net, time_step_0, load_bool, unpruned_dict, Xmin, Xmax, stride, a_conv, a_fc):
        """
        Helper function that prunes the network. Gets called either when loading an unpruned network or when pruning a freshly trained network.

        Args:
            model (DeepONet): DeepONet object.
            param (DataSet): DataSet object.
            hyperparameters (dict): Dictionary of hyperparameters.
            save_results_to (str): Directory to save results to.
            save_variables_to (str): Directory to save variables to.
            B_net (list): List of branch network architecture. 
            T_net (list): List of trunk network architecture.
            time_step_0 (int): Number of time steps. (Performance Metric)
            unpruned_dict (dict): Dictionary of unpruned weights and biases we want to prune.
            Xmin (float): Minimum value of the input spatial coordinates.
            Xmax (float): Maximum value of the input spatial coordinates.
            stride (int): Stride of the convolutional layers.
            a_conv (float): Alpha value for the convolutional layers.
            a_fc (float): Alpha value for the fully connected layers.

        Returns:
            dict: Dictionary of pruned weights and biases.
            float: Percentage of weights and biases pruned.
        """ 
        pruned_dict_0 = unpruned_dict
        pruned_dict_1 = unpruned_dict
        err_0 = 1 # perhaps change the metric used to know when to stop pruning
        err_1 = 0
        percent_pruned_0 = 0
        percent_pruned_1 = 0
        prune_iter = 0
        

        while err_1 < err_0 or prune_iter < 100:
            err_0 = err_1
            percent_pruned_0 = percent_pruned_1
            pruned_dict_0 = pruned_dict_1

            x_test, f_test, u_test, f_norm_test, batch_id = param.testbatch()

            print("##########   PRUNING MODEL RUN   ############")
            print("w shape", pruned_dict_0["W_1 conv br"].shape)
            print("b shape", pruned_dict_0["b_1 conv br"].shape)
            print("#############################################")

            pruned_dict_1, percent_pruned_1 = pruning(model, load_bool, pruned_dict_0, f_test, x_test, Xmin, Xmax, stride, a_conv, a_fc)
            Train_Model_pruned = Train_Adam(model, self.tf_data_type, B_net, T_net, pruned_dict_1, hyperparameters) # reinitialise model with pruned weights and biases
            u_pred = Train_Model_pruned.call(x_test, f_test, f_norm_test, Xmin, Xmax)
            err_1 = np.mean((u_test - u_pred) ** 2 / (u_test ** 2 + 1e-4))
            err_1 = np.reshape(err_1, (-1, 1))
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            print(
                "Pruning Iteration: %d, Test L2 error: %.4f, Time (secs): %.4f"
                % (prune_iter, err_1, T)
            )
            prune_iter += 1

        final_percent_pruned = percent_pruned_0
        final_pruned_dict = pruned_dict_0
        io.savemat(save_variables_to + "/Weight_bias_pruned.mat", final_pruned_dict)
        io.savemat(save_variables_to + "/Percent_pruned.mat", final_percent_pruned)
        print("Completed storing pruned trainable variables, and pruning metrics")

        print("Final Percent Pruned: ", final_percent_pruned)


        x_print, f_print, u_print, f_norm_print, batch_id = param.printbatch()
        Test_error_pruned = Error_Test(final_pruned_dict, model, self.tf_data_type, save_results_to) # reinitialise error class with pruned weights and biases

        batch_id, f_print, u_print, u_pred = \
            Test_error_pruned.nn_error_test(x_print, f_print, u_print, f_norm_print, stride, Xmin, Xmax, batch_id)
        err = np.mean((u_print - u_pred) ** 2 / (u_print ** 2 + 1e-4))
        err = np.reshape(err, (-1, 1))
        np.savetxt(save_results_to + "/err", err, fmt="%e")
        io.savemat(
            save_results_to + "/Darcy_pruned.mat",
            mdict={
                "test_id": batch_id,
                "x_test": f_print,
                "y_test": u_print,
                "y_pred": u_pred,
            },
        )
        print("Saved Pruned Results")

        return final_percent_pruned, final_pruned_dict