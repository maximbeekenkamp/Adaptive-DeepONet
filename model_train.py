import tensorflow as tf


class Train_Adam:
    def __init__(self, model, tf_data_type, B_net, T_net, W_b_dict, hyperparameters):
        self.model = model
        self.tf_data_type = tf_data_type
        self.B_net = B_net
        self.T_net = T_net
        self.W_b_dict = W_b_dict
        self.stride = hyperparameters["stride"]
        tf.keras.mixed_precision.set_global_policy("float64")

    def train_vars(self):
        """
        Creates a list of trainable variables for the model.

        Returns:
            List: Trainable variables, consisting of the weights and biases of the model.
        """
        Y = (
            [self.W_b_dict["W_1 conv br"]]
            + [self.W_b_dict["b_1 conv br"]]
            + [self.W_b_dict["W_2 conv br"]]
            + [self.W_b_dict["b_2 conv br"]]
            + [self.W_b_dict["W_3 conv br"]]
            + [self.W_b_dict["b_3 conv br"]]
            + [self.W_b_dict["W_4 conv br"]]
            + [self.W_b_dict["b_4 conv br"]]
            + self.W_b_dict["W_5 fc br"]
            + self.W_b_dict["b_5 fc br"]
            + self.W_b_dict["W_1 fc tr"]
            + self.W_b_dict["b_1 fc tr"]
        )

        return Y

    @tf.function(jit_compile=True)
    def call(self, X, F, F_norm, Xmin, Xmax):
        """
        Forward pass of the DeepONet, containing both Branch and Trunk networks.

        Args:
            X (Tensor object of ndarray): 2D array of the input spatial coordinates. Used in the Trunk network.
            F (Tensor object of ndarray): 2D array of the input functions. Used in the Branch network.
            F_norm (Tensor object of ndarray): Masking layer for the Branch network. 1 for spaces inside the target domain, 
            0 for spaces outside the target domain.
            Xmin (Tensor object of ndarray): Minimum value of the input spatial coordinates.
            Xmax (Tensor object of ndarray): Maximum value of the input spatial coordinates.

        Returns:
            Tensor object of ndarray: Prediction of the DeepONet.
        """
        ######################
        ### Branch Network ###
        ######################

        # CNN1
        print("#######  TRAINING   ##########")
        print("x shape", F.shape)
        print("w shape", self.W_b_dict["W_1 conv br"].shape)
        print("b shape", self.W_b_dict["b_1 conv br"].shape)
        print("##############################")
        b_out, self.W_b_dict["W_1 conv br"], self.W_b_dict["b_1 conv br"] = \
        self.model.conv_layer(F, self.W_b_dict["W_1 conv br"], self.W_b_dict["b_1 conv br"], self.stride)
        pool = self.model.avg_pool(b_out, 2, 2)# switch to max_pool

        # CNN2
        b_out, self.W_b_dict["W_2 conv br"], self.W_b_dict["b_2 conv br"] = \
            self.model.conv_layer(pool, self.W_b_dict["W_2 conv br"], self.W_b_dict["b_2 conv br"], self.stride)
        pool = self.model.avg_pool(b_out, 2, 2)

        # CNN3
        b_out, self.W_b_dict["W_3 conv br"], self.W_b_dict["b_3 conv br"] = \
            self.model.conv_layer(pool, self.W_b_dict["W_3 conv br"], self.W_b_dict["b_3 conv br"], self.stride)
        pool = self.model.avg_pool(b_out, 2, 2)

        # CNN4
        b_out, self.W_b_dict["W_4 conv br"], self.W_b_dict["b_4 conv br"] = \
            self.model.conv_layer(pool, self.W_b_dict["W_4 conv br"], self.W_b_dict["b_4 conv br"], self.stride)
        pool = self.model.avg_pool(b_out, 2, 2)
        flat = self.model.flatten_layer(pool)
        
        # FNN
        u_B = self.model.fnn_B(self.W_b_dict["W_5 fc br"], self.W_b_dict["b_5 fc br"], flat)

        #####################
        ### Trunk Network ###
        #####################

        # FNN
        u_T = self.model.fnn_T(self.W_b_dict["W_1 fc tr"], self.W_b_dict["b_1 fc tr"], X, Xmin, Xmax)

        ###

        # Combine Branch and Trunk networks
        u_nn = tf.einsum("ik,jk->ij", u_B, u_T)
        u_nn = tf.expand_dims(u_nn, axis=-1)
        u_pred = u_nn * F_norm

        return u_pred

    @tf.function(jit_compile=True)
    def nn_train(self, optimizer, X, F, U, F_norm, Xmin, Xmax):
        """
        Backward pass of the DeepONet, using the Adam optimizer.

        Args:
            optimizer (tf function): Adam optimizer.
            X (Tensor object of ndarray): 2D array of the input spatial coordinates. Used in the Trunk network.
            F (Tensor object of ndarray): 2D array of the input functions. Used in the Branch network.
            U (Tensor object of ndarray): Solution of the PDE.
            F_norm (Tensor object of ndarray): Masking layer for the Branch network. 1 for spaces inside the target domain, 
            0 for spaces outside the target domain.
            Xmin (Tensor object of ndarray): Minimum value of the input spatial coordinates.
            Xmax (Tensor object of ndarray): Maximum value of the input spatial coordinates.

        Returns:
            Tuple of Dictionaries: Returns a tuple of dictionaries. The first dictionary contains the loss
            and the predicted solution. The second contains the weights and biases of the model.
        """
        with tf.GradientTape() as tape:
            u_pred = self.call(X, F, F_norm, Xmin, Xmax)
            loss = tf.reduce_mean(tf.square(U - u_pred) / (tf.square(U) + 1e-4))

        gradients = tape.gradient(loss, self.train_vars())
        optimizer.apply_gradients(zip(gradients, self.train_vars()))

        loss_dict = {"loss": loss, "U_pred": u_pred}
        return loss_dict, self.W_b_dict
