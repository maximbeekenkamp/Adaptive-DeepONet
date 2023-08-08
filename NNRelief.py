import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys

from DeepONet import DeepONet
def pruning(network: DeepONet, load_bool, w_b_dict, F, X_tr, Xmin, Xmax, stride, a_conv=0.9, a_fc=0.95): #maybe different alpha for trunk network?
    """
    NNRelief pruning algorithm wrapper function. Goes through our w_b_dict and prunes each 
    layer in the network sequentially. As our w_b_dict contains both the weights and biases for 
    each layer the format of the dictionary is two keys per layer, meaning that to iterate through 
    our layers we have to go through two steps at a time. 
    
    As we have both a branch and trunk network in a DeepONet, we need to keep track of which network 
    we are pruning and what input data it requires and in what format. After identifying what type of 
    layer we are pruning, we call the corresponding pruning function for that layer type, as conv and
    fc layers require different pruning functions.

    Args:
        network (DeepONet): DeepONet object containing the branch and trunk forward pass functions.
        w_b_dict (dict): Dictionary containing the weights and biases for each layer in the network.
        Formatted as: {"W_1 conv br": w_1, "b_1 conv br": b_1, "W_2 conv br": w_2, "b_2 conv br": b_2, ...}
        F (nd.array): Input data for the branch network.
        X_tr (nd.array): Input data for the trunk network. Note that this is not the same as X_prune which is
        a local variable inside this file. X_prune dynamically changes content and shape to be the correct input 
        data for whatever type of layer we are pruning.
        Xmin (nd.array): Minimum value of the trunk input data.
        Xmax (nd.array): Maximum value of the trunk input data.
        stride (nd.array): Stride used in the conv layers.
        a_conv (float, optional): Hyperparameter alpha for convolutional pruning, sets the pruning 
        threshold. Defaults to 0.9.
        a_fc (float, optional): Hyperparameter alpha for fnn pruning, sets the pruning threshold. 
        Defaults to 0.95.

    Returns:
        w_b_dict (dict): Updated dictionary where pruned weights and biases have been set to zero.
        percent_pruned (float): Percentage of weights pruned in the network. This is an important
        performance metric for the algorithm, allowing us to know how much of the network has been
        pruned. Thereby allowing us to repopulate the network with new neurons for different geometries.
    """    
    counter = 0
    total_w_count = 0
    X_prune = F
    item_list = list(w_b_dict.items())
    if load_bool:
        item_list = item_list[3:]

    num_layers = len(item_list) // 2  # Number of layers in the network

    for i in range(num_layers):
        weight_key = item_list[i * 2][0]
        bias_key = item_list[(i * 2) + 1][0]

        temp_w_var = item_list[i * 2][1]
        temp_b_var = item_list[(i * 2) + 1][1]

        # If the layer is a list, then its a fully connected layer, with the list outlining 
        # the number of neurons in each layer. See main.py for specific architecture.
        if isinstance(temp_w_var, list): 
            weight = temp_w_var
            bias = temp_b_var
        else:
            weight = np.array(temp_w_var)
            bias = np.array(temp_b_var) 

        w = weight
        b = bias

        # See how dict is formatted, top of model_run.py or top of model_train.py
        if "W_1" in weight_key and "br" in weight_key:
            X_prune = F 
        # Layer 5 is at the boundary between our last convolutional layer, and our first fully connected layer.
        elif "W_5" in weight_key and "br" in weight_key:
            X_prune = network.flatten_layer(X_prune)
        elif "W_1" in weight_key and "tr" in weight_key:
            X_prune = X_tr

        if "conv" in weight_key:
            w, b, count = conv_pruning(network, weight, bias, X_prune, stride, a_conv)
            counter += count
            total_w_count += (w.shape[-2] * w.shape[-1])
            print("######### NNRELIEF line 82 #########")
            print("layer", weight_key)
            print("percent pruned (conv only)", (count / total_w_count)*100)
            print("####################################")
            X_prune, _, _ = network.conv_layer(X_prune, weight, bias, stride)
            X_prune = network.avg_pool(X_prune, 2, 2)
        elif "fc" in weight_key:
            w, b, count = fc_pruning(weight, bias, X_prune, a_fc)
            counter += count
            # w_count = 1
            # w_count *= tensor.shape[1] for tensor in w
            # total_w_count += w_count
            w_count = 1
            w_count = tf.reduce_prod([tensor.shape[1] for tensor in w])
            print("######### NNRELIEF line 96 #########")
            print("w_count", w_count)
            print("w_count correct", 180*80*80)
            print("####################################")
            total_w_count += tf.reduce_prod([tensor.shape[1] for tensor in w])
            print("######### NNRELIEF line 101 #########")
            print("layer", weight_key)
            print("percent pruned (fc only)", (count / total_w_count)*100)
            print("####################################")
            if "br" in weight_key:
                X_prune = network.fnn_B(weight, bias, X_prune) 
            if "tr" in weight_key:
                X_prune = network.fnn_T(weight, bias, X_prune, Xmin, Xmax)
            
        w_b_dict[weight_key] = w   
        w_b_dict[bias_key] = b
        
    percent_pruned = counter / total_w_count
    
    return w_b_dict, percent_pruned

"""
Implementation of NNRelief algorithm:
Dekhovich A, Tax DM, Sluiter MH, Bessa MA (2021) Neural network relief: a pruning algorithm based on neural activity. arXiv:2109.10795
Pseudocode:
Fully connected layers pruning
1: function FC_PRUNING(network, X, a)
2:      X^{(0)} <- X
3:      for every fc_layer l in FC_Layers do
4:          X^{(l)} <- fc_layer(X^{(l−1)})
5:          for every neuron j in fc_layer l do
6:              compute importance scores s_{ij}^{(l)} for every incoming connection i using importance score algorithm.
7:              shat_{ij}^{(l)} = Sort(s_{ij}^{(l)}, order = descending)
8:              p_0 = min{p:\sum_{i=1}^{p} shat_{ij}^{(l)} >= a}
9:              prune connections with importance scores s_{ij}^{(l)} < shat_{ij}^{(l)}
10:         end for
11:     end for 
12:     return pruned network
13: end function

Fully connected layers Importance score Algorithm, formatted in Latex to allow for copy paste:
Assume that we have a pruning set $X^{(l-1)}={x_1^{(l-1)}, \dots, x_N^{(l-1)}}$ with $N$ samples, where each datapoint 
$x_n^{(l-1)} =(x_{n1}^{(l-1)} ,\dots,x_{nm_{l-1}}^{(l-1)}) \in \mathbb{R^{m_{l-1}}}$
is the input for layer $l−1$ with dimension $m_{l−1}$ and where ${1 \leq l \leq L}$. 
We define the importance of the connection between neuron $i$ of layer $l-1$ and neuron $j$ of layer $l$ as:
$$s_{ij}^{(l)} = \frac{\overline{\left | w_{ij}^{(l)} x_{i}^{(l-1)}\right |}}
{\sum_{k=1}^{m_{l-1}}\overline{\left | w_{kj}^{(l)} x_{k}^{(l-1)}\right |} + \left | b_{j}^{(l)} \right |}$$
where 
$\overline{\left | w_{ij}^{(l)} x_{i}^{(l-1)}\right |} = \frac{1}{N}\sum_{n=1}^{N}\left | w_{ij}^{(l)} x_{ni}^{(l-1)}\right |$  
and $w_{ij}^{(l)}$ is the corresponding weight between neurons $i$ and $j$ and $b_{j}^{(l)}$ is the bias associated to neuron $j$. 
The importance score for the bias of neuron $j$ is 
$s_{m_{l-1}+1,j}^{(l)} = \frac{ \left | b_{j}^{(l)} \right |}
{\sum_{k=1}^{m_{l-1}}\overline{\left | w_{kj}^{(l)} x_{k}^{(l-1)}\right |} + \left | b_{j}^{(l)} \right |}$ . 
The denominator corresponds to the total importance in the neuron $j$ of layer $l$ that we denote as 
$S_{j}^{(l)} =\sum_{k=1}^{m_{l-1}}\overline{\left | w_{kj}^{(l)} x_{k}^{(l-1)}\right |} + \left | b_{j}^{(l)} \right |, 1 \leq j \leq m_l$.
"""
def fc_pruning(weight, bias, X_prune, a=0.95):
    """
    Prunes a fully connected layer using importance scores as proposed in the paper referenced above.

    Args:
        weight (list): List containing weight matrices of every layer. Where each list element has 
        shape (m_{l-1}, m_l), where m_{l-1} is the dimension of the l-1 th layer and m_l is the 
        dimension of the lth layer.
        bias (list): List containg bias vectors for each layer. Where each list element has shape 
        (1, m_l), where L is the m_l is the dimension of the lth layer.
        X_prune (nd.array): Pruning set of the l-1 th layer. Has shape (N, m_{l-1}), where N is the 
        number of samples in the pruning set and m_{l-1} is the dimension of the l-1 th layer, .
        a (float, optional): Hyperparameter alpha for fnn pruning, sets the pruning threshold. 
        Defaults to 0.95 as suggested by Dekhovich et al.

    Returns:
        weight (list): Pruned weight list. 
        bias (list): Pruned bias list.
        fc_count (int): Number of connections pruned in the lth fully connected layer. Used to update the
        total number of connections pruned in the entire network inside the pruning() function.
    """    
    
    fc_count = 0
    for l in range(len(weight)):
        w = weight[l]
        scores = tf.zeros_like(w)
        w_abs = tf.abs(w)
        b = bias[l] # input bias is a row vector, we need a column vector
        b_abs = tf.abs(tf.reshape(tf.transpose(b), [b.shape[1],]))
        X_prune_abs = tf.abs(X_prune)
        numerator = tf.einsum('ni,ij->nij', X_prune_abs, w_abs)
        scores = tf.reduce_mean(numerator, axis=0)
        # for c_out in tqdm.tqdm(range(len(w[1])), desc=f"FC layer {l} importance scores"):
        #     for c_in in range(len(w[0])):
        #         for n in range(X_prune.shape[0]):
        #             numerator = tf.abs(w[c_in, c_out] * X_prune[n, c_in])
        #             numerator = tf.reshape(numerator, [1,])
        #             scores[l] = tf.tensor_scatter_nd_add(scores[l], [[c_in, c_out]], numerator)
        # scores[l] /= X_prune.shape[0]
        
        S_c_out = (tf.reduce_sum((scores), axis=0) + b_abs) # reducing over c_in
        
        scores_w = scores / S_c_out
        scores_b = b_abs / S_c_out

        scores_sorted_w = tf.sort(scores_w, axis=0, direction='DESCENDING')
        arg_scores_sorted_w = tf.argsort(scores_w, axis=0, direction='DESCENDING')
        scores_sorted_b = tf.sort(scores_b, axis=0, direction='DESCENDING')
        arg_scores_sorted_b = tf.argsort(scores_b, axis=0, direction='DESCENDING')
        
        # pruning, calculate the index i where the cumulative sum of the scores is >= a
        w_mask = w_mask_fn(scores_sorted_w, a)
        pruned_scores_b = b_pruned_scores(scores_sorted_b, a)

        if scores_w.shape[0] > 10:
            # w = tf.transpose(w)
            # w_0 = []
            indices_w_list = [] 
            row_starts = []
            cum_sum = 0
            for c_out in tqdm(range(scores_sorted_w.shape[1]), desc=f"FC layer {l} pruning"):
                indices_w = arg_scores_sorted_w[:, c_out][w_mask[:, c_out]]
                # indices_w = tf.expand_dims(indices_w, axis=1)
                # indices_w = tf.transpose(indices_w)
                # w_0.append(w[0,c_out]) # because padding sets values to zero, the weight will at index zero will be set to zero, so we save it and set it back after pruning
                # indices_w = tf.pad(indices_w, [[w.shape[0] - indices_w.shape[0], 0], [0, 0]], constant_values=0)
                # assert indices_w.shape[0] == w.shape[0], f"Shape mismatch, indices_w.shape[0] = {indices_w.shape[0]}, w.shape[0] = {w.shape[0]}"
                indices_w_list.append(indices_w)
                row_starts.append(cum_sum)
                cum_sum += indices_w.shape[0]
            indices_w_list = tf.concat(indices_w_list, axis=0)
            indices_w_list = tf.RaggedTensor.from_row_starts(indices_w_list, row_starts=row_starts)
            indices_w_list = indices_w_list.to_tensor()
            indices_w_list = tf.transpose(indices_w_list)
            w = tf.tensor_scatter_nd_update(w, indices_w_list, tf.zeros_like(indices_w_list, dtype=tf.float64))
            # w[c_out] = tf.reshape(w[c_out], [w[c_out].shape[1],])
            # for c_out in range(scores_sorted_w.shape[1]):
            #     if scores_w[0,c_out] != 0:
            #         w = tf.tensor_scatter_nd_update(w, [[0, c_out]], [w_0[c_out]])
            weight[l] = w

        if scores_b.shape[0] > 10:
            indices_b = arg_scores_sorted_b[(scores_b.shape[0]-pruned_scores_b.shape[0]):]
            bias[l] = tf.tensor_scatter_nd_update(bias[l], tf.expand_dims(indices_b, axis=0), [0.0])

        X_prune = tf.nn.leaky_relu(tf.add(tf.matmul(X_prune, w), b))
    return weight, bias, fc_count


"""
Algorithm 2: Convolutional layers pruning
(Same paper as above, arXiv:2109.10795)

Our pruning approach for convolutional layers is similar to the one conducted on fully connected layers. 
We consider kernels and a bias in a particular filter as contributors to the signal produced by this filter.

1: function CONV_PRUNING(network, X, a)
2:      X^{(0)} <- X
3:      for every conv_layer l in CONV_Layers do
4:          X^{(l)} <- conv_layer(X^{(l−1)})
5:          for every filter F_j in conv_layer l do
6:              compute importance scores s_{ij}^{(l)} for all kernel K_{ij}^{(l)} and bias b_j^{(l)} in filter F_j^{(l)} using different importance score algorith below.
7:              shat_{ij}^{(l)} = Sort(s_{ij}^{(l)}, order = descending)
8:              p_0 = min{p:\sum_{i=1}^{p} shat_{ij}^{(l)} >= a}
9:              prune kernel K_{ij}^{(l)} with importance scores s_{ij}^{(l)} < shat_{ij}^{(l)}
10:         end for
11:     end for 
12:     return pruned network
13: end function

Convolutional layers Importance Algorithm, formatted in Latex to allow for copy paste:
Assume that we have $m_{l-1}$-channeled input samples $X^{(l-1)}={x_1^{(l-1)}, \dots, x_N^{(l-1)}}$, where 
$x_k^{(l-1)} =(x_{k1}^{(l-1)} ,\dots,x_{km_{l-1}}^{(l-1)}) \in \mathbb{R}^{m_{l-1} \times h_{l-1}^1 \times h_{l-1}^2}$, 
where $h_{l-1}^1$ and $h_{l-1}^2$ are the height and width of input images (or feature maps) for convolutional layer $l$. 
For every kernel $K_{1j}^{(l)}, K_{2j}^{(l)}, \dots, K_{m_{l}j}^{(l)}$, $
K_{ij}^{(l)} = (k_{ijqt}^{(l)}) \in \mathbb{R}^{r_{l} \times r_{l}}, 1 \leq q, t \leq r_l$, $r_l$ is a kernel size, 
and a bias $b_{j}^{(l)}$ in a filter $K_{j}^{(l)}$, we define 
$\hat{K}_{ij}^{(l)} = \left ( \left | k_{ijqt}^{(l)} \right | \right )$ as a matrix consisting of the absolute values of $K_{ij}^{(l)}$. \\
Then we compute the importance scores $s_{ij}^{(l)}, i \in {1,2,\dots m_l}$ of kernel $K_{j}^{(l)}$ as follows:
$$s_{ij}^{(l)} = \frac{\frac{1}{N}\sum_{n=1}^{N}\left \| \hat{K}_{ij}^{(l)}\ast \left | x_{ni}^{(l-1)} \right | \right \|_F}{S_{j}^{(l)}},$$
$$s_{m_{l}+1,j}^{(l)} = \frac{\left | b_j^{(l)} \right |\sqrt{h_{l}^{1}h_{l}^{2}})}{S_{j}^{(l)}},$$
where 
$S_{j}^{(l)}=\left (\sum_{n=1}^{N}\left \| \hat{K}_{ij}^{(l)}\ast \left | x_{ni}^{(l-1)} \right | \right \|_F\right ) + \left | b_j^{(l)} \right |\sqrt{h_{l}^{1}h_{l}^{2}}$ 
is the total importance score in filter $F_j^{(l)}$ of layer $l$,and where $\ast$ indicates a convolution operation,and $\left \| \cdot  \right \|_F$ the Frobenius norm. \\
In the equation starting $s_{ij}^{(l)} = \dots$ we compute the amount of information that every kernel produces on average, analogously to what we do in fully connected layers.
Importantly, note that $\ast$ is a convolution operation, not multiplication.

"""
def conv_pruning(network:DeepONet, weight, bias, X_prune, stride, a=0.9):
    """
    Prunes a convolutional layer using the algorithm described in the paper above.

    Args:
        network (DeepONet): DeepONet object containing the convolutional forward pass function.
        weight (nd.array): Weight matrix of the convolutional layer. Shape: (kernel_height, kernel_width, num_channels_in, num_channels_out)
        bias (nd.array): Bias vector of the convolutional layer. Shape: (num_kernels,)
        X_prune (nd.array): Input data to the convolutional layer. Shape: (num_samples, kernel_height, kernel_width, num_channels)
        stride (nd.array): Stride of the convolutional layer. Shape: (2,)
        a (float, optional): Hyperparameter alpha for convolutional pruning, sets the pruning 
        threshold. Defaults to 0.9 as suggested by Dekhovich et al.

    Returns:
        weight (nd.array): Pruned weight matrix of the convolutional layer, where all pruned 
        kernels have been set to zero. Shape unchanged.
        bias (nd.array): Pruned bias vector of the convolutional layer, where all pruned kernels
        have been set to zero. Shape unchanged.
        conv_count (int): Number of filters that have been pruned. Added to the total count of
        pruned weights in the pruning() function.
    """    
    scores = tf.zeros(([weight.shape[-2], weight.shape[-1]]), dtype=tf.float64)
    X_prune = tf.abs(X_prune)
    w_abs = tf.abs(weight)

    for c_out in tqdm(range(weight.shape[-1]), desc="CONV layer importance scores"):
        for c_in in range(weight.shape[-2]):
            conv_pass, _, _ = network.conv_layer(x=X_prune[:,:,:,c_in:c_in+1], 
                                                 w=w_abs[:,:,c_in:c_in+1,c_out:c_out+1], 
                                                 b=bias[c_out], 
                                                 stride=stride)
            conv_pass = tf.reduce_sum(conv_pass, axis=0, keepdims=True)
            conv_pass = tf.reshape(conv_pass, [conv_pass.shape[1], conv_pass.shape[2]])
            conv_pass = tf.reshape((tf.norm(conv_pass, ord='fro', axis= [0,1])), [1,])
            scores = tf.tensor_scatter_nd_add(scores, [[c_in, c_out]], conv_pass)

    scores /= X_prune.shape[0]
    bias_times_sqrt = tf.abs(bias) * tf.sqrt(tf.cast(X_prune.shape[1] * X_prune.shape[2], tf.float64))

    S_c_out = (tf.reduce_sum(scores, axis=0) + bias_times_sqrt)
    
    scores_w = scores / S_c_out
    scores_b = tf.abs(bias) * bias_times_sqrt / S_c_out

    scores_sorted_w = tf.sort(scores_w, axis=0, direction='DESCENDING')
    arg_scores_sorted_w = tf.argsort(scores_w, axis=0, direction='DESCENDING')
    scores_sorted_b = tf.sort(scores_b, axis=0, direction='DESCENDING')
    arg_scores_sorted_b = tf.argsort(scores_b, axis=0, direction='DESCENDING')
    
    # pruning, calculates the scores where the cumulative sum of the scores is >= a
    # if the number of kernels/neurons on layer l-1 is too few, then pruning becomes non-effective
    if scores_sorted_w.shape[0] > 10:
        w_mask = w_mask_fn(scores_sorted_w, a)

    if scores_sorted_b.shape[0] > 10: 
        pruned_scores_b = b_pruned_scores(scores_sorted_b, a)

    conv_count = 0
    
    if scores_w.shape[0] > 10:
        for c_out in tqdm(range(scores_sorted_w.shape[1]), desc="CONV layer pruning"):
            indices_w = arg_scores_sorted_w[:, c_out][w_mask[:, c_out]]
            weight[:, :, indices_w, c_out] = 0.0
            conv_count += indices_w.shape[0]
    if scores_b.shape[0] > 10:
        for c_out in range(pruned_scores_b.shape[0]):
            indices_b = arg_scores_sorted_b[c_out]
            bias[indices_b] = 0.0

    return weight, bias, conv_count

def w_mask_fn(scores_sorted, a):
    """
    Returns the scores of the kernels that are pruned, based on the 
    importance scores of the kernels and the pruning threshold a.

    Args:
        scores_sorted (Tensor): Tensor containing the scores of the 
        kernels / weights, sorted in descending order.
        a (float): Threshold for pruning. Typically 0.9 for convolutional 
        layers and 0.95 for fully connected layers.

    Returns:
        scores_to_prune (Tensor): Tensor containing only the scores 
        of the kernels / weights that will be pruned. By matching the 
        scores_to_prune with the scores_sorted, we can find the indices
        of the kernels that will be pruned.
    """      
    cumulative_sum = tf.cumsum(scores_sorted, axis=0)
    mask = tf.math.greater(cumulative_sum, a) # mask[-1] will always be True, since the cumulative sum will always be = 1")
    return mask

def b_pruned_scores(scores_sorted, a):
    """
    Returns the scores of the biases that are to be pruned, based on the
    importance scores of the biases and the pruning threshold a.

    Args:
        scores_sorted (Tensor): Tensor containing the scores of the
        biases, sorted in descending order.
        a (float): Threshold for pruning. Typically 0.9 for convolutional
        layers and 0.95 for fully connected layers.

    Returns:
        scores_to_prune (Tensor): Tensor containing only the scores
        of the biases that will be pruned. By matching the
        scores_to_prune with the scores_sorted, we can find the indices
        of the biases that will be pruned.
    """    
    cumulative_sum = tf.cumsum(scores_sorted, axis=0)
    mask = tf.math.greater(cumulative_sum, a)
    return scores_sorted[mask]
