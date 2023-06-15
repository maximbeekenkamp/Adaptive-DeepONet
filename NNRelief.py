import tensorflow as tf
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

Fully connected layers Importance score Algorithm:
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
def fc_pruning(network, X, a):
    """
    NNRelief pruning algorithm for fully connected layers.

    Args:
        network: model to be pruned
        X: input data (pruning set from layer l-1)
        a: pruning threshold (hyperparameter)

    Returns:
        network: pruned model
    """
    # for each layer l in the network
    for l in range(len(network.layers)):
        # Takes the input data from layer l-1 and computes the output of layer l by applying the weights and biases of layer l
        # X has shape (batch_size, num_neurons)
        X = network.layers[l](X)  

        # X.shape[1] is the number of neurons in the current layer
        for j in range(X.shape[1]):
            scores = tf.zeros(X.shape[0])
            
            # where X.shape[0] is the number of samples in the pruning set
            for i in range(X.shape[0]):
                # calculate the importance score between neuron i of layer l-1 and neuron j of layer l
                # |w_{ij}^{(l)} x_{i}^{(l-1)}|
                scores[i] = tf.abs(network.layers[l].weights[j][i] * X[i][j])
            
            # normalise the scores
            scores = scores / (tf.reduce_sum(scores) + tf.abs(network.layers[l].bias[j])) # b_{j}^{(l)}
            scores_sorted = tf.sort(scores, direction='DESCENDING')
            
            # pruning, calculate the index i where the cumulative sum of the scores is >= a
            p = 0
            cumulative_sum = 0
            for i in range(scores_sorted.shape[0]):
                cumulative_sum += scores_sorted[i]
                if cumulative_sum >= a:
                    p = i
                    break
            # scores are descending, so all indicies after p are pruned
            pruned_indices = scores_sorted.indices[p:]
            
            # Set the weight to zero for pruned indices
            # this allows the network to use this neuron for other connections
            for i in pruned_indices:
                network.layers[l].weights[j, i].assign(0.0) 

    return network
"""
Additional notes:
 - Model currently doesn't have layers feature so we can't currently iterate through the layers (i think)
 - Need to find a way to get the input data from the previous layer (formatted properly)
 - Apply single layers from the call function
"""


"""
Algorithm 2: Convolutional layers pruning

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

Convolutional layers Importance Algorithm:
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
"""
def conv_pruning(network, stride, X, a):
    """
    NNRelief pruning algorithm for convolutional layers.

    Args:
        network: model to be pruned
        X: input samples (pruning set from layer l-1)
        a: pruning threshold (hyperparameter)

    Returns:
        network: pruned model
    """
    # for each layer l in the network
    for l in range(len(network.layers)):
        # Takes the input data from layer l-1 and computes the output of layer l by applying the convolution operation
        X = network.layers[l](X)

        # iterate over each filter F_j in conv_layer l
        for j in range(X.shape[-1]):
            scores = tf.zeros((X.shape[0]))

            # iterate over each kernel K_ij and bias b_j in filter F_j
            for i in range(X.shape[-2]):
                # compute the importance score of kernel K_ij and bias b_j
                kernel = tf.abs(network.layers[l].weights[i, :, :, j])
                scores += tf.norm(tf.nn.conv2d(X[:, :, :, i], kernel, strides=stride, padding='SAME'), ord='fro', axis=[1, 2])

            # compute the importance score for the bias of filter F_j
            scores += tf.abs(network.layers[l].bias[j]) * tf.sqrt(tf.cast(X.shape[1] * X.shape[2], tf.float32))

            # normalize the scores
            scores /= tf.reduce_sum(scores)

            # sort the scores in descending order
            scores_sorted = tf.sort(scores, direction='DESCENDING')

            # find the index p where the cumulative sum of scores is >= a
            p = 0
            cumulative_sum = 0
            for i in range(scores_sorted.shape[0]):
                cumulative_sum += scores_sorted[i]
                if cumulative_sum >= a:
                    p = i
                    break

            # prune kernels with importance scores less than shat_ij
            pruned_indices = scores_sorted.indices[p:]
            for i in pruned_indices:
                network.layers[l].weights[i, :, :, j].assign(0.0)

    return network