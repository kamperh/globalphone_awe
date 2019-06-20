"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018, 2019
"""

import numpy.testing as npt
from tflego import *


#-----------------------------------------------------------------------------#
#                          NUMPY EQUIVALENT FUNCTIONS                         #
#-----------------------------------------------------------------------------#

def np_linear(x, W, b):
    return np.dot(x, W) + b


def np_relu(x):
    return np.maximum(0., x)


def np_conv2d(x, filters, padding="valid"):
    """
    Calculate the convolution of `x` using `filters`.
    
    A useful tutorial: http://www.robots.ox.ac.uk/~vgg/practicals/cnn/.
    
    Parameters
    ----------
    x : matrix [n_data, height, width, d_in]
    filters : matrix [filter_height, filter_width, d_in, d_out]
    """

    import scipy.signal

    # Dimensions
    n_data, height, width, d_in = x.shape
    filter_height, filter_width, _, d_out = filters.shape
    assert d_in == _
    
    # Loop over data
    conv_over_data = []
    for i_data in range(n_data):
        # Loop over output channels
        conv_over_channels = []
        for i_out_channel in range(d_out):
            conv_result = 0.
            # Loop over input channels
            for i_in_channel in range(d_in):
                conv_result += scipy.signal.correlate(
                    x[i_data, :, :, i_in_channel], filters[:, :, i_in_channel,
                    i_out_channel], mode=padding
                    )
            conv_over_channels.append(conv_result)
        conv_over_data.append(
            np.transpose(np.array(conv_over_channels), (1, 2, 0))
            )
    
    return np.array(conv_over_data)


def np_maxpool2d(x, pool_shape, ignore_border=False):
    """
    Performs max pooling on `x`.
    
    Parameters
    ----------
    x : matrix [n_data, height, width, d_in]
        Input over which pooling is performed.
    pool_shape : list
        Gives the pooling shape as (pool_height, pool_width).
    """
    
    # Dimensions
    n_data, height, width, d_in = x.shape
    pool_height, pool_width = pool_shape
    round_func = np.floor if ignore_border else np.ceil
    output_height = int(round_func(1.*height/pool_height))
    output_width = int(round_func(1.*width/pool_width))

    # Max pool
    max_pool = np.zeros((n_data, output_height, output_width, d_in))
    for i_data in range(n_data):
        for i_channel in range(d_in):
            for i in range(output_height):
                for j in range(output_width):
                    max_pool[i_data, i, j, i_channel] = np.max(x[
                        i_data,
                        i*pool_height:i*pool_height + pool_height,
                        j*pool_width:j*pool_width + pool_width,
                        i_channel
                        ])
    
    return max_pool


def np_cnn(x, input_shape, weights, biases, pool_shapes):
    """
    Push the input `x` through the CNN with `cnn_specs` matching the parameters
    passed to `build_cnn`, `weights` and `biases` the parameters of each
    convolutional layer.
    """
    cnn = x.reshape(input_shape)
    for W, b, pool_shape in zip(weights, biases, pool_shapes):
        if pool_shape is not None:
            cnn = np_relu(np_maxpool2d(np_conv2d(cnn, W) + b, pool_shape))
        else:
            cnn = np_relu(np_conv2d(cnn, W) + b)
    return cnn


def np_rnn(x, x_lengths, W, b, maxlength=None):
    """Calculates the output for a basic RNN."""
    if maxlength is None:
        maxlength = max(x_lengths)
    outputs = np.zeros((x.shape[0], maxlength, W.shape[1]))
    for i_data in range(x.shape[0]):
        cur_x_sequence = x[i_data, :x_lengths[i_data], :]
        prev_state = np.zeros(W.shape[1])
        for i_step, cur_x in enumerate(cur_x_sequence):
            cur_state = np.tanh(np.dot(np.hstack((cur_x, prev_state)), W) + b)
            outputs[i_data, i_step, :] = cur_state
            prev_state = cur_state
    return outputs


def np_multi_rnn(x, x_lengths, weights, biases, maxlength=None):
    """
    Push the input `x` through the RNN. The `weights`
    and `biases` should be lists of the parameters of each layer.
    """
    for W, b in zip(weights, biases):
        x = np_rnn(x, x_lengths, W, b, maxlength)
    return x


def np_encdec_lazydynamic(x, x_lengths, W_encoder, b_encoder, W_decoder,
        b_decoder, W_output, b_output, maxlength=None):

    if maxlength is None:
        maxlength = max(x_lengths)

    # Encoder
    encoder_output = np_rnn(x, x_lengths, W_encoder, b_encoder, maxlength)
    encoder_states = []
    for i_data, l in enumerate(x_lengths):
        encoder_states.append(encoder_output[i_data, l - 1, :])
    encoder_states = np.array(encoder_states)

    # Decoder

    # Repeat encoder states
    n_hidden = W_encoder.shape[-1]
    decoder_input = np.reshape(
        np.repeat(encoder_states, maxlength, axis=0), [-1, maxlength, n_hidden]
        )
    
    # Decoding RNN
    decoder_output = np_rnn(
        decoder_input, x_lengths, W_decoder, b_decoder, maxlength
        )

    # Final linear layer
    decoder_output_linear = np.zeros(x.shape)
    decoder_output_list = []
    for i_data in range(x.shape[0]):
        cur_decoder_sequence = decoder_output[i_data, :x_lengths[i_data], :]
        cur_decoder_list = []
        for i_step, cur_decoder in enumerate(cur_decoder_sequence):
            output_linear = np_linear(
                cur_decoder, W_output, b_output
                )
            decoder_output_linear[i_data, i_step, :] = output_linear
            cur_decoder_list.append(output_linear)
        decoder_output_list.append(np.array(cur_decoder_list))
    decoder_output = decoder_output_linear

    return encoder_states, decoder_output, decoder_output_list


#-----------------------------------------------------------------------------#
#                                TEST FUNCTIONS                               #
#-----------------------------------------------------------------------------#

def test_cnn():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 28*28
    n_data = 3
    test_data = np.asarray(np.random.randn(n_data, n_input), dtype=NP_DTYPE)

    # Model parameters
    input_shape = [-1, 28, 28, 1]  # [n_data, height, width, d_in]
    filter_shapes = [
        [5, 5, 1, 32],
        [5, 5, 32, 64]
        ]
    pool_shapes = [
        [2, 2], 
        [2, 2]
        ]

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, n_input])
    cnn = build_cnn(x, input_shape, filter_shapes, pool_shapes, padding="VALID")
    cnn = tf.contrib.layers.flatten(cnn)
    with tf.variable_scope("cnn_layer_0", reuse=True):
        W_0 = tf.get_variable("W")
        b_0 = tf.get_variable("b")
    with tf.variable_scope("cnn_layer_1", reuse=True):
        W_1 = tf.get_variable("W")
        b_1 = tf.get_variable("b")

    # TensorFlow graph
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        
        # Output
        tf_output = cnn.eval({x: test_data})
        
        # Parameters
        W_0 = W_0.eval()
        b_0 = b_0.eval()
        W_1 = W_1.eval()
        b_1 = b_1.eval()

    # NumPy model
    np_output = np_cnn(
        test_data, input_shape, [W_0, W_1], [b_0, b_1], pool_shapes
        )
    np_output = np_output.reshape(np_output.shape[0], -1)

    npt.assert_almost_equal(tf_output, np_output, decimal=5)


def test_conv2d_transpose():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 4*4*5
    n_data = 2
    test_data = np.asarray(np.random.randn(n_data, n_input), dtype=NP_DTYPE)

    # Model parameters
    input_shape = [-1, 4, 4, 5]  # [n_data, height, width, d_in]
    filter_shape = [3, 3, 2, 5]  # [filter_height, filter_width, d_out, d_in]

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, n_input])
    x_reshaped = tf.reshape(x, input_shape)
    i_layer = 0
    with tf.variable_scope("cnn_transpose_layer_{}".format(i_layer)):
        conv_transpose = build_conv2d_transpose(
            x_reshaped, filter_shape, activation=tf.identity
            )
    with tf.variable_scope("cnn_transpose_layer_0", reuse=True):
        W_0 = tf.get_variable("W")
        b_0 = tf.get_variable("b")

    # TensorFlow graph
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        # Output
        tf_output = conv_transpose.eval({x: test_data})

        # Parameters
        W_0 = W_0.eval()
        b_0 = b_0.eval()

    # NumPy model
    x_np = test_data.reshape(input_shape)
    W_dash = np.rot90(W_0, k=2)
    W_dash = np.transpose(W_dash, (0, 1, 3, 2))
    np_output = np_conv2d(x_np, W_dash, padding="full") + b_0

    npt.assert_almost_equal(tf_output, np_output, decimal=5)


def test_rnn():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 10
    n_data = 11
    n_maxlength = 12
    test_data = np.zeros((n_data, n_maxlength, n_input), dtype=NP_DTYPE)
    lengths = []
    for i_data in range(n_data):
        length = np.random.randint(1, n_maxlength + 1)
        lengths.append(length)
        test_data[i_data, :length, :] = np.random.randn(length, n_input)
    lengths = np.array(lengths, dtype=NP_ITYPE)

    # Model parameters
    n_hidden = 13
    rnn_type = "rnn"

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, n_maxlength, n_input])
    x_lengths = tf.placeholder(TF_DTYPE, [None])
    rnn_outputs, rnn_states = build_rnn(
        x, x_lengths, n_hidden, rnn_type=rnn_type
        )
    with tf.variable_scope("rnn/basic_rnn_cell", reuse=True):
        W = tf.get_variable("kernel")
        b = tf.get_variable("bias")

    # TensorFlow graph
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        
        # Output
        tf_output = rnn_outputs.eval({x: test_data, x_lengths: lengths})
        
        # Weights
        W = W.eval()
        b = b.eval()

    # NumPy model
    np_output = np_rnn(test_data, lengths, W, b, n_maxlength)

    npt.assert_almost_equal(tf_output, np_output, decimal=5)


def test_multi_rnn():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 10
    n_data = 11
    n_maxlength = 12
    test_data = np.zeros((n_data, n_maxlength, n_input), dtype=NP_DTYPE)
    lengths = []
    for i_data in range(n_data):
        length = np.random.randint(1, n_maxlength + 1)
        lengths.append(length)
        test_data[i_data, :length, :] = np.random.randn(length, n_input)
    lengths = np.array(lengths, dtype=NP_ITYPE)

    # Model parameters
    n_hiddens = [13, 14]
    rnn_type = "rnn"

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, n_maxlength, n_input])
    x_lengths = tf.placeholder(TF_DTYPE, [None])
    rnn_outputs, rnn_states = build_multi_rnn(x, x_lengths, n_hiddens, rnn_type=rnn_type)
    with tf.variable_scope("rnn_layer_0/rnn/basic_rnn_cell", reuse=True) as vs:
        W_0 = tf.get_variable("kernel")
        b_0 = tf.get_variable("bias")
    with tf.variable_scope("rnn_layer_1/rnn/basic_rnn_cell", reuse=True) as vs:
        W_1 = tf.get_variable("kernel")
        b_1 = tf.get_variable("bias")

    # TensorFlow graph
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        # Output
        tf_output = rnn_outputs.eval({x: test_data, x_lengths: lengths})

        # Weights
        W_0 = W_0.eval()
        b_0 = b_0.eval()
        W_1 = W_1.eval()
        b_1 = b_1.eval()

    # NumPy model
    np_output = np_multi_rnn(test_data, lengths, [W_0, W_1], [b_0, b_1], n_maxlength)

    npt.assert_almost_equal(tf_output, np_output, decimal=5)


def test_encdec_lazydynamic():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 4
    n_data = 3
    n_maxlength = 5
    test_data = np.zeros((n_data, n_maxlength, n_input), dtype=NP_DTYPE)
    lengths = []
    for i_data in range(n_data):
        length = np.random.randint(1, n_maxlength + 1)
        lengths.append(length)
        test_data[i_data, :length, :] = np.random.randn(length, n_input)
    lengths = np.array(lengths, dtype=NP_ITYPE)

    # Model parameters
    n_hidden = 6
    rnn_type = "rnn"

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, None, n_input])
    x_lengths = tf.placeholder(TF_ITYPE, [None])
    network_dict = build_encdec_lazydynamic(
        x, x_lengths, n_hidden, rnn_type=rnn_type
        )
    encoder_states = network_dict["encoder_states"]
    decoder_output = network_dict["decoder_output"]
    with tf.variable_scope("rnn_encoder/basic_rnn_cell", reuse=True):
        W_encoder = tf.get_variable("kernel")
        b_encoder = tf.get_variable("bias")
    with tf.variable_scope("rnn_decoder/basic_rnn_cell", reuse=True):
        W_decoder = tf.get_variable("kernel")
        b_decoder = tf.get_variable("bias")
    with tf.variable_scope("rnn_decoder/linear_output", reuse=True):
        W_output = tf.get_variable("W")
        b_output = tf.get_variable("b")

    # TensorFlow graph
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        # Output
        tf_encoder_states = encoder_states.eval(
            {x: test_data, x_lengths: lengths}
            )
        tf_decoder_output = decoder_output.eval(
            {x: test_data, x_lengths: lengths}
            )

        # Weights
        W_encoder = W_encoder.eval()
        b_encoder = b_encoder.eval()
        W_decoder = W_decoder.eval()
        b_decoder = b_decoder.eval()
        W_output = W_output.eval()
        b_output = b_output.eval()

    np_encoder_states, np_decoder_output, _ = np_encdec_lazydynamic(
        test_data, lengths, W_encoder, b_encoder, W_decoder, b_decoder,
        W_output, b_output, n_maxlength
        )

    npt.assert_almost_equal(tf_encoder_states, np_encoder_states, decimal=5)
    npt.assert_almost_equal(tf_decoder_output, np_decoder_output, decimal=5)


def test_encdec_lazydynamic_masked_loss():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 4
    n_data = 3
    n_maxlength = 5
    test_data = np.zeros((n_data, n_maxlength, n_input), dtype=NP_DTYPE)
    test_data_list = []
    lengths = []
    for i_data in range(n_data):
        length = np.random.randint(1, n_maxlength + 1)
        lengths.append(length)
        seq = np.random.randn(length, n_input)
        test_data[i_data, :length, :] = seq
        test_data_list.append(seq)
    lengths = np.array(lengths, dtype=NP_ITYPE)

    # Model parameters
    n_hidden = 6
    rnn_type = "rnn"

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, None, n_input])
    x_lengths = tf.placeholder(TF_ITYPE, [None])
    network_dict = build_encdec_lazydynamic(
        x, x_lengths, n_hidden, rnn_type=rnn_type
        )
    encoder_states = network_dict["encoder_states"]
    decoder_output = network_dict["decoder_output"]
    mask = network_dict["mask"]
    with tf.variable_scope("rnn_encoder/basic_rnn_cell", reuse=True):
        W_encoder = tf.get_variable("kernel")
        b_encoder = tf.get_variable("bias")
    with tf.variable_scope("rnn_decoder/basic_rnn_cell", reuse=True):
        W_decoder = tf.get_variable("kernel")
        b_decoder = tf.get_variable("bias")
    with tf.variable_scope("rnn_decoder/linear_output", reuse=True):
        W_output = tf.get_variable("W")
        b_output = tf.get_variable("b")

    # TensorFlow loss
    loss = tf.reduce_mean(
        tf.reduce_sum(tf.reduce_mean(tf.square(x - decoder_output), -1), -1) /
        tf.reduce_sum(mask, 1)
        )  # https://danijar.com/variable-sequence-lengths-in-tensorflow/

    # TensorFlow graph
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        # Output
        tf_encoder_states = encoder_states.eval(
            {x: test_data, x_lengths: lengths}
            )
        tf_decoder_output = decoder_output.eval(
            {x: test_data, x_lengths: lengths}
            )
        tf_loss = loss.eval({x: test_data, x_lengths: lengths})

        # Weights
        W_encoder = W_encoder.eval()
        b_encoder = b_encoder.eval()
        W_decoder = W_decoder.eval()
        b_decoder = b_decoder.eval()
        W_output = W_output.eval()
        b_output = b_output.eval()

    _, _, np_decoder_list = np_encdec_lazydynamic(
        test_data, lengths, W_encoder, b_encoder, W_decoder, b_decoder,
        W_output, b_output, n_maxlength
        )

    # NumPy loss
    losses = []
    for i_data, x_seq in enumerate(test_data_list):
        y_seq = np_decoder_list[i_data]
        mse = ((y_seq - x_seq)**2).mean()
        losses.append(mse)
    np_loss = np.mean(losses)

    npt.assert_almost_equal(tf_loss, np_loss, decimal=5)


def test_vq():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    D = 5
    K = 3
    test_data = np.random.randn(4, D)

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, D])
    vq = build_vq(x, K, D)
    embeds = vq["embeds"]
    z_q = vq["z_q"]

    # TensorFlow graph
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        tf_embeds = embeds.eval()
        tf_z_q = z_q.eval({x: test_data})

    # NumPy equivalent
    np_z_q = []
    for x_test in test_data:
        dists = []
        for embed in tf_embeds:
            dists.append(np.linalg.norm(x_test - embed))
        np_z_q.append(tf_embeds[np.argmin(dists)])

    npt.assert_almost_equal(tf_z_q, np_z_q)
