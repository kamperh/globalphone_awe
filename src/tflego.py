"""
Neural network building blocks.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018, 2019
"""

import numpy as np
import tensorflow as tf

NP_DTYPE = np.float32
TF_DTYPE = tf.float32
NP_ITYPE = np.int32
TF_ITYPE = tf.int32


#-----------------------------------------------------------------------------#
#                                 BASIC BLOCKS                                #
#-----------------------------------------------------------------------------#

def build_linear(x, n_output):
    n_input = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        "W", [n_input, n_output], dtype=TF_DTYPE,
        initializer=tf.contrib.layers.xavier_initializer()
        )
    b = tf.get_variable(
        "b", [n_output], dtype=TF_DTYPE,
        initializer=tf.random_normal_initializer()
        )
    return tf.matmul(x, W) + b


def build_feedforward(x, n_hiddens, keep_prob=1., activation=tf.nn.relu):
    """
    Build a feedforward neural network.
    
    The final layer is linear.
    
    Parameters
    ----------
    n_hiddens : list
        Hidden units in each of the layers.
    """
    for i_layer, n_hidden in enumerate(n_hiddens):
        with tf.variable_scope("ff_layer_{}".format(i_layer)):
            x = build_linear(x, n_hidden)
            if i_layer != len(n_hiddens) - 1:
                x = activation(x)
            x = tf.nn.dropout(x, keep_prob)
    return x


#-----------------------------------------------------------------------------#
#                               RECURRENT BLOCKS                              #
#-----------------------------------------------------------------------------#

def build_rnn_cell(n_hidden, rnn_type="lstm", **kwargs):
    """
    The `kwargs` parameters are passed directly to the constructor of the cell
    class, e.g. peephole connections can be used by adding `use_peepholes=True`
    when `rnn_type` is "lstm".
    """
    if rnn_type == "lstm":
        cell_args = {"state_is_tuple": True}  # default LSTM parameters
        cell_args.update(kwargs)
        cell = tf.nn.rnn_cell.LSTMCell(n_hidden, **cell_args)
    elif rnn_type == "gru":
        cell = tf.nn.rnn_cell.GRUCell(n_hidden, **kwargs)
    elif rnn_type == "rnn":
        cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden, **kwargs)
    else:
        assert False, "Invalid RNN type: {}".format(rnn_type)
    return cell


def build_rnn(x, x_lengths, n_hidden, rnn_type="lstm", keep_prob=1.,
        scope=None, **kwargs):
    """
    Build a recurrent neural network (RNN) with architecture `rnn_type`.
    
    The RNN is dynamic, with `x_lengths` giving the lengths as a Tensor with
    shape [n_data]. The input `x` should be padded to have shape [n_data,
    n_padded, d_in].
    
    Parameters
    ----------
    rnn_type : str
        Can be "lstm", "gru" or "rnn".
    kwargs : dict
        These are passed directly to the constructor of the cell class, e.g.
        peephole connections can be used by adding `use_peepholes=True` when
        `rnn_type` is "lstm".
    """
    
    # RNN cell
    cell = build_rnn_cell(n_hidden, rnn_type, **kwargs)
    
    # Dropout
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, input_keep_prob=1., output_keep_prob=keep_prob
        )
    
    # Dynamic RNN
    return tf.nn.dynamic_rnn(
        cell, x, sequence_length=x_lengths, dtype=TF_DTYPE, scope=scope
        )


def build_multi_rnn(x, x_lengths, n_hiddens, rnn_type="lstm", keep_prob=1.,
        **kwargs):
    """
    Build a multi-layer RNN.
    
    Apart from those below, parameters are similar to that of `build_rnn`.

    Parameters
    ----------
    n_hiddens : list
        Hidden units in each of the layers.
    """
    for i_layer, n_hidden in enumerate(n_hiddens):
        with tf.variable_scope("rnn_layer_{}".format(i_layer)):
            outputs, states = build_rnn(
                x, x_lengths, n_hidden, rnn_type, keep_prob, **kwargs
                )
            x = outputs
    return outputs, states


def build_bidirectional_rnn(x, x_lengths, n_hidden, rnn_type="lstm",
        keep_prob=1., scope=None, **kwargs):
    """
    Build a bidirectional recurrent neural network (RNN).
    
    The RNN is dynamic, with `x_lengths` giving the lengths as a Tensor with
    shape [n_data]. The input `x` should be padded to have shape [n_data,
    n_padded, d_in].
    
    Parameters
    ----------
    rnn_type : str
        Can be "lstm", "gru" or "rnn".
    kwargs : dict
        These are passed directly to the constructor of the cell class, e.g.
        peephole connections can be used by adding `use_peepholes=True` when
        `rnn_type` is "lstm".
    """

    # RNN cell
    cell_fw = build_rnn_cell(n_hidden, rnn_type, **kwargs)
    cell_bw = build_rnn_cell(n_hidden, rnn_type, **kwargs)
    
    # Dropout
    cell_fw = tf.nn.rnn_cell.DropoutWrapper(
        cell_fw, input_keep_prob=1., output_keep_prob=keep_prob
        )
    cell_bw = tf.nn.rnn_cell.DropoutWrapper(
        cell_bw, input_keep_prob=1., output_keep_prob=keep_prob
        )
    
    # Dynamic RNN
    return tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, x, sequence_length=x_lengths, dtype=TF_DTYPE,
        scope=scope
        )


def build_bidirectional_multi_rnn(x, x_lengths, n_hiddens, rnn_type="lstm",
        keep_prob=1., **kwargs):
    """
    Build a bidirectional multi-layer RNN.
    
    Apart from those below, parameters are similar to that of `build_rnn`.

    Parameters
    ----------
    n_hiddens : list
        Hidden units in each of the layers.
    """
    for i_layer, n_hidden in enumerate(n_hiddens):
        with tf.variable_scope("rnn_layer_{}".format(i_layer)):
            outputs, states = build_bidirectional_rnn(
                x, x_lengths, n_hidden, rnn_type, keep_prob, **kwargs
                )
            outputs = tf.concat(outputs, 2)
            states = tf.concat(states, 1)

            x = outputs
    return outputs, states


#-----------------------------------------------------------------------------#
#                             CONVOLUTIONAL BLOCKS                            #
#-----------------------------------------------------------------------------#

def build_conv2d_relu(x, filter_shape, stride=1, padding="VALID"):
    """Single convolutional layer with bias and ReLU activation."""
    W = tf.get_variable(
        "W", filter_shape, dtype=TF_DTYPE,
        initializer=tf.contrib.layers.xavier_initializer()
        )
    b = tf.get_variable(
        "b", [filter_shape[-1]], dtype=TF_DTYPE,
        initializer=tf.random_normal_initializer()
        )
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def build_conv2d_linear(x, filter_shape, stride=1, padding="VALID"):
    """Single convolutional layer with bias and linear activation."""
    W = tf.get_variable(
        "W", filter_shape, dtype=TF_DTYPE,
        initializer=tf.contrib.layers.xavier_initializer()
        )
    b = tf.get_variable(
        "b", [filter_shape[-1]], dtype=TF_DTYPE,
        initializer=tf.random_normal_initializer()
        )
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return x


def build_maxpool2d(x, pool_shape, padding="VALID", name=None):
    """Max pool over `x` using a `pool_shape` of [pool_height, pool_width]."""
    ksize = [1,] + pool_shape + [1,]
    return tf.nn.max_pool(
        x, ksize=ksize, strides=ksize, padding=padding, name=name
        )


def build_cnn(x, input_shape, filter_shapes, pool_shapes, strides=None,
        padding="VALID", return_shapes=False):
    """
    Build a convolutional neural network (CNN).
    
    As an example, a CNN with single-channel [28, 28] shaped input with two
    convolutional layers can be constructud using:
    
        x = tf.placeholder(TF_DTYPE, [None, 28*28])
        input_shape = [-1, 28, 28, 1] # [n_data, height, width, d_in]
        filter_shapes = [
            [5, 5, 1, 32],  # filter shape of first layer
            [5, 5, 32, 64]  # filter shape of second layer
            ]   
        pool_shapes = [
            [2, 2],         # pool shape of first layer
            [2, 2]          # pool shape of second layer
            ]
        cnn = build_cnn(x, input_shape, filter_shapes, pool_shapes)
    
    Parameters
    ----------
    x : Tensor [n_data, n_input]
        Input to the CNN, which is reshaped to match `input_shape`.
    input_shape : list
        The shape of the input to the CNN as [n_data, height, width, d_in].
    filter_shapes : list of list
        The filter shape of each layer as [height, width, d_in, d_out].
    pool_shape : list of list
        The pool shape of each layer as [height, width]. If None, then no
        pooling is applied.
    strides : list of int
        This list gives the stride for each layer. If None, then a stride of 1
        is used.
    return_shapes : bool
        If True, a list of list of shapes in the order of the layers are
        additionally returned.
    """
    assert len(filter_shapes) == len(pool_shapes)
    x = tf.reshape(x, input_shape)
    cnn = x
    layer_shapes = []
    for i_layer, (filter_shape, pool_shape) in enumerate(
            zip(filter_shapes, pool_shapes)):
        with tf.variable_scope("cnn_layer_{}".format(i_layer)):
            cnn = build_conv2d_relu(
                cnn, filter_shape, padding=padding, stride=1 if strides is None
                else strides[i_layer]
                )
            if pool_shape is not None:
                cnn = build_maxpool2d(cnn, pool_shape, padding=padding)
            shape = cnn.get_shape().as_list()
            layer_shapes.append(shape)
            print("CNN layer {} shape: {}".format(i_layer, shape))
    if return_shapes:
        return cnn, layer_shapes
    else:
        return cnn


#-----------------------------------------------------------------------------#
#                        TRANSPOSED CONVOLUTION BLOCKS                        #
#-----------------------------------------------------------------------------#

def get_conv2d_transpose_output_shape(input_shape, filter_shape, stride=1):
    """
    Calculate the output shape of a transposed convolution operation.
    
    See https://stackoverflow.com/questions/46885191.
    
    Parameters
    ----------
    input_shape : list
        The shape of the input to the CNN as [n_data, height, width, d_in].
    filter_shape : list
        The filter shape of as [height, width, d_out, d_in].
    """
    input_height = input_shape[1]
    input_width = input_shape[2]
    filter_height = filter_shape[0]
    filter_width = filter_shape[1]
    output_height = (input_height - 1)*stride + filter_height
    output_width = (input_width - 1)*stride + filter_width
    return [input_shape[0], output_height, output_width, filter_shape[2]]    


def build_conv2d_transpose(x, filter_shape, stride=1, activation=tf.nn.relu):
    """
    Single transposed convolutional layer.

    Parameters
    ----------
    filter_shape : list
        The filter shape of as [height, width, d_out, d_in].
    """
    W = tf.get_variable(
        "W", filter_shape, dtype=TF_DTYPE,
        initializer=tf.contrib.layers.xavier_initializer()
        )
    b = tf.get_variable(
        "b", [filter_shape[-2]], dtype=TF_DTYPE,
        initializer=tf.random_normal_initializer()
        )
    input_shape = x.get_shape().as_list()
    output_shape = get_conv2d_transpose_output_shape(
        x.get_shape().as_list(), W.get_shape().as_list(), stride
        )
    output_shape[0] = tf.shape(x)[0]
    x = tf.nn.conv2d_transpose(
        x, W, output_shape, strides=[1, stride, stride, 1], padding="VALID"
        )
    x = tf.nn.bias_add(x, b)
    return activation(x)


def build_unmaxpool2d(x, pool_shape):
    """
    Unpool by repeating units.
    
    See:
    - https://github.com/keras-team/keras/issues/378
    - https://swarbrickjones.wordpress.com/2015/04/29
    """
    from tensorflow.keras.backend import repeat_elements
    s1 = pool_shape[0]
    s2 = pool_shape[1]
    return repeat_elements(repeat_elements(x, s1, axis=1), s2, axis=2)


#-----------------------------------------------------------------------------#
#                            ENCODER-DECODER BLOCKS                           #
#-----------------------------------------------------------------------------#

def build_encdec_lazydynamic(x, x_lengths, n_hidden, rnn_type="lstm",
        keep_prob=1., **kwargs):
    """
    Encoder-decoder with the encoder state fed in at each decoding step.

    The function name refers to the simple implementation essentially using
    `tf.nn.dynamic_rnn` for both the encoder and decoder. Since the encoding
    state is used as input at each decoding time step, the output of the
    decoder is never used. As in `build_encdec_outback`, a linear
    transformation is applied to the output of the decoder such that the final
    output dimensionality matches that of the input `x`.

    Parameters
    ----------
    x : Tensor [n_data, maxlength, d_in]
    """

    maxlength = tf.reduce_max(x_lengths)
    n_output = x.get_shape().as_list()[-1]

    # Encoder
    encoder_output, encoder_states = build_rnn(
        x, x_lengths, n_hidden, rnn_type, keep_prob, scope="rnn_encoder",
        **kwargs
        )
    if rnn_type == "lstm":
        encoder_states = encoder_states.h

    # Decoder

    # Repeat encoder states
    decoder_input = tf.reshape(
        tf.tile(encoder_states, [1, maxlength]), [-1, maxlength, n_hidden]
        )

    # Decoding RNN
    decoder_output, decoder_states = build_rnn(
        decoder_input, x_lengths, n_hidden, rnn_type, scope="rnn_decoder",
        **kwargs
        )
    mask = tf.sign(tf.reduce_max(tf.abs(decoder_output), 2))

    # Final linear layer
    with tf.variable_scope("rnn_decoder/linear_output"):
        decoder_output = tf.reshape(decoder_output, [-1, n_hidden])
        decoder_output = build_linear(decoder_output, n_output)
        decoder_output = tf.reshape(decoder_output, [-1, maxlength, n_output])
        decoder_output *= tf.expand_dims(mask, -1)

    return {
        "encoder_states": encoder_states, "decoder_output":
        decoder_output, "mask": mask
        }


def build_encdec_lazydynamic_latentfunc(x, x_lengths, n_hidden,
        build_latent_func, latent_func_kwargs, rnn_type="lstm", keep_prob=1.,
        **kwargs):
    """
    Encoder-decoder with repeated conditioning and a generic latent layer.

    The function name refers to the simple implementation essentially using
    `tf.nn.dynamic_rnn` for both the encoder and decoder. Since the encoding
    state is used as input at each decoding time step, the output of the
    decoder is never used. As in `build_encdec_outback`, a linear
    transformation is applied to the output of the decoder such that the final
    output dimensionality matches that of the input `x`. A generic latent layer
    is built according to the `build_latent_func` and `latent_func_kwargs`
    parameters.

    Parameters
    ----------
    x : Tensor [n_data, maxlength, d_in]
    build_latent_func : function
        The function to build the latent layer. The function's first parameter
        should be the input Tensor, and it should return a dictionary with an
        element "y" giving the output.
    latent_func_kargs : dict
        Arguments to pass on to `build_latent_func`.
    """

    maxlength = tf.reduce_max(x_lengths)
    n_output = x.get_shape().as_list()[-1]

    # Encoder
    encoder_output, encoder_states = build_rnn(
        x, x_lengths, n_hidden, rnn_type, keep_prob, scope="rnn_encoder",
        **kwargs
        )

    # Latent layer
    if rnn_type == "lstm":
        c, h = encoder_states
    elif rnn_type == "gru" or rnn_type == "rnn":
        c = encoder_states
    latent_layer = build_latent_func(c, **latent_func_kwargs)
    x = latent_layer["y"]

    # Decoder

    # Repeat encoder states
    decoder_input = tf.reshape(
        tf.tile(x, [1, maxlength]), [-1, maxlength, n_hidden]
        )

    # Decoding RNN
    decoder_output, decoder_states = build_rnn(
        decoder_input, x_lengths, n_hidden, rnn_type, keep_prob,
        scope="rnn_decoder", **kwargs
        )
    mask = tf.sign(tf.reduce_max(tf.abs(decoder_output), 2))

    # Final linear layer
    with tf.variable_scope("rnn_decoder/linear_output"):
        decoder_output = tf.reshape(decoder_output, [-1, n_hidden])
        decoder_output = build_linear(decoder_output, n_output)
        decoder_output = tf.reshape(decoder_output, [-1, maxlength, n_output])
        decoder_output *= tf.expand_dims(mask, -1)

    return {
        "encoder_states": encoder_states, "latent_layer": latent_layer, 
        "decoder_output": decoder_output, "mask": mask
        }


def build_multi_encdec_lazydynamic_latentfunc(x, x_lengths, enc_n_hiddens,
        dec_n_hiddens, build_latent_func, latent_func_kwargs, rnn_type="lstm",
        keep_prob=1., y_lengths=None, bidirectional=False,
        add_conditioning_tensor=None, **kwargs):
    """
    Multi-layer encoder-decoder with conditioning and a generic latent layer.

    The function name refers to the simple implementation essentially using
    `tf.nn.dynamic_rnn` for both the encoder and decoder. Since the encoding
    state is used as input at each decoding time step, the output of the
    decoder is never used. As in `build_encdec_outback`, a linear
    transformation is applied to the output of the decoder such that the final
    output dimensionality matches that of the input `x`. A generic latent layer
    is built according to the `build_latent_func` and `latent_func_kwargs`
    parameters.

    Parameters
    ----------
    x : Tensor [n_data, maxlength, d_in]
    build_latent_func : function
        The function to build the latent layer. The function's first parameter
        should be the input Tensor, and it should return a dictionary with an
        element "y" giving the output.
    latent_func_kargs : dict
        Arguments to pass on to `build_latent_func`.
    """

    maxlength = (
        tf.reduce_max(x_lengths) if y_lengths is None else
        tf.reduce_max([tf.reduce_max(x_lengths), tf.reduce_max(y_lengths)])
        )
    n_output = x.get_shape().as_list()[-1]

    # Encoder
    if bidirectional:
        encoder_output, encoder_states = build_bidirectional_multi_rnn(
            x, x_lengths, enc_n_hiddens, rnn_type, keep_prob,
            scope="rnn_encoder", **kwargs
            )
    else:
        encoder_output, encoder_states = build_multi_rnn(
            x, x_lengths, enc_n_hiddens, rnn_type, keep_prob,
            scope="rnn_encoder", **kwargs
            )

    # Latent layer
    if rnn_type == "lstm":
        c, h = encoder_states
    elif rnn_type == "gru" or rnn_type == "rnn":
        c = encoder_states
    latent_layer = build_latent_func(c, **latent_func_kwargs)
    x = latent_layer["y"]

    # Add additional conditioning if provided
    if add_conditioning_tensor is not None:
        x = tf.concat([latent_layer["y"], add_conditioning_tensor], axis=1)
    else:
        x = latent_layer["y"]
    d_latent_layer_output = x.get_shape().as_list()[-1]

    # Decoder

    # Repeat encoder states
    decoder_input = tf.reshape(
        tf.tile(x, [1, maxlength]), [-1, maxlength, d_latent_layer_output]
        )

    # Decoding RNN
    if bidirectional:
        decoder_output, decoder_states = build_bidirectional_multi_rnn(
            decoder_input, x_lengths if y_lengths is None else y_lengths,
            dec_n_hiddens, rnn_type, keep_prob, scope="rnn_decoder", **kwargs
            )
    else:
        decoder_output, decoder_states = build_multi_rnn(
            decoder_input, x_lengths if y_lengths is None else y_lengths,
            dec_n_hiddens, rnn_type, keep_prob, scope="rnn_decoder", **kwargs
            )
    mask = tf.sign(tf.reduce_max(tf.abs(decoder_output), 2))

    # Final linear layer
    with tf.variable_scope("rnn_decoder/linear_output"):
        decoder_output = tf.reshape(
            decoder_output, [-1, int(dec_n_hiddens[-1] * 2) if bidirectional
            else dec_n_hiddens[-1]]
            )
        decoder_output = build_linear(decoder_output, n_output)
        decoder_output = tf.reshape(decoder_output, [-1, maxlength, n_output])
        decoder_output *= tf.expand_dims(mask, -1)

    return {
        "encoder_states": encoder_states, "latent_layer": latent_layer, 
        "decoder_output": decoder_output, "mask": mask
        }


#-----------------------------------------------------------------------------#
#                                 AUTOENCODER                                 #
#-----------------------------------------------------------------------------#

def build_autoencoder(x, enc_n_hiddens, n_z, dec_n_hiddens,
        activation=tf.nn.relu):
    """
    Build an autoencoder with the number of encoder and decoder units.
    
    The number of encoder and decoder units are given as lists. The middle
    (encoding/latent) layer has dimensionality `n_z`. This layer and the final
    layer are linear.
    """

    # Encoder
    for i_layer, n_hidden in enumerate(enc_n_hiddens):
        with tf.variable_scope("ae_enc_{}".format(i_layer)):
            x = build_linear(x, n_hidden)
            x = activation(x)

    # Latent variable
    with tf.variable_scope("ae_latent"):
        x = build_linear(x, n_z)
        z = x

    # Decoder
    for i_layer, n_hidden in enumerate(dec_n_hiddens):
        with tf.variable_scope("ae_dec_{}".format(i_layer)):
            x = build_linear(x, n_hidden)
            if i_layer != len(dec_n_hiddens) - 1:
                x = activation(x)
    y = x
    
    return {"z": z, "y": y}


#-----------------------------------------------------------------------------#
#                           VARIATIONAL AUTOENCODER                           #
#-----------------------------------------------------------------------------#

def build_vae(x, enc_n_hiddens, n_z, dec_n_hiddens, activation=tf.nn.relu):
    """
    Build a VAE with the number of encoder and decoder units.
    
    Parameters
    ----------
    The number of encoder and decoder units are given as lists. The middle
    (encoding/latent) layer has dimensionality `n_z`. The final layer is
    linear.

    Return
    ------
    A dictionary with the mean `z_mean`, and log variance squared
    `z_log_sigma_sq` of the latent variable; the latent variable `z` itself
    (the output of the encoder); and the final output `y` of the network (the
    output of the decoder).
    """
    
    # Encoder
    for i_layer, n_hidden in enumerate(enc_n_hiddens):
        with tf.variable_scope("vae_enc_{}".format(i_layer)):
            x = build_linear(x, n_hidden)
            x = activation(x)
    
    # Latent variable
    with tf.variable_scope("vae_latent_mean"):
        z_mean = build_linear(x, n_z)
    with tf.variable_scope("vae_latent_log_sigma_sq"):
        z_log_sigma_sq = build_linear(x, n_z)
    with tf.variable_scope("vae_latent"):
        eps = tf.random_normal((tf.shape(x)[0], n_z), 0, 1, dtype=TF_DTYPE)
        
        # Reparametrisation trick
        z = z_mean + tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)

    # Decoder
    x = z
    for i_layer, n_hidden in enumerate(dec_n_hiddens):
        with tf.variable_scope("vae_dec_{}".format(i_layer)):
            x = build_linear(x, n_hidden)
            if i_layer != len(dec_n_hiddens) - 1:
                x = activation(x)
    y = x
    
    return {"z_mean": z_mean, "z_log_sigma_sq": z_log_sigma_sq, "z": z, "y": y}


def vae_loss_gaussian(x, y, sigma_sq, z_mean, z_log_sigma_sq,
        reconstruction_loss=None):
    """
    Use p(x|z) = Normal(x; f(z), sigma^2 I), with y = f(z) the decoder output.
    
    A custom `reconstruction_loss` can also be provided.
    """
    
    # Gaussian reconstruction loss
    if reconstruction_loss is None:
        reconstruction_loss = 1./(2*sigma_sq)*(
            tf.losses.mean_squared_error(x, y)
            )
    
    # Regularisation loss
    regularisation_loss = -0.5*tf.reduce_sum(
        1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1
        )
    
    return reconstruction_loss + tf.reduce_mean(regularisation_loss)


def vae_loss_bernoulli(x, y, z_mean, z_log_sigma_sq, reconstruction_loss=None):
    """
    Use a Bernoulli distribution for p(x|z), with the y = f(z) the mean.

    A custom `reconstruction_loss` can also be provided.
    """
    
    # Bernoulli reconstruction loss
    if reconstruction_loss is None:
        reconstruction_loss = -tf.reduce_sum(
            x*tf.log(1e-10 + y) + (1 - x)*tf.log(1e-10 + 1 - y), 1
            )
    
    # Regularisation loss
    regularisation_loss = -0.5*tf.reduce_sum(
        1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1
        )
    
    return tf.reduce_mean(reconstruction_loss + regularisation_loss)


#-----------------------------------------------------------------------------#
#                             VECTOR-QUANTISED VAE                            #
#-----------------------------------------------------------------------------#

def build_vq(x, K, D):
    """
    A vector quantisation layer with `K` components of dimensionality `D`.
    
    See https://github.com/hiwonjoon/tf-vqvae/blob/master/model.py.
    """
    
    # Embeddings
    embeds = tf.get_variable(
        "embeds", [K, D], dtype=TF_DTYPE,
        initializer=tf.contrib.layers.xavier_initializer()
        )
    
    # Quantise inputs
    embeds_tiled = tf.reshape(embeds, [1, K, D])  # [batch_size, K, D]
    x_tiled = tf.tile(tf.expand_dims(x, -2), [1, K, 1])
    dist = tf.norm(x_tiled - embeds_tiled, axis=-1)
    k = tf.argmin(dist, axis=-1)
    z_q = tf.gather(embeds, k)
    
    return {"embeds": embeds, "z_q": z_q}


def build_vqvae(x, enc_n_hiddens, n_z, dec_n_hiddens, K,
        activation=tf.nn.relu):
    """
    Build a VQ-VAE with `K` components.
    
    Parameters
    ----------
    The number of encoder and decoder units are given as lists. The embeddings
    have dimensionality `n_z`. The final layer is linear.

    Return
    ------
    A dictionary with the embeddings, the embedded output `z_e` from the
    encoder, the quantised output `z_q` from the encoder, and the final output
    `y` from the decoder.
    """
    
    # Encoder
    with tf.variable_scope("vqvae_enc"):
        i_layer = 0
        for i_layer, n_hidden in enumerate(enc_n_hiddens):
            with tf.variable_scope("enc_{}".format(i_layer)):
                x = build_linear(x, n_hidden)
                x = activation(x)
        with tf.variable_scope("enc_{}".format(i_layer + 1)):
            z_e = build_linear(x, n_z)
    
    # Quantisation
    with tf.variable_scope("vqvae_quantise"):
        vq = build_vq(z_e, K, n_z)
        embeds = vq["embeds"]
        z_q = vq["z_q"]

    # Decoder
    x = z_q
    with tf.variable_scope("vqvae_dec"):
        for i_layer, n_hidden in enumerate(dec_n_hiddens):
            with tf.variable_scope("dec_{}".format(i_layer)):
                x = build_linear(x, n_hidden)
                if i_layer != len(dec_n_hiddens) - 1:
                    x = activation(x)
    y = x

    return {"embeds": embeds, "z_e": z_e, "z_q": z_q, "y": y}


def vqvae_loss(x, z_e, z_q, embeds, y, learning_rate=0.001, beta=0.25,
        sigma_sq=0.5):
    """
    Return the different loss components and the training operation.
    
    If `sigma_sq` is "bernoulli", then p(x|z) is assumed to be a Bernoulli
    distribution.
    """

    # Losses
    if sigma_sq == "bernoulli":
        recon_loss = tf.reduce_mean(
            -tf.reduce_sum(x*tf.log(1e-10 + y) + (1 - x)*tf.log(1e-10 + 1 - y),
            1)
            )
    else:
        recon_loss = 1./(2*sigma_sq)*tf.losses.mean_squared_error(x, y)
    vq_loss = tf.reduce_mean(
        tf.norm(tf.stop_gradient(z_e) - z_q, axis=-1)**2
        )
    commit_loss = tf.reduce_mean(
        tf.norm(z_e - tf.stop_gradient(z_q), axis=-1)**2
        )
    loss = recon_loss + vq_loss + beta*commit_loss
    
    # Backpropagation: Copy gradients for quantisation
    with tf.variable_scope("backward"):
        
        # Decoder gradients
        decoder_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "vqvae_dec"
            )
        decoder_vars.extend(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rnn_decoder")
            )
        # print(decoder_vars)
        decoder_grads = list(
            zip(tf.gradients(loss, decoder_vars), decoder_vars)
            )
        
        # Encoder gradients
        encoder_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "vqvae_enc"
            )
        encoder_vars.extend(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rnn_encoder")
            )
        # print(encoder_vars)
        z_q_grad = tf.gradients(recon_loss, z_q)
        encoder_grads = [
            (tf.gradients(z_e, var, z_q_grad)[0] +
            beta*tf.gradients(commit_loss, var)[0], var) for var in
            encoder_vars
            ]
        
        # Quantisation gradients
        embeds_grads = list(zip(tf.gradients(vq_loss, embeds), [embeds]))
        
        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(
            decoder_grads + encoder_grads + embeds_grads
            )

    return loss, recon_loss, vq_loss, commit_loss, train_op


#-----------------------------------------------------------------------------#
#                               CATEGORICAL VAE                               #
#-----------------------------------------------------------------------------#

# Code adapted from https://github.com/ericjang/gumbel-softmax/

def sample_gumbel(shape, eps=1e-20): 
    """Sample from Gumbel(0, 1) distribution."""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature): 
    """Draw a sample from the Gumbel-Softmax distribution."""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y/temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretise.
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(
            tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype
            )
        y = tf.stop_gradient(y_hard - y) + y
    return y


def build_catvae(x, enc_n_hiddens, dec_n_hiddens, K, N, activation=tf.nn.relu):
    """
    Build a categorical VAE with `N` distributions each with `K` components.

    Parameters
    ----------
    The number of encoder and decoder units are given as lists.

    Return
    ------
    A dictionary with the log of the categorical distribution based directly on
    the logits `log_logits_categorical`, the one-hot latent variable output `z`
    from the encoder, the final output `y` from the decoder, and the temperate
    variable `tau`.
    """

    tau = tf.placeholder(TF_DTYPE, [])
    
    # Encoder
    for i_layer, n_hidden in enumerate(enc_n_hiddens):
        with tf.variable_scope("catvae_enc_{}".format(i_layer)):
            x = build_linear(x, n_hidden)
            x = activation(x)

    # Latent variable
    with tf.variable_scope("catvae_latent"):
        logits = build_linear(x, K*N)  # the log(pi_i)'s  # log_pis
        softmax_logits = tf.nn.softmax(logits)
        log_logits_categorical = tf.log(softmax_logits + 1e-20)
        z = tf.reshape(gumbel_softmax(logits, tau, hard=False), [-1, N, K])

    # Decoder
    x = tf.reshape(z, [-1, N*K])
    for i_layer, n_hidden in enumerate(dec_n_hiddens):
        with tf.variable_scope("catvae_dec_{}".format(i_layer)):
            x = build_linear(x, n_hidden)
            if i_layer != len(dec_n_hiddens) - 1:
                x = activation(x)
    y = x
    
    return {
        "softmax_logits": softmax_logits, "log_logits_categorical":
        log_logits_categorical, "z": z, "y": y, "tau": tau
        }
