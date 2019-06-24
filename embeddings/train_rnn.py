#!/usr/bin/env python

"""
Train a word classifier RNN.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018, 2019
"""

from datetime import datetime
from os import path
from scipy.spatial.distance import pdist
import argparse
import pickle
import hashlib
import numpy as np
import os
import random
import sys
import tensorflow as tf

sys.path.append(path.join("..", "src"))
sys.path.append(path.join("..", "..", "src", "speech_dtw", "utils"))

from link_mfcc import sixteen_languages
from tflego import NP_DTYPE, TF_DTYPE, NP_ITYPE, TF_ITYPE
import batching
import data_io
import samediff
import tflego
import training


#-----------------------------------------------------------------------------#
#                           DEFAULT TRAINING OPTIONS                          #
#-----------------------------------------------------------------------------#

default_options_dict = {
        "train_lang": None,                 # language code
        "val_lang": None,
        "train_tag": "utd",                 # "gt", "utd", "rnd"
        "max_length": 100,
        "bidirectional": False,
        "rnn_type": "gru",                  # "lstm", "gru", "rnn"
        "rnn_n_hiddens": [400, 400, 400],
        "ff_n_hiddens": [130],              # embedding dimensionality
        "learning_rate": 0.001,
        "rnn_keep_prob": 1.0,
        "ff_keep_prob": 1.0,
        "n_epochs": 10,
        "batch_size": 300,
        "n_buckets": 3,
        "extrinsic_usefinal": False,        # if True, during final extrinsic
                                            # evaluation, the final saved model
                                            # will be used (instead of the
                                            # validation best)
        "n_val_interval": 1,
        "rnd_seed": 1,
    }


#-----------------------------------------------------------------------------#
#                              TRAINING FUNCTIONS                             #
#-----------------------------------------------------------------------------#

def build_rnn_from_options_dict(x, x_lengths, options_dict):
    network_dict = {}
    if options_dict["bidirectional"]:
        rnn_outputs, rnn_states = tflego.build_bidirectional_multi_rnn(
            x, x_lengths, options_dict["rnn_n_hiddens"],
            rnn_type=options_dict["rnn_type"],
            keep_prob=options_dict["rnn_keep_prob"]
            )
    else:
        rnn_outputs, rnn_states = tflego.build_multi_rnn(
            x, x_lengths, options_dict["rnn_n_hiddens"],
            rnn_type=options_dict["rnn_type"],
            keep_prob=options_dict["rnn_keep_prob"]
            )
    if options_dict["rnn_type"] == "lstm":
        rnn_states = rnn_states.h
    rnn = tflego.build_feedforward(
        rnn_states, options_dict["ff_n_hiddens"],
        keep_prob=options_dict["ff_keep_prob"]
        )
    network_dict["encoding"] = rnn
    with tf.variable_scope("ff_layer_final"):
        rnn = tflego.build_linear(rnn, options_dict["n_classes"])
    network_dict["output"] = rnn
    return network_dict


def train_rnn(options_dict):
    """Train and save a Siamese triplets model."""

    # PRELIMINARY

    print(datetime.now())

    # Output directory
    hasher = hashlib.md5(repr(sorted(options_dict.items())).encode("ascii"))
    hash_str = hasher.hexdigest()[:10]
    model_dir = path.join(
        "models", options_dict["train_lang"] + "." + options_dict["train_tag"],
        options_dict["script"], hash_str
        )
    options_dict_fn = path.join(model_dir, "options_dict.pkl")
    print("Model directory:", model_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    print("Options:", options_dict)

    # Random seeds
    random.seed(options_dict["rnd_seed"])
    np.random.seed(options_dict["rnd_seed"])
    tf.set_random_seed(options_dict["rnd_seed"])


    # LOAD AND FORMAT DATA

    # Training data
    train_tag = options_dict["train_tag"]
    npz_fn = path.join(
        "data", options_dict["train_lang"], "train." + train_tag + ".npz"
        )
    train_x, train_labels, train_lengths, train_keys, train_speakers = (
        data_io.load_data_from_npz(npz_fn, None)
        )

    # Convert training labels to integers
    train_label_set = list(set(train_labels))
    label_to_id = {}
    for i, label in enumerate(sorted(train_label_set)):
        label_to_id[label] = i
    train_y = []
    for label in train_labels:
        train_y.append(label_to_id[label])
    train_y = np.array(train_y, dtype=NP_ITYPE)
    options_dict["n_classes"] = len(label_to_id)
    print("No. classes:", options_dict["n_classes"])

    # Validation data
    if options_dict["val_lang"] is not None:
        npz_fn = path.join("data", options_dict["val_lang"], "val.npz")
        val_x, val_labels, val_lengths, val_keys, val_speakers = (
            data_io.load_data_from_npz(npz_fn)
            )

    # Truncate and limit dimensionality
    max_length = options_dict["max_length"]
    d_frame = 13  # None
    options_dict["n_input"] = d_frame
    print("Limiting dimensionality:", d_frame)
    print("Limiting length:", max_length)
    data_io.trunc_and_limit_dim(train_x, train_lengths, d_frame, max_length)
    if options_dict["val_lang"] is not None:
        data_io.trunc_and_limit_dim(val_x, val_lengths, d_frame, max_length)


    # DEFINE MODEL

    print(datetime.now())
    print("Building model")

    # Model filenames
    intermediate_model_fn = path.join(model_dir, "rnn.tmp.ckpt")
    model_fn = path.join(model_dir, "rnn.best_val.ckpt")

    # Model graph
    x = tf.placeholder(TF_DTYPE, [None, None, options_dict["n_input"]])
    x_lengths = tf.placeholder(TF_ITYPE, [None])
    y = tf.placeholder(TF_ITYPE, [None])
    network_dict = build_rnn_from_options_dict(x, x_lengths, options_dict)
    encoding = network_dict["encoding"]
    output = network_dict["output"]

    # Cross entropy loss
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output)
        )
    optimizer = tf.train.AdamOptimizer(
        learning_rate=options_dict["learning_rate"]
        ).minimize(loss)


    # TRAIN AND VALIDATE

    print(datetime.now())
    print("Training model")

    # Validation function
    def samediff_val(normalise=True):
        # Embed validation
        np.random.seed(options_dict["rnd_seed"])
        val_batch_iterator = batching.SimpleIterator(val_x, len(val_x), False)
        labels = [val_labels[i] for i in val_batch_iterator.indices]
        speakers = [val_speakers[i] for i in val_batch_iterator.indices]
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, val_model_fn)
            for batch_x_padded, batch_x_lengths in val_batch_iterator:
                np_x = batch_x_padded
                np_x_lengths = batch_x_lengths
                np_z = session.run(
                    [encoding], feed_dict={x: np_x, x_lengths: np_x_lengths}
                    )[0]
                break  # single batch

        embed_dict = {}
        for i, utt_key in enumerate(
                [val_keys[i] for i in val_batch_iterator.indices]):
            embed_dict[utt_key] = np_z[i]

        # Same-different
        if normalise:
            np_z_normalised = (np_z - np_z.mean(axis=0))/np_z.std(axis=0)
            distances = pdist(np_z_normalised, metric="cosine")
        else:
            distances = pdist(np_z, metric="cosine")
        # matches = samediff.generate_matches_array(labels)
        # ap, prb = samediff.average_precision(
        #     distances[matches == True], distances[matches == False]
        #     )
        word_matches = samediff.generate_matches_array(labels)
        speaker_matches = samediff.generate_matches_array(speakers)
        sw_ap, sw_prb, swdp_ap, swdp_prb = samediff.average_precision_swdp(
            distances[np.logical_and(word_matches, speaker_matches)],
            distances[np.logical_and(word_matches, speaker_matches == False)],
            distances[word_matches == False]
            )
        # return [sw_prb, -sw_ap, swdp_prb, -swdp_ap]
        return [swdp_prb, -swdp_ap]

    # Train RNN
    val_model_fn = intermediate_model_fn
    train_batch_iterator = batching.LabelledBucketIterator(
        train_x, train_y, options_dict["batch_size"],
        n_buckets=options_dict["n_buckets"], shuffle_every_epoch=True
        )
    if options_dict["val_lang"] is None:
        record_dict = training.train_fixed_epochs(
            options_dict["n_epochs"], optimizer, loss, train_batch_iterator,
            [x, x_lengths, y],
            save_model_fn=intermediate_model_fn
            )
    else:
        record_dict = training.train_fixed_epochs_external_val(
            options_dict["n_epochs"], optimizer, loss, train_batch_iterator,
            [x, x_lengths, y], samediff_val,
            save_model_fn=intermediate_model_fn,
            save_best_val_model_fn=model_fn,
            n_val_interval=options_dict["n_val_interval"]
            )

    # Save record
    record_dict_fn = path.join(model_dir, "record_dict.pkl")
    print("Writing:", record_dict_fn)
    with open(record_dict_fn, "wb") as f:
        pickle.dump(record_dict, f, -1)

    # Save options_dict
    options_dict_fn = path.join(model_dir, "options_dict.pkl")
    print("Writing:", options_dict_fn)
    with open(options_dict_fn, "wb") as f:
        pickle.dump(options_dict, f, -1)


    # FINAL EXTRINSIC EVALUATION

    if options_dict["val_lang"] is not None:
        print ("Performing final validation")
        if options_dict["extrinsic_usefinal"]:
            val_model_fn = intermediate_model_fn
        else:
            val_model_fn = model_fn
        # sw_prb, sw_ap, swdp_prb, swdp_ap = samediff_val(normalise=False)
        swdp_prb, swdp_ap = samediff_val(normalise=False)
        # sw_ap = -sw_ap
        swdp_ap = -swdp_ap
        # (sw_prb_normalised, sw_ap_normalised, swdp_prb_normalised,
        #     swdp_ap_normalised) = samediff_val(normalise=True)
        swdp_prb_normalised, swdp_ap_normalised = samediff_val(normalise=True)
        # sw_ap_normalised = -sw_ap_normalised
        swdp_ap_normalised = -swdp_ap_normalised
        print("Validation SWDP AP:", swdp_ap)
        print("Validation SWDP AP with normalisation:", swdp_ap_normalised)
        ap_fn = path.join(model_dir, "val_ap.txt")
        print("Writing:", ap_fn)
        with open(ap_fn, "w") as f:
            f.write(str(swdp_ap) + "\n")
            f.write(str(swdp_ap_normalised) + "\n")
        print("Validation model:", val_model_fn)

    print(datetime.now())


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "train_lang", type=str,
        help="GlobalPhone training language {BG, CH, CR, CZ, FR, GE, HA, KO, "
        "PL, PO, RU, SP, SW, TH, TU, VN}",
        )
    parser.add_argument(
        "--val_lang", type=str, help="validation language",
        choices=sixteen_languages, default=None
        )
    parser.add_argument(
        "--n_epochs", type=int,
        help="number of epochs of training (default: %(default)s)",
        default=default_options_dict["n_epochs"]
        )
    parser.add_argument(
        "--batch_size", type=int,
        help="size of mini-batch (default: %(default)s)",
        default=default_options_dict["batch_size"]
        )
    parser.add_argument(
        "--train_tag", type=str, choices=["gt", "utd"],
        help="training set tag (default: %(default)s)",
        default=default_options_dict["train_tag"]
        )
    parser.add_argument(
        "--extrinsic_usefinal", action="store_true",
        help="if set, during final extrinsic evaluation, the final saved "
        "model will be used instead of the validation best (default: "
        "%(default)s)",
        default=default_options_dict["extrinsic_usefinal"]
        )
    parser.add_argument(
        "--rnd_seed", type=int, help="random seed (default: %(default)s)",
        default=default_options_dict["rnd_seed"]
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Set options
    options_dict = default_options_dict.copy()
    options_dict["script"] = "train_rnn"
    options_dict["train_lang"] = args.train_lang
    options_dict["val_lang"] = args.val_lang
    options_dict["n_epochs"] = args.n_epochs
    options_dict["batch_size"] = args.batch_size
    options_dict["train_tag"] = args.train_tag
    options_dict["extrinsic_usefinal"] = args.extrinsic_usefinal
    options_dict["rnd_seed"] = args.rnd_seed

    # Do not output TensorFlow info and warning messages
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.logging.set_verbosity(tf.logging.ERROR)
    if type(tf.contrib) != type(tf):
        tf.contrib._warning = None

    # Train model
    train_rnn(options_dict)    


if __name__ == "__main__":
    main()
