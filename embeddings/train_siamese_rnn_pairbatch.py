#!/usr/bin/env python

"""
Train a Siamese triplets network ensuring that each batch contains pairs.

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
        "margin": 0.25,
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
        "n_max_pairs": None,
        "n_min_tokens_per_type": None,      # if None, no filter is applied
        "n_max_types": None,
        "rnd_seed": 1,
    }


#-----------------------------------------------------------------------------#
#                                BATCH ITERATOR                               #
#-----------------------------------------------------------------------------#

class LabelledPairedBucketIterator(object):
    """
    Iterator over pairs of sequences which are yielded unpaired with labels.
    """
    
    def __init__(self, x_list, y, pair_list, batch_size, n_buckets,
            shuffle_every_epoch=False, speaker_ids=None):

        # Attributes
        self.x_list = x_list
        self.y = y
        self.pair_list = pair_list
        self.batch_size = batch_size
        self.shuffle_every_epoch = shuffle_every_epoch
        self.speaker_ids = speaker_ids

        self.n_input = self.x_list[0].shape[-1]
        self.x_lengths = np.array([i.shape[0] for i in x_list])
        self.n_batches = int(len(self.pair_list)/self.batch_size)
        
        # Set up bucketing
        self.n_buckets = n_buckets
        sorted_indices = np.argsort(
            [max(len(x_list[i]), len(x_list[j])) for i, j in pair_list]
            )
        bucket_size = int(len(self.pair_list)/self.n_buckets)
        self.buckets = []
        for i_bucket in range(n_buckets):
            self.buckets.append(
                sorted_indices[i_bucket*bucket_size:(i_bucket + 1)*bucket_size]
                )
        self.shuffle()

    def shuffle(self):
        for i_bucket in range(self.n_buckets):
            np.random.shuffle(self.buckets[i_bucket])
        self.indices = np.concatenate(self.buckets)
    
    def __iter__(self):

        if self.shuffle_every_epoch:
            self.shuffle()
        
        for i_batch in range(self.n_batches):
            
            batch_pair_list = [
                self.pair_list[i] for i in self.indices[
                i_batch*self.batch_size:(i_batch + 1)*self.batch_size]
                ]

            batch_indices_a = [i for i, j in batch_pair_list]
            batch_indices_b = [j for i, j in batch_pair_list]
            batch_indices = list(
                set(batch_indices_a).union(set(batch_indices_b))
                )

            batch_x_lengths = self.x_lengths[batch_indices]
            batch_y = self.y[batch_indices]
            if self.speaker_ids is not None:
                batch_speakers = self.speaker_ids[batch_indices]

            # Pad to maximum length in batch
            batch_x_padded = np.zeros(
                (len(batch_indices), np.max(batch_x_lengths), self.n_input),
                 dtype=NP_DTYPE
                )
            for i, length in enumerate(batch_x_lengths):
                seq = self.x_list[batch_indices[i]]
                batch_x_padded[i, :length, :] = seq

            if self.speaker_ids is None:
                yield (batch_x_padded, batch_x_lengths, batch_y)
            else:
                yield (
                    batch_x_padded, batch_x_lengths, batch_y, batch_speakers
                    )


#-----------------------------------------------------------------------------#
#                              TRAINING FUNCTIONS                             #
#-----------------------------------------------------------------------------#

def build_siamese_rnn_side(x, x_lengths, rnn_n_hiddens, ff_n_hiddens,
        rnn_type="lstm", rnn_keep_prob=1.0, ff_keep_prob=1.0,
        bidirectional=False):
    """
    Multi-layer RNN serving as one side of a Siamese model.

    Parameters
    ----------
    x : Tensor [n_data, maxlength, d_in]
    """

    if bidirectional:
        rnn_outputs, rnn_states = tflego.build_bidirectional_multi_rnn(
            x, x_lengths, rnn_n_hiddens, rnn_type=rnn_type,
            keep_prob=rnn_keep_prob
            )
    else:
        rnn_outputs, rnn_states = tflego.build_multi_rnn(
            x, x_lengths, rnn_n_hiddens, rnn_type=rnn_type,
            keep_prob=rnn_keep_prob
            )
    if rnn_type == "lstm":
        rnn_states = rnn_states.h
    rnn = tflego.build_feedforward(
        rnn_states, ff_n_hiddens, keep_prob=ff_keep_prob
        )
    return rnn


def build_siamese_from_options_dict(x, x_lengths, options_dict):
    network_dict = {}
    rnn = build_siamese_rnn_side(
        x, x_lengths, options_dict["rnn_n_hiddens"],
        options_dict["ff_n_hiddens"], options_dict["rnn_type"],
        options_dict["rnn_keep_prob"], options_dict["ff_keep_prob"],
        options_dict["bidirectional"]
        )
    rnn = tf.nn.l2_normalize(rnn, axis=1)
    network_dict["output"] = rnn
    return network_dict


def train_siamese(options_dict):
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
    if "+" in options_dict["train_lang"]:
        train_languages = options_dict["train_lang"].split("+")
        train_x = []
        train_labels = []
        train_lengths = []
        train_keys = []
        train_speakers = []
        for cur_lang in train_languages:
            cur_npz_fn = path.join(
                "data", cur_lang, "train." + options_dict["train_tag"] + ".npz"
                )
            (cur_train_x, cur_train_labels, cur_train_lengths, cur_train_keys,
                cur_train_speakers) = data_io.load_data_from_npz(cur_npz_fn,
                None)
            (cur_train_x, cur_train_labels, cur_train_lengths, cur_train_keys,
                cur_train_speakers) = data_io.filter_data(cur_train_x,
                cur_train_labels, cur_train_lengths, cur_train_keys,
                cur_train_speakers,
                n_min_tokens_per_type=options_dict["n_min_tokens_per_type"],
                n_max_types=options_dict["n_max_types"]
                )
            train_x.extend(cur_train_x)
            train_labels.extend(cur_train_labels)
            train_lengths.extend(cur_train_lengths)
            train_keys.extend(cur_train_keys)
            train_speakers.extend(cur_train_speakers)
        print("Total no. items:", len(train_labels))
    else:
        npz_fn = path.join(
            "data", options_dict["train_lang"], "train." +
            options_dict["train_tag"] + ".npz"
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

    # Get pairs
    pair_list = batching.get_pair_list(
        train_labels, both_directions=True,
        n_max_pairs=options_dict["n_max_pairs"]
        )
    print("No. pairs:", int(len(pair_list)/2.0))  # pairs in both directions


    # DEFINE MODEL

    print(datetime.now())
    print("Building model")

    # Model filenames
    intermediate_model_fn = path.join(model_dir, "siamese.tmp.ckpt")
    model_fn = path.join(model_dir, "siamese.best_val.ckpt")

    # Model graph
    x = tf.placeholder(TF_DTYPE, [None, None, options_dict["n_input"]])
    x_lengths = tf.placeholder(TF_ITYPE, [None])
    y = tf.placeholder(TF_ITYPE, [None])
    network_dict = build_siamese_from_options_dict(x, x_lengths, options_dict)
    output = network_dict["output"]

    # Semi-hard triplets loss
    loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
        labels=y, embeddings=output, margin=options_dict["margin"]
        )
    optimizer = tf.train.AdamOptimizer(
        learning_rate=options_dict["learning_rate"]
        ).minimize(loss)

    # Save options_dict
    options_dict_fn = path.join(model_dir, "options_dict.pkl")
    print("Writing:", options_dict_fn)
    with open(options_dict_fn, "wb") as f:
        pickle.dump(options_dict, f, -1)


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
                    [output], feed_dict={x: np_x, x_lengths: np_x_lengths}
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

    # Train Siamese model
    val_model_fn = intermediate_model_fn
    train_batch_iterator = LabelledPairedBucketIterator(
        train_x, train_y, pair_list, options_dict["batch_size"],
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
        "--n_max_pairs", type=int,
        help="maximum number of same-word pairs to use (default: %(default)s)",
        default=default_options_dict["n_max_pairs"]
        )
    parser.add_argument(
        "--n_min_tokens_per_type", type=int,
        help="minimum number of tokens per type (default: %(default)s)",
        default=default_options_dict["n_min_tokens_per_type"]
        )
    parser.add_argument(
        "--n_max_types", type=int,
        help="maximum number of types per language (default: %(default)s)",
        default=default_options_dict["n_max_types"]
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
        "--train_tag", type=str, choices=["gt", "utd", "utd.fixed_labels",
        "utd.fixed_labels_segs", "utd.fixed_segs"],
        help="training set tag (default: %(default)s)",
        default=default_options_dict["train_tag"]
        )
    parser.add_argument(
        "--margin", type=float,
        help="margin for contrastive loss (default: %(default)s)",
        default=default_options_dict["margin"]
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
    options_dict["script"] = "train_siamese_rnn"
    options_dict["train_lang"] = args.train_lang
    options_dict["val_lang"] = args.val_lang
    options_dict["n_max_pairs"] = args.n_max_pairs
    options_dict["n_min_tokens_per_type"] = args.n_min_tokens_per_type
    options_dict["n_max_types"] = args.n_max_types
    options_dict["n_epochs"] = args.n_epochs
    options_dict["batch_size"] = args.batch_size
    options_dict["train_tag"] = args.train_tag
    options_dict["margin"] = args.margin
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
    train_siamese(options_dict)    


if __name__ == "__main__":
    main()
