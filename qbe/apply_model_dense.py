#!/usr/bin/env python

"""
Apply a model to dense segmentationi intervals.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018, 2019
"""

from datetime import datetime
from os import path
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append(path.join("..", "src"))
sys.path.append(path.join("..", "embeddings"))

from apply_model import build_model
from tflego import NP_DTYPE, TF_DTYPE, NP_ITYPE, TF_ITYPE
import batching
import data_io


#-----------------------------------------------------------------------------#
#                            APPLY MODEL FUNCTIONS                            #
#-----------------------------------------------------------------------------#

"""
def build_model(x, x_lengths, options_dict):
    model_dict = {}
    if options_dict["script"] == "train_cae_rnn":
        import train_cae_rnn
        cae = train_cae_rnn.build_cae_from_options_dict(
            x, x_lengths, x_lengths, options_dict
            )
        model_dict["output"] = cae["y"]
        model_dict["encoding"] = cae["z"]
        model_dict["mask"] = cae["mask"]
    elif options_dict["script"] == "train_vae":
        import train_vae
        vae = train_vae.build_vae_from_options_dict(x, x_lengths, options_dict)
        model_dict["output"] = vae["decoder_output"]
        model_dict["encoding"] = vae["latent_layer"]["z_mean"]
        model_dict["mask"] = vae["mask"]
    elif options_dict["script"] == "train_siamese_rnn":
        import train_siamese_rnn
        siamese = train_siamese_rnn.build_siamese_from_options_dict(
            x, x_lengths, options_dict
            )
        model_dict["encoding"] = siamese["output"]
    elif options_dict["script"] == "train_siamese_cnn":
        import train_siamese_cnn
        siamese = train_siamese_cnn.build_siamese_cnn_from_options_dict(
            x, options_dict
            )
        model_dict["encoding"] = siamese["output"]
    elif options_dict["script"] == "train_rnn":
        import train_rnn
        rnn = train_rnn.build_rnn_from_options_dict(
            x, x_lengths, options_dict
            )
        model_dict["encoding"] = rnn["encoding"]
    else:
        assert False, "model type not supported"
    return model_dict
"""


def apply_model(model_fn, language, subset, segtag):

    # Load the model options
    model_dir = path.split(model_fn)[0]
    options_dict_fn = path.join(model_dir, "options_dict.pkl")
    print("Reading:", options_dict_fn)
    with open(options_dict_fn, "rb") as f:
        options_dict = pickle.load(f)

    # Load data and intervals
    npz_fn = path.join("data", language, subset + ".npz")
    x_data, labels, lengths, keys, speakers = data_io.load_data_from_npz(
        npz_fn
        )
    seglist_fn = path.join(
        "data", language, "search.seglist." + segtag + ".pkl"
        )
    print("Reading:", seglist_fn)
    with open(seglist_fn, "rb") as f:
        seglist_dict = pickle.load(f)
    seglists = [seglist_dict[i] for i in keys]
    print("No. utterances:", len(x_data))
    n_intervals = sum([len(i) for i in seglists])
    print("No. intervals:", n_intervals)

    # assert False
    # print("Reading:", input_npz_fn)
    # features_dict = np.load(input_npz_fn)
    # seglist_fn = path.join(
    #     "data", language, "search.seglist." + segtag + ".pkl"
    #     )
    # print("Reading:", seglist_fn)
    # with open(seglist_fn, "rb") as f:
    #     seglist_dict = pickle.load(f)
    # utterances = sorted(features_dict.keys())
    # input_sequences = [features_dict[i] for i in utterances]
    # seglists = [seglist_dict[i] for i in utterances]
    # print("No. utterances:", len(input_sequences))
    # n_intervals = sum([len(i) for i in seglists])
    # print("No. intervals:", n_intervals)

    # if "cnn" in options_dict["script"]:
    #     assert False, "to-do"
    # else:  # rnn

    # print("No. utterances:", len(input_sequences))
    # n_intervals = sum([len(i) for i in seglists])
    # print("No. intervals:", n_intervals)


    # # Load data
    # npz_fn = path.join("data", language, subset + ".npz")
    # x_data, labels, lengths, keys, speakers = data_io.load_data_from_npz(
    #     npz_fn
    #     )


    if "cnn" in options_dict["script"]:

        assert False, "to-do"

        # Pad and flatten data
        x_data, _ = data_io.pad_sequences(
            x_data, options_dict["max_length"], True
            )
        x_data = np.transpose(x_data, (0, 2, 1))
        x_data = x_data.reshape((-1, options_dict["d_in"]))

        # Build model
        x = tf.placeholder(TF_DTYPE, [None, options_dict["d_in"]])
        model = build_model(x, None, options_dict)

        # Embed data
        batch_iterator = batching.LabelledIterator(
            x_data, None, x_data.shape[0], False
            )
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, model_fn)
            for batch_x in batch_iterator:
                np_z = session.run(
                    [model["encoding"]], feed_dict={x: batch_x})[0]
                break  # single batch

    else:  # rnn
        
        # Truncate and limit dimensionality
        data_io.trunc_and_limit_dim(
            x_data, lengths, options_dict["n_input"], None
            )
        print("len(x_data)", len(x_data))
        print("x_data[0].shape", x_data[0].shape)

        class DenseBatchFeedIterator(object):

            def __init__(self, input_sequences, seglists):
                self.input_sequences = input_sequences
                self.n_input = self.input_sequences[0].shape[-1]
                self.seglists = seglists
                print("len(self.input_sequences)", len(self.input_sequences))

            def __iter__(self):
                print("len(self.input_sequences)", len(self.input_sequences))
                for i_utt in range(len(self.input_sequences)):
                    
                    # Get intervals
                    seglist = self.seglists[i_utt]
                    input_sequence = self.input_sequences[i_utt]

                    # Get segments for intervals
                    segments = []
                    for i, j in seglist:
                        segments.append(input_sequence[i:j, :])

                    batch_x_lengths = [i.shape[0] for i in segments]

                    # Pad to maximum length in batch
                    batch_x_padded = np.zeros(
                        (len(batch_x_lengths), np.max(batch_x_lengths),
                        self.n_input), dtype=NP_DTYPE
                        )
                    for i, length in enumerate(batch_x_lengths):
                        seq = segments[i]
                        batch_x_padded[i, :length, :] = seq

                    yield (batch_x_padded, batch_x_lengths)

        batch_iterator = DenseBatchFeedIterator(x_data, seglists)

        # Build model
        x = tf.placeholder(TF_DTYPE, [None, None, options_dict["n_input"]])
        x_lengths = tf.placeholder(TF_ITYPE, [None])
        model = build_model(x, x_lengths, options_dict)

        # Embed data
        batch_iterator = batching.SimpleIterator(x_data, len(x_data), False)
        saver = tf.train.Saver()
        n_outputs = 0
        embed_dict = {}
        with tf.Session() as session:
            saver.restore(session, model_fn)
            print(datetime.now())
            print("Applying model to segments")
            for i_batch, (batch_x_padded, batch_x_lengths) in \
                    enumerate(batch_iterator):
                print("i_batch", i_batch)
                print("batch_x_padded.shape", batch_x_padded.shape)
                cur_output = session.run(
                    [model["encoding"]], feed_dict={x: batch_x_padded,
                    x_lengths: batch_x_lengths}
                    )[0]
                print("cur_output.shape", cur_output.shape)
                utt_key = keys[i_batch]
                seglist = seglists[i_batch]
                embeddings = []
                for i in range(cur_output.shape[0]):
                    embeddings.append(cur_output[i, :])
                    n_outputs += 1
                embed_dict[utt_key] = np.array(embeddings)
            print(datetime.now())

            # for batch_x_padded, batch_x_lengths in batch_iterator:
            #     np_x = batch_x_padded
            #     np_x_lengths = batch_x_lengths
            #     np_z = session.run(
            #         [model["encoding"]], feed_dict={x: np_x, x_lengths:
            #         np_x_lengths}
            #         )[0]
            #     break  # single batch

    print("Processed {} inputs out of {}".format(n_outputs, n_intervals))
    
    return embed_dict


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("model_fn", type=str, help="model checkpoint filename")
    parser.add_argument(
        "language", type=str, help="GlobalPhone language",
        choices=["HA"]
        )
    parser.add_argument(
        "subset", type=str, help="subset to apply model to",
        choices=["search.0", "search.1", "search.test"]
        )
    parser.add_argument(
        "--segtag", type=str,
        help="a tag to identify the dense segments lists "
        "(default: %(default)s)", default="min_20.max_60.step_3"
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

    # Do not output TensorFlow info and warning messages
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.logging.set_verbosity(tf.logging.ERROR)
    if type(tf.contrib) != type(tf):
        tf.contrib._warning = None

    # Embed data
    embed_dict = apply_model(
        args.model_fn, args.language, args.subset, args.segtag
        )

    # Save embeddings
    model_dir, model_fn = path.split(args.model_fn)
    model_key = path.split(path.normpath(model_dir))[1]
    output_dir = path.join("exp", args.language, model_key + "." + args.segtag)
    if not path.isdir(output_dir):
        os.makedirs(output_dir)
    npz_fn = path.join(output_dir, args.subset + ".npz")
    print("Writing:", npz_fn)
    np.savez_compressed(npz_fn, **embed_dict)
    print(datetime.now())


if __name__ == "__main__":
    main()
