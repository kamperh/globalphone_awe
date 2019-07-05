#!/usr/bin/env python

"""
Extract MFCC features for a GlobalPhone language for further analysis.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from collections import Counter
from os import path
import argparse
import codecs
import numpy as np
import os
import random
import sys

sys.path.append("..")
sys.path.append(path.join("..", "features"))

from paths import gp_alignments_dir
import utils


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "language", type=str, help="GlobalPhone language",
        choices=["BG", "CH", "CR", "CZ", "FR", "GE", "HA", "KO", "PL", "PO",
        "RU", "SP", "SW", "TH", "TU", "VN"]
        )
    parser.add_argument(
        "--analyse", action="store_true",
        help="intermediate list analysis", default=False
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def read_fa(fa_fn):
    """
    Return a dict of list of (start_time, end_time, label) with utterance keys.
    """
    fa_dict = {}
    with codecs.open(fa_fn) as f:
        for line in f:
            utt_key, _, start, duration, label = line.strip().split()
            start = float(start)
            duration = float(duration)
            end = start + duration
            if not utt_key in fa_dict:
                fa_dict[utt_key] = []
            fa_dict[utt_key].append((start, end, label))
    return fa_dict


def pronunciations_from_fa(word_fa_fn, phone_fa_fn):
    """
    Return a dict of word tokens with pronunciations using forced alignments.

    The dictionary keys are the word token keys and the values are lists of
    phone labels.
    """

    # Read forced alignments
    # phone_fa[utt_key] is list of (start_time, end_time, phone)
    print("Reading:", phone_fa_fn)
    phone_fa = read_fa(phone_fa_fn) 
    print("Reading:", word_fa_fn)
    word_fa = read_fa(word_fa_fn)

    # For each word
    pronunciations_dict = {}
    for utt_key in sorted(word_fa):
        for word_start, word_end, word in word_fa[utt_key]:

            if word == "<unk>":
                continue

            # Find phone sequence
            phone_sequence = []
            for (phone_start, phone_end, phone) in phone_fa[utt_key]:
                if (phone_start >= word_start and phone_start < word_end and
                        phone != "sil"):
                    # Phone is in word
                    phone = phone.split("_")[0]
                    phone_sequence.append(phone)
            assert len(phone_sequence) != 0, "pronunciation not found"
            word_start_frame = int(round(word_start*100))
            word_end_frame = int(round(word_end*100))
            segment_key = "{}_{}_{:06d}-{:06d}".format(
                word, utt_key, word_start_frame, word_end_frame + 1
                )
            pronunciations_dict[segment_key] = phone_sequence

    return pronunciations_dict


def filter_segment_keys(segment_keys, n_min_tokens_per_type=0,
        n_max_tokens_per_type=np.inf):

    random.seed(1)
    random.shuffle(segment_keys)
    labels = [i.split("_")[0] for i in segment_keys]

    # Find valid types
    valid_types = []
    counts = Counter(labels)
    for key in counts:
        if counts[key] >= n_min_tokens_per_type:
            valid_types.append(key)

    # Filter
    filtered_keys = []
    tokens_per_type = Counter()
    for i in range(len(labels)):
        label = labels[i]
        if (label in valid_types and tokens_per_type[label] <=
                n_max_tokens_per_type):
            filtered_keys.append(segment_keys[i])
            tokens_per_type[label] += 1

    return filtered_keys


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    feat_type = "mfcc"

    list_dir = path.join("lists", args.language)
    if not path.isdir(list_dir):
        os.makedirs(list_dir)
    feat_dir = path.join(feat_type, args.language)
    if not path.isdir(feat_dir):
        os.makedirs(feat_dir)

    # All ground truth word segments with pronunciations
    for subset in ["dev"]:  #, "eval", "train"]:

        list_fn = path.join(list_dir, subset + ".all_gt_words.list")
        pronunciations_fn = path.join(list_dir, subset + ".prons")

        # Read forced alignments and obtain pronunciations
        word_fa_fn = path.join(
            gp_alignments_dir, args.language, subset + ".ctm"
            )
        phone_fa_fn = path.join(
            gp_alignments_dir, args.language, subset + ".phone.ipa.ctm"
            )
        pronunciations_dict = pronunciations_from_fa(
            word_fa_fn, phone_fa_fn
            )

        # Write pronunciation list
        if not path.isfile(pronunciations_fn):
            print("Writing:", pronunciations_fn)
            with codecs.open(pronunciations_fn, "w", "utf-8") as f:
                for segment_key in sorted(pronunciations_dict):
                    f.write(
                        segment_key + " " +
                        ",".join(pronunciations_dict[segment_key]) + "\n"
                        )
        else:
            print("Using existing file:", pronunciations_fn)

        # Write segments
        if not path.isfile(list_fn):
            print("Writing:", list_fn)
            with codecs.open(list_fn, "w", "utf-8") as f:
                for segment_key in sorted(pronunciations_dict):
                    f.write(segment_key + "\n")
        else:
            print("Using existing file:", list_fn)

        if args.analyse:
            import matplotlib.pyplot as plt
            import numpy as np

            # Most common words
            labels = [i.split("_")[0] for i in pronunciations_dict]
            counter = Counter(labels)
            print("No. word types:", len(counter))
            print("No. word tokens:", len(labels))
            print("Most common words:", counter.most_common(10))

            # Histogram of word count
            counts = counter.values()
            plt.figure()
            plt.hist(counts, 50)
            plt.yscale("log")
            plt.ylabel("No. of types with this many tokens")
            plt.xlabel("No. of tokens")

            # # Temp
            # # Most common words
            # labels = [i.split("_")[0] for i in filtered_keys]
            # counter = Counter(labels)
            # print("No. word types:", len(counter))
            # print("No. word tokens:", len(labels))
            # print("Most common words:", counter.most_common(10))

            # # Histogram of word count
            # counts = counter.values()
            # plt.figure()
            # plt.hist(counts, 50)
            # plt.yscale("log")
            # plt.ylabel("No. of types with this many tokens")
            # plt.xlabel("No. of tokens")

            plt.show()

        # Filter 1
        print("Applying filter 1")
        n_min_tokens_per_type = 10
        n_max_tokens_per_type = 25
        filtered_keys = filter_segment_keys(
            list(pronunciations_dict), n_min_tokens_per_type,
            n_max_tokens_per_type
            )
        print("No. tokens:", len(filtered_keys))
        print(
            "No. types:", len(set([i.split("_")[0] for i in filtered_keys]))
            )
        filtered_list_fn = path.join(list_dir, subset + ".filter1_gt.list")
        print("Writing:", filtered_list_fn)
        if not path.isfile(filtered_list_fn):
            with codecs.open(filtered_list_fn, "w", "utf-8") as f:
                for segment_key in sorted(filtered_keys):
                    f.write(segment_key + "\n")
        else:
            print("Using existing file:", filtered_list_fn)

        # Extract word segments from the MFCC NumPy archives
        input_npz_fn = path.join(
            "..", "features", feat_type, args.language, args.language.lower() +
            "." + subset + ".npz"
            )
        output_npz_fn = path.join(
            feat_dir, args.language.lower() + "." + subset + ".filter1_gt.npz"
            )
        if not path.isfile(output_npz_fn):
            utils.segments_from_npz(
                input_npz_fn, filtered_list_fn, output_npz_fn
                )
        else:
            print("Using existing file:", output_npz_fn)

        # dev.filtered_gt_words.list

if __name__ == "__main__":
    main()
