#!/usr/bin/env python

"""
Perform same-different evaluation of fixed-dimensional representations.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2018, 2019
"""

from datetime import datetime
from os import path
from scipy.spatial.distance import pdist
import argparse
import numpy as np
import sys

sys.path.append(path.join("..", "..", "src", "speech_dtw", "utils"))

import samediff


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("npz_fn", type=str, help="NumPy archive")
    parser.add_argument(
        "--metric", choices=["cosine", "euclidean", "hamming", "chebyshev"],
        default="cosine",
        help="distance metric (default: %(default)s)"
        )
    parser.add_argument(
        "--mean_ap", dest="mean_ap", action="store_true",
        help="also compute mean average precision (this is significantly "
        "more resource intensive)"
        )
    parser.add_argument(
        "--mvn", action="store_true",
        help="mean and variance normalise (default: False)"
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

    print(datetime.now())

    print("Reading:", args.npz_fn)
    npz = np.load(args.npz_fn)

    print(datetime.now())

    # if args.normalize:
    #     print("Normalizing embeddings")
    # else:
    print("Ordering embeddings")
    n_embeds = 0
    X = []
    ids = []
    for label in sorted(npz):
        ids.append(label)
        X.append(npz[label])
        n_embeds += 1
    X = np.array(X)
    print("No. embeddings:", n_embeds)
    print("Embedding dimensionality:", X.shape[1])

    if args.mvn:
        normed = (X - X.mean(axis=0)) / X.std(axis=0)
        X = normed

    print(datetime.now())

    print("Calculating distances")
    distances = pdist(X, metric=args.metric)

    print(datetime.now())

    print("Getting labels and speakers")
    labels = []
    speakers = []
    for utt_id in ids:
        utt_id = utt_id.split("_")
        word = utt_id[0]
        speaker = utt_id[1]
        labels.append(word)
        speakers.append(speaker)

    if args.mean_ap:
        print(datetime.now())
        print("Calculating mean average precision")
        mean_ap, mean_prb, ap_dict = samediff.mean_average_precision(
            distances, labels
            )
        print("Mean average precision:", mean_ap)
        print("Mean precision-recall breakeven:", mean_prb)

    print(datetime.now())

    print("Calculating average precision")
    # matches = samediff.generate_matches_array(labels)  # Temp
    word_matches = samediff.generate_matches_array(labels)
    speaker_matches = samediff.generate_matches_array(speakers)
    print("No. same-word pairs:", sum(word_matches))
    print("No. same-speaker pairs:", sum(speaker_matches))
    
    # # Temp
    # ap, prb = samediff.average_precision(
    #     distances[matches == True], distances[matches == False]
    #     )
    # print("Average precision:", ap)
    # print("Precision-recall breakeven:", prb)

    sw_ap, sw_prb, swdp_ap, swdp_prb = samediff.average_precision_swdp(
        distances[np.logical_and(word_matches, speaker_matches)],
        distances[np.logical_and(word_matches, speaker_matches == False)],
        distances[word_matches == False]
        )
    print("-"*79)
    print("Average precision:", sw_ap)
    print("Precision-recall breakeven:", sw_prb)
    print("SWDP average precision:", swdp_ap)
    print("SWDP precision-recall breakeven:", swdp_prb)
    print("-"*79)

    print(datetime.now())


if __name__ == "__main__":
    main()
