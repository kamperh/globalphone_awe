#!/usr/bin/env python

"""
Perform same-different evaluation of fixed-dimensional representations.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2016, 2018, 2019
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
    parser.add_argument("npz_fn", type=str, help="Numpy archive")
    parser.add_argument(
        "--metric", choices=["cosine", "euclidean", "hamming", "chebyshev",
        "kl"], default="cosine", help="distance metric (default: %(default)s)"
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
    metric = args.metric
    if metric == "kl":
        import scipy.stats
        metric = scipy.stats.entropy
    distances = pdist(X, metric=metric)

    print("Getting labels")
    labels = []
    for utt_id in ids:
        word = utt_id.split("_")[0]  # "_".join(utt_id.split("_")[:-2])
        labels.append(word)

    print("Calculating average precision")
    matches = samediff.generate_matches_array(labels)

    ap, prb = samediff.average_precision(
        distances[matches == True], distances[matches == False]
        )
    print("Average precision: {:.4f}".format(ap))
    print("Precision-recall breakeven: {:.4f}".format(prb))

    print(datetime.now())


if __name__ == "__main__":
    main()
