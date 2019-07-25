#!/usr/bin/env python

"""
Extract queries and link search datasets for a particular GlobalPhone language.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from os import path
from tqdm import tqdm
import argparse
import codecs
import numpy as np
import os
import sys

sys.path.append(path.join("..", "embeddings"))

from link_mfcc import link_features


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
        choices=["HA"]
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

    # Create feature/link directory
    feat_dir = path.join("data", args.language)
    if not path.isdir(feat_dir):
        os.makedirs(feat_dir)

    # Read keywords
    keywords_fn = path.join("..", "data", args.language, "keywords.txt")
    with codecs.open(keywords_fn, "r", "utf-8") as f:
        keywords = [line.strip() for line in f]
    print("No. keywords:", len(keywords))

    # Extract queries from development data
    queries_feat_fn = path.join(feat_dir, "queries.npz")
    if not path.isfile(queries_feat_fn):
        dev_feat_fn = path.join(
            "..", "features", "mfcc", args.language, args.language.lower() +
            ".dev.gt_words.npz"
            )
        assert path.isfile(dev_feat_fn), "file not found: " + dev_feat_fn
        print("Reading:", dev_feat_fn)
        dev_feat_dict = np.load(dev_feat_fn)
        print("Extracting queries:")
        queries_feat_dict = {}
        for utterance_key in tqdm(dev_feat_dict):
            label = utterance_key.split("_")[0]
            if label in keywords:
                queries_feat_dict[utterance_key] = dev_feat_dict[utterance_key]
        print("No. queries tokens:", len(queries_feat_dict))
        print("Writing:", queries_feat_fn)
        np.savez(queries_feat_fn, **queries_feat_dict)
    else:
        print("Using existing file:", queries_feat_fn)

    # Link test search utterances
    search_feat_fn = path.join(
        "..", "..", "..", "features", "mfcc", args.language,
        args.language.lower() + ".eval.npz"
        )  # relative path
    link_fn = path.join(feat_dir, "search.npz")
    link_features(search_feat_fn, link_fn, feat_dir)


if __name__ == "__main__":
    main()
