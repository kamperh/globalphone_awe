#!/usr/bin/env python

"""
Create links to the MFCC files.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from os import path
import numpy as np
import os

import argparse
import sys

relative_features_dir = path.join("..", "..", "..", "features")
sixteen_languages = [
    "BG", "CH", "CR", "CZ", "FR", "GE", "HA", "KO", "PL", "PO", "RU", "SP",
    "SW", "TH", "TU", "VN"
    ]

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
        choices=sixteen_languages + ["all"]
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def link_features(npz_fn, link_fn, link_dir):
    assert (
        path.isfile(path.join(link_dir, npz_fn))
        ), "missing file: {}".format(path.join(link_dir, npz_fn))
    if not path.isfile(link_fn):
        print("Linking:", npz_fn, "to", link_fn)
        os.symlink(npz_fn, link_fn)
    else:
        print("Using existing link:", link_fn)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    if args.language == "all":
        languages = sixteen_languages
    else:
        languages = [args.language]

    for language in languages:

        print("Linking features for", language)

        # Create link directory
        link_dir = path.join("data", language)
        if not path.isdir(link_dir):
            os.makedirs(link_dir)

        # Training: All features
        npz_fn = path.join(
            relative_features_dir, "mfcc", language, language.lower() +
            ".train.npz"
            )
        link_fn = path.join(link_dir, "train.all.npz")
        link_features(npz_fn, link_fn, link_dir)

        # Training: Ground truth words
        npz_fn = path.join(
            relative_features_dir, "mfcc", language, language.lower() +
            ".train.gt_words.npz"
            )
        link_fn = path.join(link_dir, "train.gt.npz")
        link_features(npz_fn, link_fn, link_dir)

        # Training: UTD words
        npz_fn = path.join(
            relative_features_dir, "mfcc", language, language.lower() +
            ".train.utd_terms.npz"
            )
        link_fn = path.join(link_dir, "train.utd.npz")
        link_features(npz_fn, link_fn, link_dir)

        # Validation: Ground truth words
        npz_fn = path.join(
            relative_features_dir, "mfcc", language, language.lower() +
            ".dev.gt_words.npz"
            )
        link_fn = path.join(link_dir, "val.npz")
        link_features(npz_fn, link_fn, link_dir)

        # Testing: Ground truth words
        npz_fn = path.join(
            relative_features_dir, "mfcc", language, language.lower() +
            ".eval.gt_words.npz"
            )
        link_fn = path.join(link_dir, "test.npz")
        link_features(npz_fn, link_fn, link_dir)


if __name__ == "__main__":
    main()
