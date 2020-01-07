#!/usr/bin/env python

"""
Prepare the data for dense segmental QbE search.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017, 2019
"""

from datetime import datetime
from os import path
import argparse
import pickle
import numpy as np
import os
import sys


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
    parser.add_argument(
        "--min_frames", type=int,
        help="minimum number of frames (default: %(default)s)", default=20
        )
    parser.add_argument(
        "--max_frames", type=int,
        help="maximum number of frames (default: %(default)s)", default=60
        )
    parser.add_argument(
        "--step", type=int,
        help="frame step (default: %(default)s)", default=3
        )
    parser.add_argument(
        "--n_splits", type=int,
        help="number of search collection splits (default: %(default)s)",
        default=2
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

    output_dir = path.join("data", args.language)
    if not path.isdir(output_dir):
        os.makedirs(output_dir)
    segtag = "min_{}.max_{}.step_{}".format(
        args.min_frames, args.max_frames, args.step
        )

    # Subset search collection
    search_dict_fn = path.join("data", args.language, "search.npz")
    print("Reading:", search_dict_fn)
    search_dict = np.load(search_dict_fn)
    search_keys = sorted(search_dict.keys())
    print("No. search utterances:", len(search_keys))

    # Dense search segments list
    seglist_fn = path.join(
        output_dir, "search.seglist." + segtag + ".pkl"
        )
    if not path.isfile(seglist_fn):
        print("Getting segmentation lists")
        seglist_dict = {}
        n_intervals = 0
        for utt_key in search_keys:
            seglist = []
            length = search_dict[utt_key].shape[0]
            i_start = 0
            while i_start < length:
                i_end = i_start + args.min_frames
                while i_end <= length and i_end - i_start <= args.max_frames:
                    seglist.append((i_start, i_end))
                    i_end += args.step
                    n_intervals += 1
                i_start += args.step
            seglist_dict[utt_key] = seglist
        print("No. segmentation intervals:", n_intervals)
        print("Writing:", seglist_fn)
        with open(seglist_fn, "wb") as f:
            pickle.dump(seglist_dict, f, -1)
    else:
        print("Using existing file:", seglist_fn)

    # Split the search collection
    split_dict_fn = path.join(
        "data", args.language, "search." + str(args.n_splits - 1) + ".npz"
        )
    if not path.isfile(split_dict_fn):
        n_items = int(np.ceil(np.float(len(search_keys)) / args.n_splits))
        n_total = 0
        for i_split in range(args.n_splits):
            split_search_keys = search_keys[i_split*n_items:(i_split + 1)*n_items]
            split_dict = {}
            for utt_key in split_search_keys:
                split_dict[utt_key] = search_dict[utt_key]
            split_dict_fn = path.join(
                "data", args.language, "search." + str(i_split) + ".npz"
                )
            print("Writing:", split_dict_fn)
            np.savez(split_dict_fn, **split_dict)
            n_total += len(split_dict)
        print(
            "Wrote {} out of {} utterances".format(len(search_dict.keys()),
            n_total)
            )
    else:
        print("Using existing splits:", split_dict_fn)

    print(datetime.now())


if __name__ == "__main__":
    main()
