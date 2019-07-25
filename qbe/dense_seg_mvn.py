#!/usr/bin/env python

"""
Perform mean and variance normalisation.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017, 2019
"""

from datetime import datetime
from os import path
from tqdm import tqdm
import argparse
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
    parser.add_argument("exp_dir", type=str, help="experiments directory")
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

    # Read queries
    fn = path.join(args.exp_dir, "queries.npz")
    if not path.isfile(fn):
        import re
        fn = path.join(
            re.sub("min\_.*step\_\d*\.", "", args.exp_dir), "queries.npz"
            )
    print("Reading:", fn)
    queries_dict = np.load(fn)
    print("No. queries:", len(queries_dict.keys()))

    # Read search collection
    fn = path.join(args.exp_dir, "search.npz")
    print("Reading:", fn)
    search_dict = np.load(fn)
    print("No. search utterances:", len(search_dict.keys()))

    # Calculate mean and variance
    search_stacked = np.vstack([search_dict[i] for i in search_dict])
    mean = np.mean(search_stacked, axis=0)
    std = np.std(search_stacked, axis=0)
    std[std == 0] = np.mean(std)  # hack

    # Apply normalisation
    mvn_queries_dict = {}
    print("Normalising queries:")
    for query_key in tqdm(queries_dict):
        mvn_queries_dict[query_key] = (
            np.array(queries_dict[query_key]) - mean
            ) / std
    print("No. queries:", len(mvn_queries_dict))
    mvn_search_dict = {}
    print("Normalising search utterances:")
    for search_key in tqdm(search_dict):
        mvn_search_dict[search_key] = (
            np.array(search_dict[search_key]) - mean
            ) / std
    print("No. search utterances:", len(mvn_search_dict))

    print(datetime.now())

    # Create output directory
    exp_dir = path.normpath(args.exp_dir)
    output_dir = path.join(
        path.split(exp_dir)[0], "mvn." + path.split(exp_dir)[1]
        )
    if not path.isdir(output_dir):
        os.makedirs(output_dir)

    # Write normalized Numpy archives
    fn = path.join(output_dir, "queries.npz")
    print("Writing:", fn)
    np.savez(fn, **mvn_queries_dict)
    fn = path.join(output_dir, "search.npz")
    print("Writing:", fn)
    np.savez(fn, **mvn_search_dict)

    print(datetime.now())


if __name__ == "__main__":
    main()
