#!/usr/bin/env python

"""
Combine `apply_model_dense_seg.py` output into a single Numpy archive.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017, 2019
"""

from os import path
from tqdm import tdqm
import argparse
import pickle
import glob
import numpy as np
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
    
    features_dict = {}
    for fn in glob.glob(path.join(args.exp_dir, "search.*.npz")):
        print("Reading:", fn)
        split_features_dict = np.load(fn)
        for key in tqdm(split_features_dict):
            features_dict[key] = split_features_dict[key]
            # print(split_features_dict[key].shape)

    fn = path.join(args.exp_dir, "search.npz")
    print("Writing:", fn)
    np.savez(fn, **features_dict)


if __name__ == "__main__":
    main()
