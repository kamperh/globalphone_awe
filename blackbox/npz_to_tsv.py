#!/usr/bin/env python

"""
Convert a NumPy archive to a TSV file for visualising embeddings.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from os import path
from tqdm import tqdm
import argparse
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
    parser.add_argument("npz_fn", type=str, help="input NumPy archive")
    parser.add_argument(
        "tsv_fn", type=str, help="output TSV file; if 'auto', then an output "
        "filename is generated automatically based on the input filename"
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

    print("Reading:", args.npz_fn)
    features = np.load(args.npz_fn)
    
    if args.tsv_fn == "auto":
        npz_fn_split = path.split(args.npz_fn)
        args.tsv_fn = (
            path.split(npz_fn_split[-2])[-1] + "." +
            path.splitext(npz_fn_split[-1])[0] + ".tsv"
            )
    metadata_fn = args.tsv_fn + ".metadata"
    print("Writing:", args.tsv_fn)
    print("Writing:", metadata_fn)
    with open(args.tsv_fn, "w") as f_tsv, open(metadata_fn, "w") as f_metadata:
        f_metadata.write("word\tspeaker\n")
        for utt_key in tqdm(sorted(features)):
            f_tsv.write(
                "\t".join(["{:.5f}".format(i) for i in features[utt_key]]) +
                "\n"
                )
            utt_key_split = utt_key.split("_")
            f_metadata.write(utt_key_split[0] + "\t" + utt_key_split[1] + "\n")


if __name__ == "__main__":
    main()
