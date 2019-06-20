#!/usr/bin/env python

"""
Write the keys in a given Numpy archive.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2018, 2019
"""

import argparse
import sys
import numpy as np

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("npz_fn", type=str, help="the Numpy archive")
    parser.add_argument(
        "keys_fn", type=str, help="the file to write the keys to"
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

    npz = np.load(args.npz_fn)

    print("Writing keys:", args.keys_fn)
    open(args.keys_fn, "w").write("\n".join(npz.keys()) + "\n")


if __name__ == "__main__":
    main()
