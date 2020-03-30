#!/usr/bin/env python

"""
Print the contents of a pickled dictionary.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2020
"""

from os import path
import argparse
import pickle
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("pickle_dict_fn", type=str, help="pickled dictionary")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    if path.isfile(args.pickle_dict_fn):
        with open(args.pickle_dict_fn, "rb") as f:
            d = pickle.load(f)
        for key in sorted(d):
            print(key, ":", d[key])


if __name__ == "__main__":
    main()
