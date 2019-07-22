#!/usr/bin/env python

"""
Obtain the QbE costs for a given set of queries and search utterances.

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
import timeit

sys.path.append(path.join("..", "..", "src", "speech_dtw"))

from speech_dtw import qbe


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "--n_cpus", type=int,
        help="number of CPUs to parallelise over (default: %(default)s)",
        default=1
        )
    parser.add_argument(
        "feature_label", type=str,
        help="identifier for the set of queries and search utterances"
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

    # Read queries into a list
    fn = path.join("data", args.feature_label, "queries.npz")
    print("Reading:", fn)
    queries_dict = np.load(fn)
    queries_keys = sorted(queries_dict.keys())
    queries_list = [
        np.asarray(queries_dict[i], np.double) for i in queries_keys
        ]
    print("No. queries:", len(queries_list))

    # Read search collection into a list
    fn = path.join("data", args.feature_label, "search.npz")
    print("Reading:", fn)
    search_dict = np.load(fn)
    search_keys = sorted(search_dict.keys())
    search_list = [
        np.asarray(search_dict[i], np.double) for i in search_keys
        ]
    print("No. search items:", len(search_list))

    print(datetime.now())

    # Perform QbE
    print("Calculating costs: {} cores".format(args.n_cpus))
    start_time = timeit.default_timer()
    dtw_costs = qbe.parallel_dtw_sweep_min(
        queries_list, search_list, n_cpus=args.n_cpus
        )
    end_time = timeit.default_timer()
    duration = end_time - start_time
    print(datetime.now())
    print(
        "Avg. duration per comparison: {:.3f} sec".format(duration *
        args.n_cpus / (len(queries_list) * len(search_list)))
        )

    # Write costs
    cost_dict = {}
    for i_query, key_query in enumerate(queries_keys):
        if key_query not in cost_dict:
            cost_dict[key_query] = {}
        for i_search, key_search in enumerate(search_keys):
            cost_dict[key_query][key_search] = dtw_costs[i_query][i_search]
    output_dir = path.join("exp", args.feature_label, "dtw")
    if not path.isdir(output_dir):
        os.makedirs(output_dir)
    fn = path.join(output_dir, "cost_dict.pkl")
    print("Writing:", fn)
    with open(fn, "wb") as f:
        pickle.dump(cost_dict, f, -1)

    print(datetime.now())


if __name__ == "__main__":
    main()
