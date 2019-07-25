#!/usr/bin/env python

"""
Calculate costs for dense search.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017, 2019
"""

from datetime import datetime
from os import path
from scipy.spatial.distance import cdist
from tqdm import tqdm
import argparse
import cPickle as pickle
import numpy as np
import os
import sys
import timeit


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("eval_dir", type=str, help="evaluation directory")
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean", "hamming", "chebyshev",
        "symsumxentropy"],
        default="cosine", help="distance metric (default: %(default)s)"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def sweep_min(query_vec, search_array, metric):
    """
    Return the minimum cost between `query_vec` and rows of `search_array`.
    """
    if metric == "symsumxentropy":
        return np.min(
            cdist_sumxentropy(np.array([query_vec]), search_array, True)
            )
    else:
        return np.min(cdist(np.array([query_vec]), search_array, metric))



def cdist_sumxentropy(queries_array, search_array, symmetric=False):
    distances = np.zeros((queries_array.shape[0], search_array.shape[0]))
    for i_query, query_vec in enumerate(queries_array):
        for i_search, search_vec in enumerate(search_array):
            if symmetric:
                distances[i_query, i_search] = sumxentroy(
                    search_vec, query_vec
                    ) + sumxentroy(query_vec, search_vec)
            else:
                distances[i_query, i_search] = sumxentroy(
                    search_vec, query_vec
                    )
    return distances


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    print(datetime.now())

    # Read queries
    fn = path.join(args.eval_dir, "queries.npz")
    if not path.isfile(fn):
        import re
        fn = path.join(
            re.sub("min\_.*step\_\d*\.", "", args.eval_dir), "queries.npz"
            )
    print("Reading:", fn)
    queries_dict = np.load(fn)
    queries_keys = sorted(queries_dict.keys())
    # queries_keys = queries_keys[:10]
    queries_list = [queries_dict[i] for i in queries_keys]
    print("No. queries:", len(queries_list))
    print("Query array shape:", queries_dict[queries_dict.keys()[0]].shape)

    # Read search collection
    fn = path.join(args.eval_dir, "search.npz")
    print("Reading:", fn)
    search_dict = np.load(fn)
    search_keys = sorted(search_dict.keys())
    search_list = [search_dict[i] for i in search_keys]
    print("No. search utterances:", len(search_list))
    print("Search array shape:", search_dict[search_dict.keys()[0]].shape)

    # print(datetime.now())

    print("Calculating costs:")
    start_time = timeit.default_timer()
    costs = []
    for query_vec in tqdm(queries_list):
        for search_array in search_list:
            costs.append(sweep_min(
                query_vec, search_array, args.metric
                ))
    end_time = timeit.default_timer()
    n_search = len(search_list)
    costs = [
        costs[i*n_search:(i + 1)*n_search] for i in
        range(int(np.floor(len(costs)/n_search)))
        ]
    duration = end_time - start_time
    print(
        "Avg. duration per comparison: {:.3f} sec".format(duration /
        (len(queries_list) * len(search_list)))
        )

    # Write costs
    fn = path.join(args.eval_dir, "cost_dict." + args.metric + ".pkl")
    print("Writing:", fn)
    cost_dict = {}
    for i_query, key_query in enumerate(queries_keys):
        if key_query not in cost_dict:
            cost_dict[key_query] = {}
        for i_search, key_search in enumerate(search_keys):
            cost_dict[key_query][key_search] = costs[i_query][i_search]
    # print(datetime.now())
    with open(fn, "wb") as f:
        pickle.dump(cost_dict, f, -1)

    print(datetime.now())


if __name__ == "__main__":
    main()
