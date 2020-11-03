#!/usr/bin/env python

"""
Analyse UTD pairs for the indicated language.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2020
"""

from os import path
from tqdm import tqdm
import argparse
import codecs
import glob
import numpy as np
import os
import shutil
import sys

sys.path.append("..")

from extract_features import get_overlap
from paths import gp_data_dir, gp_alignments_dir


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
        choices=["BG", "CH", "CR", "CZ", "FR", "GE", "HA", "KO", "PL", "PO",
        "RU", "SP", "SW", "TH", "TU", "VN"]
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
    subset = "train"

    # Read UTD terms
    utd_list_fn = path.join("lists", args.language, "train.utd_terms.list")
    print("Reading:", utd_list_fn)
    # overlap_dict[speaker_utt][(start, end)] is list a tuples of
    # (label, (start, end), overlap, cluster_label)
    overlap_dict = {}
    with codecs.open(utd_list_fn, "r", "utf-8") as utd_list_f:
        for line in utd_list_f:
            term, speaker, utt, start_end = line.strip().split("_")
            start, end = start_end.split("-")
            start = int(start)
            end = int(end)
            if not speaker + "_" + utt in overlap_dict:
                overlap_dict[speaker + "_" + utt] = {}
            overlap_dict[speaker + "_" + utt][(start, end, term)] = []

    # Read forced alignments
    fa_fn = path.join(gp_alignments_dir, args.language, subset + ".ctm")
    print("Reading:", fa_fn)
    fa_dict = {}
    with codecs.open(fa_fn, "r", "utf-8") as fa_f:
        for line in fa_f:
            utt_key, _, start, duration, label = line.strip().split()
            start = float(start)
            duration = float(duration)
            end = start + duration
            start_frame = int(round(start*100))
            end_frame = int(round(end*100))
            if (label != "<unk>" and label != "sil" and label != "?" and
                    label != "spn"):
                if not utt_key in fa_dict:
                    fa_dict[utt_key] = {}
                fa_dict[utt_key][start_frame, end_frame] = label

    # Find ground truth terms with maximal overlap
    print("Getting ground truth terms with overlap:")
    overlap_label_dict = {}
    for utt_key in tqdm(fa_dict):
        # print(utt_key)
        if utt_key not in overlap_dict:
            continue
        for (fa_start, fa_end) in fa_dict[utt_key]:
            for (utd_start, utd_end, utd_term) in overlap_dict[utt_key]:
                overlap = get_overlap(
                    utd_start, utd_end, fa_start, fa_end
                    )
                if overlap == 0:
                    continue
                overlap_dict[utt_key][(utd_start, utd_end, utd_term)].append((
                    fa_dict[utt_key][(fa_start, fa_end)],
                    (fa_start, fa_end), overlap
                    ))
                term_key = "{}_{}_{:06d}-{:06d}".format(
                    utd_term, utt_key, utd_start, utd_end
                    )
                if not term_key in overlap_label_dict:
                    overlap_label_dict[term_key] = set()
                overlap_label_dict[term_key].add(
                    fa_dict[utt_key][(fa_start, fa_end)]
                    )

    # Read UTD pairs
    pairs_fn = path.join("lists", args.language, "train.utd_pairs.list")
    pairs = []
    n_pairs = 0
    n_correct = 0
    n_missing = 0
    with codecs.open(pairs_fn, "r", "utf-8") as pairs_f:
        for line in pairs_f:
            term1, term2 = line.strip().split(" ")
            pairs.append((term1, term2))
            if (term1 not in overlap_label_dict or term2 not in
                    overlap_label_dict):
                n_missing += 1
                continue
            if (len(overlap_label_dict[term1].intersection(
                    overlap_label_dict[term2])) > 0):
                n_correct += 1
            n_pairs += 1
    print("Correct pairs: {:.2f}%".format(n_correct/n_pairs*100.0))
    print("No. missing pairs: {} out of {}".format(n_missing, n_pairs))



    # # Construct list of UTD labels and list of list of overlapping GT terms
    # labels = []
    # overlap_lists = []
    # for utt_key in tqdm(overlap_dict):
    #     for (utd_start, utd_end, utd_term) in overlap_dict[utt_key]:
    #         overlap_list = overlap_dict[utt_key][
    #             (utd_start, utd_end, utd_term)
    #             ]
    #         if len(overlap_list) == 0:
    #             continue
    #         labels.append(utd_term)
    #         overlap_lists.append([i[0] for i in overlap_list])



if __name__ == "__main__":
    main()
