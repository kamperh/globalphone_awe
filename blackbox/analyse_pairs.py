#!/usr/bin/env python

"""
Analyse all pair-wise distances and compare to a number of other properties.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from os import path
from scipy.spatial.distance import pdist
from tqdm import tqdm
import argparse
import codecs
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

sys.path.append(path.join("..", "..", "src", "speech_dtw", "utils"))

from dp_align import DPEntry, DPError
import dp_align
import samediff


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("npz_fn", type=str, help="NumPy archive of embeddings")
    parser.add_argument(
        "--pronunciation", type=str,
        help="if provided, the pronunciations for this GlobalPhone "
        "language is used", choices=["BG", "CH", "CR", "CZ", "FR", "GE", "HA",
        "KO", "PL", "PO", "RU", "SP", "SW", "TH", "TU", "VN"], default=None
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def editdistance_array(labels):
    """
    Return an array of int in the same order as the distances from
    `scipy.spatial.distance.pdist` indicating the edit distance between all
    pairs of labels.
    """
    N = len(labels)
    edits = np.zeros(int(N*(N - 1)/2), dtype=int)

    # Calculate the edit distance for every pair of labels
    cur_edits_i = 0
    for n in tqdm(range(N - 1)):
        cur_label = labels[n]
        # distances = []
        for i_offset, test_label in enumerate(labels[n + 1:]):
            a = dp_align.dp_align(cur_label, test_label)
            edits[cur_edits_i + i_offset] = a.get_levenshtein()
            # print(
            #     "Distance between {} and {}: {}".format(cur_label, test_label,
            #     a.get_levenshtein())
            #     )
        # edits[cur_edits_i:cur_edits_i + (N - n) - 1] = distances
        # edits[cur_edits_i:cur_edits_i + (N - n) - 1] = np.asarray(
        #     labels[n + 1:]
        #     ) == cur_label
        cur_edits_i += N - n - 1

    return edits


def read_pronunciations(fn):
    pronunciations = {}
    with codecs.open(fn, "r", "utf-8") as f:
        for line in f:
            utt_key, pronunciation = line.strip().split()
            pronunciations[utt_key] = pronunciation.split(",")
    return pronunciations


#-----------------------------------------------------------------------------#
#                        SPECIALISED ALIGNMENT FUNCTION                       #
#-----------------------------------------------------------------------------#

def dp_align_edit_positions(ref_list, test_list, ins_penalty=3, del_penalty=3,
        sub_penalty=4):
    """
    Determines whether a edit operation occurs in the beginning, middle or end.

    Parameters
    ----------
    ref_list : list
    test_list : list

    Return
    ------
    dp_errors, edit_start, edit_middle, edit_end : DPError, (bool, bool, bool)
    """

    # Initialise the alignment matrix
    dp_matrix = np.empty(
        [len(test_list) + 1, len(ref_list) + 1], dtype = object
        )
    for i in range(len(test_list) + 1):
        for j in range(len(ref_list) + 1):
            dp_matrix[i][j] = DPEntry()

    # Initialise the origin
    dp_matrix[0][0].score = 0
    dp_matrix[0][0].align = "m"

    # The first row is all delections:
    for j in range(1, len(ref_list) + 1):
        dp_matrix[0][j].score = j*del_penalty
        dp_matrix[0][j].align = "d"

    # Fill dp_matrix
    for i in range(1, len(test_list) + 1):

        # First column is all insertions
        dp_matrix[i][0].score = i*ins_penalty
        dp_matrix[i][0].align = "i"

        for j in range(1, len(ref_list) + 1):
            del_score = dp_matrix[i, j - 1].score + del_penalty
            ins_score = dp_matrix[i - 1, j].score + ins_penalty

            if test_list[i - 1] == ref_list[j - 1]:

                # Considering a match
                match_score = dp_matrix[i - 1, j - 1].score

                # Test for a match
                if match_score <= del_score and match_score <= ins_score:
                    dp_matrix[i, j].score = match_score
                    dp_matrix[i, j].align = "m"
                # Test for a deletion
                elif del_score <= ins_score:
                    dp_matrix[i, j].score = del_score
                    dp_matrix[i, j].align = "d"
                # Test for an insertion (only option left)
                else:
                    dp_matrix[i, j].score = ins_score
                    dp_matrix[i, j].align = "i"

            else:

                # Considering a substitution
                sub_score = dp_matrix[i - 1, j - 1].score + sub_penalty

                # Test for a substitution
                if sub_score < del_score and sub_score <= ins_score:
                    dp_matrix[i, j].score = sub_score
                    dp_matrix[i, j].align = "s"
                # Test for a deletion
                elif del_score <= ins_score:
                    dp_matrix[i, j].score = del_score
                    dp_matrix[i, j].align = "d"
                # Test for an insertion (only option left)
                else:
                    dp_matrix[i, j].score = ins_score
                    dp_matrix[i, j].align = "i"

    # Perform alignment by tracking through the dp_matrix
    dp_errors = DPError()
    dp_errors.n_total = len(ref_list)
    i = len(test_list)
    j = len(ref_list)
    edit_start = False
    edit_end = False
    edit_middle = False
    while i > 0 or j > 0:
        if dp_matrix[i, j].align == "m":
            i -= 1
            j -= 1
            dp_errors.n_match += 1
        elif dp_matrix[i, j].align == "s":
            if i == len(test_list) and j == len(ref_list):
                edit_end = True
            elif i == 1 and j == 1:
                edit_start = True
            else:
                edit_middle = True
            i -= 1
            j -= 1
            dp_errors.n_sub += 1
        elif dp_matrix[i, j].align == "d":
            if i == len(test_list) and j == len(ref_list):
                edit_end = True
            elif i == 0 and j == 1:
                edit_start = True
            else:
                edit_middle = True
            j -= 1
            dp_errors.n_del += 1
        elif dp_matrix[i, j].align == "i":
            if i == len(test_list) and j == len(ref_list):
                edit_end = True
            elif i == 1 and j == 0:
                edit_start = True
            else:
                edit_middle = True
            i -= 1
            dp_errors.n_ins += 1

    # Return the alignment and edit positions
    return dp_errors, edit_start, edit_middle, edit_end


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print("Reading:", args.npz_fn)
    embeddings = np.load(args.npz_fn)

    # # Temp
    # data = {}
    # a = list(embeddings)
    # random.shuffle(a)
    # for key in a[:100]:
    #     data[key] = embeddings[key]
    # embeddings = data

    print("Ordering embeddings:")
    n_embeds = 0
    X = []
    utt_keys = []
    labels = []
    speakers = []
    for utt_key in tqdm(sorted(embeddings)):
        utt_keys.append(utt_key)
        X.append(embeddings[utt_key])
        utt_key = utt_key.split("_")
        label = utt_key[0]
        speaker = utt_key[1]
        labels.append(label)
        speakers.append(speaker)
    X = np.array(X)
    print("No. embeddings:", X.shape[0])
    print("Embedding dimensionality:", X.shape[1])

    # Normalise
    normed = (X - X.mean(axis=0)) / X.std(axis=0)
    X = normed

    print("Calculating distances")
    distances = pdist(X, metric="cosine")

    # Plot: Matching words
    print("Getting word matches")
    word_matches = samediff.generate_matches_array(labels)
    print("Total no. pairs:", word_matches.shape[0])
    print("No. same-word pairs:", sum(word_matches))
    distances_pos_avg = np.mean(distances[word_matches == True])
    distances_neg_avg = np.mean(distances[word_matches == False])
    distances_pos_std = np.std(distances[word_matches == True])
    distances_neg_std = np.std(distances[word_matches == False])
    plt.figure()
    plt.bar(
        [0, 1], [distances_neg_avg, distances_pos_avg],
        yerr=[distances_neg_std, distances_pos_std]
        )
    plt.xticks([0, 1], ("No", "Yes"))
    plt.xlabel("Matching words")
    plt.ylabel("Cosine distance")
    plt.ylim([0, 1.2])

    # Plot: Same speakers
    print("Getting speaker matches")
    speaker_matches = samediff.generate_matches_array(speakers)
    print("No. same-speaker pairs:", sum(speaker_matches))    
    distances_pos_avg = np.mean(
        distances[np.logical_and(word_matches, speaker_matches)]
        )
    distances_neg_avg = np.mean(
        distances[np.logical_and(word_matches, speaker_matches == False)]
        )
    distances_pos_std = np.std(
        distances[np.logical_and(word_matches, speaker_matches)]
        )
    distances_neg_std = np.std(
        distances[np.logical_and(word_matches, speaker_matches == False)]
        )
    # distances_pos_avg = np.mean(distances[speaker_matches == True])
    # distances_neg_avg = np.mean(distances[speaker_matches == False])
    # distances_pos_std = np.std(distances[speaker_matches == True])
    # distances_neg_std = np.std(distances[speaker_matches == False])
    plt.figure()
    plt.bar(
        [0, 1], [distances_neg_avg, distances_pos_avg],
        yerr=[distances_neg_std, distances_pos_std]
        )
    plt.xticks([0, 1], ("No", "Yes"))
    plt.xlabel("Matching speakers")
    plt.ylabel("Cosine distance")
    plt.ylim([0, 1.2])
    plt.title("Distances between same-word pairs")

    # Plot: Edit distances
    if args.pronunciation is not None:

        # Pronunciations
        pron_fn = path.join("lists", args.pronunciation, "dev.prons")
        print("Reading:", pron_fn)
        pronunciations = read_pronunciations(pron_fn)
        pron_labels = []
        for utt_key in utt_keys:
            pron_labels.append(pronunciations[utt_key])

        # Get distances
        print("Getting edit distances:")
        # edit_distances = editdistance_array(labels)
        edit_distances = editdistance_array(pron_labels)

        # Plot distances
        edits = sorted(set(edit_distances))
        averages = []
        stds = []
        for edit in edits:
            averages.append(np.mean(distances[edit_distances == edit]))
            stds.append(np.std(distances[edit_distances == edit]))
        plt.figure()
        plt.bar(edits, averages, yerr=stds)
        plt.ylim([0, 1.2])
        plt.xlabel("Phone edit distance")
        plt.ylabel("Cosine distance")

    plt.show()


if __name__ == "__main__":
    main()
