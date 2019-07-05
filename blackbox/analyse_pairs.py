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


def generate_editdistance_array(labels):
    """
    Return an array of int in the same order as the distances from
    `scipy.spatial.distance.pdist` indicating the edit distance between all
    pairs of labels.
    """
    N = len(labels)
    edits = np.zeros(int(N*(N - 1)/2), dtype=int)

    # For every distance, mark whether it is a true match or not
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
        pronunciations = read_pronunciations(fn)
        pron_labels = []
        for utt_key in utt_keys:
            pron_labels.append(pronunciations[utt_key])

        # Get distances
        print("Getting edit distances:")
        # edit_distances = generate_editdistance_array(labels)
        edit_distances = generate_editdistance_array(pron_labels)

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
