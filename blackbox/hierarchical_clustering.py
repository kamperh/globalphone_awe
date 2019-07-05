#!/usr/bin/env python

"""
Apply agglomerative clustering to embeddings and plot a labelled dendrogram.

See
https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from os import path
from scipy.cluster.hierarchy import dendrogram, linkage
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import sys


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
        "--n_samples", type=int,
        help="if given, the embeddings are subsampled"
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
    embeddings = np.load(args.npz_fn)

    if args.n_samples is not None:
        utt_keys = list(embeddings)
        random.seed(1)
        random.shuffle(utt_keys)
        new_embeddings = {}
        for utt_key in utt_keys[:args.n_samples]:
            new_embeddings[utt_key] = embeddings[utt_key]
        embeddings = new_embeddings

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

    # Get a speaker colour map
    # cmap = plt.cm.jet
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    speakers_set = set(speakers)
    n_speakers = len(speakers_set)
    speaker_to_color = {}
    for i_speaker, speaker in enumerate(sorted(list(speakers_set))):
        speaker_to_color[speaker] = cmaplist[
            int(i_speaker/n_speakers * (len(cmaplist) - 1))
            ]
    # speakers_to_id = dict(
    #     zip(sorted(list(speakers_set)), range(len(speakers_set)))
    #     )
    # speakers_to_color = {}
    # for speaker in speakers_to_id:
    #     speakers_to_color[speaker] = 

    # Cluster
    print("Clustering")
    Z = linkage(X, method="ward", metric="euclidean")

    # Plot dendrogram
    print("Plotting")
    plt.figure()
    R = dendrogram(
        Z,
        leaf_rotation=90,
        leaf_font_size=8,
        labels=labels
        )
    leaves = R["leaves"]

    ax = plt.gca()
    x_labels = ax.get_xmajorticklabels()
    for i, x in enumerate(x_labels):
        x.set_color(speaker_to_color[speakers[leaves[i]]])
        # c = 
        # print(x.get_text(), labels[leaves[i]], speakers[leaves[i]])
        # x.set_color(colorDict[x.get_text()])

    plt.show()



if __name__ == "__main__":
    main()
