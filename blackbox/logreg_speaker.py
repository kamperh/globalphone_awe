#!/usr/bin/env python

"""
Use logistic regression for speaker classification.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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
    parser.add_argument("npz_fn", type=str, help="NumPy archive of embeddings")
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

    # # Temp
    # import random
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
    words = []
    speakers = []
    for utt_key in tqdm(sorted(embeddings)):
        utt_keys.append(utt_key)
        X.append(embeddings[utt_key])
        utt_key = utt_key.split("_")
        word = utt_key[0]
        speaker = utt_key[1]
        words.append(word)
        speakers.append(speaker)
    X = np.array(X)
    print("No. embeddings:", X.shape[0])
    print("Embedding dimensionality:", X.shape[1])

    # Convert words to IDs
    speaker_set = set(speakers)
    speaker_to_id = dict(
        zip(sorted(list(speaker_set)), range(len(speaker_set)))
        )
    id_to_speaker = dict([[v,k] for k, v in speaker_to_id.items()])
    y = []
    for speaker in speakers:
        y.append(speaker_to_id[speaker])
    y = np.array(y, dtype=int)
    print("No. speakers:", len(speaker_to_id))

    # Split training and test sets 80/20
    indices = np.arange(X.shape[0])
    np.random.seed(1)
    np.random.shuffle(indices)
    n_train = int(round(X.shape[0]*0.8))
    X_train = X[indices[:n_train]]
    X_test = X[indices[n_train:]]
    y_train = y[indices[:n_train]]
    y_test = y[indices[n_train:]]
    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)

    # Multi-class logistic regression
    print(datetime.now())
    print("Fitting multi-class logistic regression model")
    logreg = LogisticRegression(
        C=1e5, solver="lbfgs", multi_class="multinomial"
        # solver="lbfgs", multi_class="ovr", max_iter=200
        )
    logreg.fit(X_train, y_train)
    print(datetime.now())

    # Predict classes
    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Speaker classification accuracy: {:.2f}%".format(accuracy*100))
    print(
        classification_report(y_test, y_pred,
        target_names=[id_to_speaker[i] for i in range(max(y) + 1)])
        )

if __name__ == "__main__":
    main()
