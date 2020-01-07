#!/usr/bin/env python

"""
Evaluate QbE performance for a given costs directory.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017, 2019
"""

from collections import Counter
from os import path
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from tqdm import tqdm
import argparse
import codecs
import pickle
import numpy as np
import sklearn.metrics as metrics
import sys

sys.path.append("..")

from paths import gp_alignments_dir


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
        choices=["HA"]
        )
    parser.add_argument(
        "cost_dict_fn", type=str,
        help="filename of the cost dictionary"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()



#-----------------------------------------------------------------------------#
#                             EVALUATION FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def calculate_eer(y_true, y_score):
    # https://yangcha.github.io/EER-ROC/
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer


def calculate_auc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    return metrics.auc(fpr, tpr)


def eval_precision_recall_fscore(cost_dict, label_dict, threshold,
        analyse=False):
    """Evaluate precision and recall for a particular output."""

    # # Get average scores
    # avg_keyword_scores = {}
    # for keyword in cost_dict:
    #     scores = []
    #     for utt in cost_dict[keyword]:
    #         scores.append(cost_dict[keyword][utt])
    #     avg_keyword_scores[keyword] = np.mean(scores)
    # print(avg_keyword_scores)

    # For each utterance, which keywords above threshold
    threshold_dict = {}
    for keyword in cost_dict:
        for utt in cost_dict[keyword]:
            if utt not in threshold_dict:
                threshold_dict[utt] = []
            if cost_dict[keyword][utt] <= threshold:
            # if (cost_dict[keyword][utt] <=
            #         avg_keyword_scores[keyword]*threshold):
                threshold_dict[utt].append(keyword)
    keywords = cost_dict.keys()

    # Calculate precision and recall
    n_tp = 0
    n_pred = 0
    n_true = 0
    word_tokens_correct = []
    if analyse:
        print()
    for utt in sorted(threshold_dict):
        if utt not in label_dict:
            continue
        y_pred = threshold_dict[utt]
        y_true = [i for i in label_dict[utt].split() if i in keywords]
        cur_tokens_correct = set([i for i in y_true if i in y_pred])
        word_tokens_correct.extend(cur_tokens_correct)
        n_tp += len(cur_tokens_correct)
        n_pred += len(y_pred)
        n_true += len(set(y_true))
        if analyse:
            if len(y_pred) > 0:
                print("-"*79)
                print("Utterance:", utt)
                print("Predicted:", sorted(y_pred))
                print("Ground truth:", y_true)
                if n_pred > 0:
                    print(
                        "Current precision: {} / {} = {:.4f}".format( n_tp,
                        n_pred, float(n_tp)/n_pred*100.)
                        )
                if n_true > 0:
                    print(
                        "Current recall: {} / {} = {:.4f}".format(
                        n_tp, n_true, float(n_tp)/n_true*100.)
                        )
    precision = float(n_tp)/n_pred if n_pred != 0 else 0
    recall = float(n_tp)/n_true
    f_score = (
        2*precision*recall/(precision + recall) if precision + recall != 0 else
        0
        )

    if analyse:
        print("-"*79)
        print
        print(
            "Most common correctly predicted words:",
            Counter(word_tokens_correct).most_common(15)
            )

    return n_tp, n_pred, n_true, precision, recall, f_score


def eval_qbe(cost_dict, label_dict, analyse=False):
    """
    Return dictionaries of P@10, P@N and EER for each query item.

    The keys of each of the returned dictionaries are the unique keyword types,
    with the value a list of the scores for each of the queries of that keyword
    type.
    """

    # Unique keywords with query keys
    keyword_dict = {}
    for query_key in cost_dict:
        keyword = query_key.split("_")[0]
        if keyword not in keyword_dict:
            keyword_dict[keyword] = []
        keyword_dict[keyword].append(query_key)

    # For each keywords
    eer_dict = {}  # `eer_dict[keyword]` is a list of EER scores for each query
                   # of that keyword type
    auc_dict = {}
    p_at_10_dict = {}
    p_at_n_dict = {}
    if analyse:
        print()
    for keyword in tqdm(sorted(keyword_dict)):

        eer_dict[keyword] = []
        auc_dict[keyword] = []
        p_at_10_dict[keyword] = []
        p_at_n_dict[keyword] = []

        # For each query key
        for query_key in sorted(keyword_dict[keyword]):

            # Rank search keys
            utt_order = [
                utt_key for utt_key in sorted(cost_dict[query_key],
                key=cost_dict[query_key].get) if utt_key in label_dict
                ]

            # EER
            y_true = []
            for utt_key in utt_order:
                if keyword in label_dict[utt_key]:
                    y_true.append(1)
                else:
                    y_true.append(0)
            y_score = [cost_dict[query_key][utt_key] for utt_key in utt_order]
            cur_eer = calculate_eer(y_true, [-i for i in y_score])
            cur_auc = calculate_auc(y_true, [-i for i in y_score])
            eer_dict[keyword].append(cur_eer)
            auc_dict[keyword].append(cur_auc)

            # P@10
            cur_p_at_10 = float(sum(y_true[:10]))/10.
            p_at_10_dict[keyword].append(cur_p_at_10)

            # P@N
            cur_p_at_n = np.float64(sum(y_true[:sum(y_true)]))/sum(y_true)
            p_at_n_dict[keyword].append(cur_p_at_n)

            if analyse:
                print("-"*79)
                print("Query:", query_key)
                print("Current P@10: {:.4f}".format(cur_p_at_10))
                print("Current P@N: {:.4f}".format(cur_p_at_n))
                print("Current EER: {:.4f}".format(cur_eer))
                print("Current AUC: {:.4f}".format(cur_auc))
                # print("Top 10 utterances: ", utt_order[:10])
                print("Top 10 utterances:")
                for i_utt, utt in enumerate(utt_order[:10]):
                    print("{}: {}".format(
                        # utt, " ".join(label_dict[utt])), end=''
                        utt, label_dict[utt]), end=''
                        )
                    if y_true[i_utt] == 0:
                        print(" *")
                    else:
                        print()

    if analyse:
        print("-"*79)
        print()

    return eer_dict, auc_dict, p_at_10_dict, p_at_n_dict


def get_avg_scores(score_dict):
    """
    Return the overall average, and unweighted average, median and maximum
    scores over all keyword types.

    Return
    ------
    avg_all_scores, avg_avg_scores, avg_median_scores, avg_max_scores
    """
    all_scores = []
    avg_scores = []
    median_scores = []
    max_scores = []
    min_scores = []

    for keyword in score_dict:
        all_scores.extend(score_dict[keyword])
        avg_scores.append(np.mean(score_dict[keyword]))
        median_scores.append(np.median(score_dict[keyword]))
        max_scores.append(np.max(score_dict[keyword]))
        min_scores.append(np.min(score_dict[keyword]))

    avg_all_scores = np.mean(all_scores)
    avg_avg_scores = np.mean(avg_scores)
    avg_median_scores = np.mean(median_scores)
    avg_max_scores = np.mean(max_scores)
    avg_min_scores = np.mean(min_scores)

    return (
        avg_all_scores, avg_avg_scores, avg_median_scores, avg_max_scores,
        avg_min_scores
        )

def read_forced_alignment(globalphone_fa_fn):
    """
    Return a dictionary of transcriptions obtained from a GlobalPhone forced
    alignment file.
    """
    transcription_dict = {}
    with codecs.open(globalphone_fa_fn, "r", "utf-8") as f:
        for line in f:
            line = line.strip().split(" ")
            utterance_key = line[0]
            label = line[4].lower()
            if utterance_key not in transcription_dict:
                transcription_dict[utterance_key] = label
                # transcription_dict[utterance_key] = []
            else:
                transcription_dict[utterance_key] += " " + label
            # transcription_dict[utterance_key].append(label)
    return transcription_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    fn = path.join(args.cost_dict_fn)
    print("Reading:", fn)
    with open(fn, "rb") as f:
        cost_dict = pickle.load(f)
    print(
        "Keywords: " + ", ".join(sorted(set([i.split("_")[0] for i in
        cost_dict.keys()])))
        )

    globalphone_fa_fn = path.join(gp_alignments_dir, args.language, "eval.ctm")
    print("Reading:", globalphone_fa_fn)
    transcription_dict = read_forced_alignment(globalphone_fa_fn)
    # print(transcription_dict)

    print("Evaluating:")
    eer_dict, auc_dict, p_at_10_dict, p_at_n_dict = eval_qbe(
        cost_dict, transcription_dict
        )

    eer_overall, eer_avg, eer_median, eer_max, eer_min = get_avg_scores(
        eer_dict
        )
    auc_overall, auc_avg, auc_median, auc_max, auc_min = get_avg_scores(
        auc_dict
        )
    p_at_10_overall, p_at_10_avg, p_at_10_median, p_at_10_max, p_at_10_min = (
        get_avg_scores(p_at_10_dict)
        )
    p_at_n_overall, p_at_n_avg, p_at_n_median, p_at_n_max, p_at_n_min = (
        get_avg_scores(p_at_n_dict)
        )

    print()
    print("-"*79)
    print(
        "EER:  {:.4f}, avg: {:.4f}, median: {:.4f}, max: {:.4f}, "
        "min: {:.4f}".format(eer_overall, eer_avg, eer_median, eer_max,
        eer_min)
        )
    print(
        "AUC:  {:.4f}, avg: {:.4f}, median: {:.4f}, max: {:.4f}, "
        "min: {:.4f}".format(auc_overall, auc_avg, auc_median, auc_max,
        auc_min)
        )
    print(
        "P@10: {:.4f}, avg: {:.4f}, median: {:.4f}, max: {:.4f}, "
        "min: {:.4f}".format(p_at_10_overall, p_at_10_avg, p_at_10_median,
        p_at_10_max, p_at_10_min)
        )
    print(
        "P@N:  {:.4f}, avg: {:.4f}, median: {:.4f}, max: {:.4f}, "
        "min: {:.4f}".format(p_at_n_overall, p_at_n_avg, p_at_n_median,
        p_at_n_max, p_at_n_min)
        )
    print("-"*79)


if __name__ == "__main__":
    main()
