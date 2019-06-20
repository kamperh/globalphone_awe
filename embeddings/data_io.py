"""
Data input and output functions.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018, 2019
"""

from os import path
import numpy as np
import sys

sys.path.append(path.join("..", "src"))

from tflego import NP_DTYPE



def load_data_from_npz(npz_fn, min_length=None):
    print("Reading:", npz_fn)
    npz = np.load(npz_fn)
    x = []
    labels = []
    speakers = []
    lengths = []
    keys = []
    n_items = 0
    for utt_key in sorted(npz):
        if min_length is not None and len(npz[utt_key]) <= min_length:
            continue
        keys.append(utt_key)
        x.append(npz[utt_key])
        word = utt_key.split("_")[0]
        speaker = utt_key.split("_")[1][:3]
        labels.append(word)
        speakers.append(speaker)
        lengths.append(npz[utt_key].shape[0])
        n_items += 1
    print("No. items:", n_items)
    print("E.g. item shape:", x[0].shape)
    return (x, labels, lengths, keys, speakers)


def trunc_and_limit_dim(x, lengths, d_frame, max_length):
    for i, seq in enumerate(x):
        x[i] = x[i][:max_length, :d_frame]
        lengths[i] = min(lengths[i], max_length)


def pad_sequences(x, n_padded, center_padded=True, return_mask=False):
    """Return the padded sequences and their original lengths."""
    padded_x = np.zeros((len(x), n_padded, x[0].shape[1]), dtype=NP_DTYPE)
    if return_mask:
        mask_x = np.zeros((len(x), n_padded), dtype=NP_DTYPE)
    lengths = []
    for i_data, cur_x in enumerate(x):
        length = cur_x.shape[0]
        if center_padded:
            padding = int(np.round((n_padded - length) / 2.))
            if length <= n_padded:
                padded_x[i_data, padding:padding + length, :] = cur_x
                if return_mask:
                    mask_x[i_data, padding:padding + length] = 1
            else:
                # Cut out snippet from sequence exceeding n_padded
                padded_x[i_data, :, :] = cur_x[-padding:-padding + n_padded]
                if return_mask:
                    mask_x[i_data, :] = 1
            lengths.append(min(length, n_padded))
        else:
            length = min(length, n_padded)
            padded_x[i_data, :length, :] = cur_x[:length, :]
            if return_mask:
                mask_x[i_data, :length] = 1
            lengths.append(length)
    if return_mask:
        return padded_x, lengths, mask_x
    else:
        return padded_x, lengths

