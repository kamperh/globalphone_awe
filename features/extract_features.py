#!/usr/bin/env python

"""
Extract MFCC features for a particular GlobalPhone language.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from os import path
from tqdm import tqdm
import argparse
import glob
import numpy as np
import os
import shutil
import sys

sys.path.append("..")

from paths import gp_data_dir, gp_alignments_dir
import features
import utils

shorten_bin = path.join("..", "..", "src", "shorten-3.6.1", "bin", "shorten")
language_codes = {
    "AR": "Arabic", "BG": "Bulgarian", "CR": "Croatian", "CZ": "Czech", "FR":
    "French", "GE": "German", "HA": "Hausa", "JA": "Japanese", "KO": "Korean",
    "CH": "Mandarin", "PL": "Polish", "PO": "Portuguese", "RU": "Russian",
    "SA": "Swahili", "SP": "Spanish", "SW": "Swedish", "TA": "Tamil", "TH":
    "Thai", "TU": "Turkish", "VN": "Vietnamese", "WU": "Wu"
    }


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
        "RU", "SP", "SW", "TH", "TU", "VN", ]
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def shorten_to_wav(language, speakers, output_dir):
    """Convert audio in shorten format to wav."""

    # Source filenames
    shorten_files = []
    if language == "HA":
        # Hausa needs special treatment because its audio is not shortened
        shorten_dir = path.join(
            gp_data_dir, language_codes[language], "Hausa", "Data", "adc"
            )
        for speaker in speakers:
            shorten_files.extend(
                glob.glob(path.join(shorten_dir, speaker, "*.adc"))
                )       
    else:
        shorten_dir = path.join(gp_data_dir, language_codes[language], "adc")
        for speaker in speakers:
            shorten_files.extend(
                glob.glob(path.join(shorten_dir, speaker, "*.shn"))
                )

    # Convert to wav
    for shorten_fn in tqdm(shorten_files):
        basename = path.split(shorten_fn)[-1].split(".")[0]
        raw_fn = path.join(output_dir, basename + ".raw")
        wav_fn = path.join(output_dir, basename + ".wav")
        if not path.isfile(raw_fn):
            if language == "HA":
                # Special treatment for Hausa
                shutil.copyfile(shorten_fn, raw_fn)
            else:
                utils.shell(shorten_bin + " -x " + shorten_fn + " " + raw_fn)
            assert path.isfile(raw_fn)
        if not path.isfile(wav_fn):
            utils.shell(
                "sox -t raw -r 16000 -e signed-integer -b 16 " + raw_fn +
                " -t wav " + wav_fn
                )
            assert path.isfile(wav_fn)
        if path.isfile(raw_fn):
            os.remove(raw_fn)


def read_speakers(speakers_fn, language):
    with open(speakers_fn) as f:
        for line in f:
            line = line.strip().split()
            if line[0] == language:
                return line[1:]
    assert False


def extract_features_for_subset(language, subset, feat_type, output_fn):
    """
    Extract features for the subset in this language.

    The `feat_type` parameter can be "mfcc" or "fbank".
    """

    # Get speakers for subset
    speakers_fn = path.join("..", "data", subset + "_spk.list")
    print("Reading:", speakers_fn)
    speakers = read_speakers(speakers_fn, language)

    # Convert shorten audio to wav
    wav_dir = path.join("wav", language, subset)
    if not path.isdir(wav_dir):
        os.makedirs(wav_dir)
    print("Converting .shn audio to .wav:")
    shorten_to_wav(language, speakers, wav_dir)

    # Extract raw features
    print("Extracting features:")
    if feat_type == "mfcc":
        feat_dict = features.extract_mfcc_dir(wav_dir)
    elif feat_type == "fbank":
        feat_dict = features.extract_fbank_dir(wav_dir)
    else:
        assert False, "invalid feature type"

    # Perform per speaker mean and variance normalisation
    print("Per speaker mean and variance normalisation:")
    feat_dict = features.speaker_mvn(feat_dict)

    # Write output
    print("Writing:", output_fn)
    np.savez_compressed(output_fn, **feat_dict)



#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    feat_type = "mfcc"


    # RAW FEATURES

    # Extract MFCCs for the different sets
    feat_dir = path.join(feat_type, args.language)
    if not path.isdir(feat_dir):
        os.makedirs(feat_dir)
    for subset in ["train"]:  #, ["dev", "eval", "train"]:  # Temp
        raw_feat_fn = path.join(
            feat_dir, args.language.lower() + "." + subset + ".npz"
            )
        if not path.isfile(raw_feat_fn):
            print("Extracting MFCCs:", subset)
            extract_features_for_subset(
                args.language, subset, feat_type, raw_feat_fn
                )
        else:
            print("Using existing file:", raw_feat_fn)


    # GROUND TRUTH WORD SEGMENTS

    for subset in ["train"]:  #, ["dev", "eval", "train"]:  # Temp
    
        # Create a ground truth word list of at least 50 frames and 5 characters
        fa_fn = path.join(gp_alignments_dir, args.language, subset + ".ctm")
        list_dir = path.join("lists", args.language)
        list_fn = path.join(list_dir, subset + ".gt_words.list")
        if not path.isdir(list_dir):
            os.makedirs(list_dir)
        if not path.isfile(list_fn):
            utils.filter_words(fa_fn, list_fn)
        else:
            print("Using existing file:", list_fn)

        # Extract word segments from the MFCC NumPy archives
        input_npz_fn = path.join(
            feat_dir, args.language.lower() + "." + subset + ".npz"
            )
        output_npz_fn = path.join(
            feat_dir, args.language.lower() + "." + subset + ".gt_words.npz"
            )
        if not path.isfile(output_npz_fn):
            print("Extracting MFCCs for ground truth word tokens:", subset)
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
        else:
            print("Using existing file:", output_npz_fn)


if __name__ == "__main__":
    main()
