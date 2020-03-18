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
import codecs
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
        "RU", "SP", "SW", "TH", "TU", "VN"]
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

    assert len(shorten_files) > 0, "no audio found; check paths.py"

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
            if not path.isfile(raw_fn):
                print(
                    "Warning: File not converted:", path.split(shorten_fn)[-1]
                    )
                continue
            # assert path.isfile(raw_fn)
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
    assert False, "invalid language"


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
    print("Converting shn audio to wav:")
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


def get_overlap(ref_start, ref_end, test_start, test_end):
    """Calculate the number of frames of `test` that overlaps with `ref`."""

    # Check whether there is any overlap
    if test_end <= ref_start or test_start >= ref_end:
        return 0

    n_overlap = ref_end - ref_start
    n_overlap -= max(0, test_start - ref_start)
    n_overlap -= max(0, ref_end - test_end)

    return n_overlap


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
    for subset in ["dev", "eval", "train"]:
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

    # assert False


    # GROUND TRUTH WORD SEGMENTS

    list_dir = path.join("lists", args.language)
    if not path.isdir(list_dir):
        os.makedirs(list_dir)
    for subset in ["dev", "eval", "train"]:
    
        # Create a ground truth word list (at least 50 frames and 5 characters)
        fa_fn = path.join(gp_alignments_dir, args.language, subset + ".ctm")
        list_fn = path.join(list_dir, subset + ".gt_words.list")
        if not path.isfile(list_fn):
            if args.language == "KO":
                min_frames = 26
                min_chars = 3
            elif args.language == "TH":
                min_frames = 38
                min_chars = 2
            elif args.language == "VN":
                min_frames = 30
                min_chars = 4
            else:
                min_frames = 50
                min_chars = 5
            utils.filter_words(
                fa_fn, list_fn, min_frames=min_frames, min_chars=min_chars
                )
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


    # UTD-DISCOVERED WORD SEGMENTS

    # Change Enno Hermann's pair file to the appropriate format
    enno_pairs_fn = path.join(
        "..", "data", args.language, # "pairs_sw_utd.train"
        "pairs_sw_utd_plp_vtln.train"
        )
    if not path.isfile(enno_pairs_fn):
        # This might not be an evaluation language
        return
    pairs_fn = path.join("lists", args.language, "train.utd_pairs.list")
    if not path.isfile(pairs_fn):
        utils.format_enno_pairs(enno_pairs_fn, pairs_fn)
    else:
        print("Using existing file:", pairs_fn)
    list_fn = path.join("lists", args.language, "train.utd_terms.list")
    if not path.isfile(list_fn):
        print("Reading:", pairs_fn)
        terms = set()
        with codecs.open(pairs_fn, "r", "utf-8") as pairs_f:
            for line in pairs_f:
                term1, term2 = line.strip().split(" ")
                terms.add(term1)
                terms.add(term2)
        print("Writing:", list_fn)
        with codecs.open(list_fn, "w", "utf-8") as list_f:
            for term in sorted(terms):
                list_f.write(term + "\n")
    else:
        print("Using existing file:", list_fn)

    # Extract UTD segments
    input_npz_fn = path.join(
        feat_dir, args.language.lower() + ".train.npz"
        )
    output_npz_fn = path.join(
        feat_dir, args.language.lower() + ".train.utd_terms.npz"
        )
    if not path.isfile(output_npz_fn):
        print("Extracting MFCCs for UTD word tokens")
        utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
    else:
        print("Using existing file:", output_npz_fn)


    # UTD SEGMENTS THAT HAVE BEEN PARTIALLY FIXED

    # Write list with fixed labels
    fixed_labels_list_fn = path.join(
        "lists", args.language, "train.utd_terms.fixed_labels.list"
        )
    if not path.isfile(fixed_labels_list_fn):

        # Read UTD terms
        utd_list_fn = path.join("lists", args.language, "train.utd_terms.list")
        print("Reading:", utd_list_fn)
        # overlap_dict[speaker_utt][(start, end)] is a tuple of
        # (label, (start, end), overlap)
        overlap_dict = {}
        with codecs.open(utd_list_fn, "r", "utf-8") as utd_list_f:
            for line in utd_list_f:
                term, speaker, utt, start_end = line.strip().split("_")
                start, end = start_end.split("-")
                start = int(start)
                end = int(end)
                if not speaker + "_" + utt in overlap_dict:
                    overlap_dict[speaker + "_" + utt] = {}
                overlap_dict[speaker + "_" + utt][(start, end)] = (
                    "label", (0, 0), 0
                    )

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
        print("Getting ground truth terms with maximal overlap:")
        for utt_key in tqdm(fa_dict):
            # print(utt_key)
            if utt_key not in overlap_dict:
                continue
            for (fa_start, fa_end) in fa_dict[utt_key]:
                for (utd_start, utd_end) in overlap_dict[utt_key]:
                    overlap = get_overlap(
                        utd_start, utd_end, fa_start, fa_end
                        )
                    if overlap == 0:
                        continue
                    if (overlap > overlap_dict[utt_key][(utd_start,
                            utd_end)][2]):
                        overlap_dict[utt_key][(utd_start, utd_end)] = (
                            fa_dict[utt_key][(fa_start, fa_end)],
                            (fa_start, fa_end), overlap
                            )

        # Write list
        print("Writing:", fixed_labels_list_fn)
        with codecs.open(fixed_labels_list_fn, "w", "utf-8") as list_f:
            for utt_key in overlap_dict:
                for (utd_start, utd_end) in overlap_dict[utt_key]:
                    label = overlap_dict[utt_key][(utd_start, utd_end)][0]
                    list_f.write(
                        "{}_{}_{:06d}-{:06d}\n".format(label, utt_key,
                        utd_start, utd_end)
                        )
    else:
        print("Using existing file:", fixed_labels_list_fn)

    # Extract partially fixed UTD segments
    input_npz_fn = path.join(
        feat_dir, args.language.lower() + ".train.npz"
        )
    output_npz_fn = path.join(
        feat_dir, args.language.lower() + ".train.utd_terms.fixed_labels.npz"
        )
    if not path.isfile(output_npz_fn):
        print("Extracting MFCCs for partially fixed UTD word tokens")
        utils.segments_from_npz(
            input_npz_fn, fixed_labels_list_fn, output_npz_fn
            )
    else:
        print("Using existing file:", output_npz_fn)


if __name__ == "__main__":
    main()
