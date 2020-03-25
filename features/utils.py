"""
Utility functions.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from tqdm import tqdm
import codecs
import numpy as np
import subprocess


shell = lambda command: subprocess.Popen(
    command, shell=True, stdout=subprocess.PIPE
    ).communicate()[0]


def filter_words(fa_fn, output_fn, min_frames=50, min_chars=5):
    """
    Find words of at least `min_frames` frames and `min_chars` characters.

    Ground truth words are extracted from the forced alignment file `fa_fn` and
    written to the word list file `output_fn`.
    """
    print("Reading:", fa_fn)
    print("Writing:", output_fn)
    n_tokens = 0
    with codecs.open(fa_fn, "r", "utf-8") as fa_f:
        with codecs.open(output_fn, "w", "utf-8") as output_f:
            for line in fa_f:
                utt_key, _, start, duration, label = line.strip().split()
                start = float(start)
                duration = float(duration)
                end = start + duration
                start_frame = int(round(start*100))
                end_frame = int(round(end*100))
                if (end_frame - start_frame >= min_frames and len(label) >=
                        min_chars and label != "<unk>" and label != "sil"
                        and label != "?" and label != "spn"):
                    output_f.write(
                        "{}_{}_{:06d}-{:06d}\n".format(label, utt_key,
                        start_frame, end_frame + 1)
                        )
                    n_tokens += 1
    print("No. tokens:", n_tokens)


def segments_from_npz(input_npz_fn, segments_fn, output_npz_fn):
    """
    Cut segments from a NumPy archive and save in a new archive.

    As keys, the archives use the format "label_spkr_utterance_start-end".
    """

    # Read the .npz file
    print("Reading npz:", input_npz_fn)
    input_npz = np.load(input_npz_fn)

    # Create input npz segments dict
    utterance_segs = {}  # utterance_segs["s08_02b_029657-029952"]
                         # is (29657, 29952)
    for key in input_npz.keys():
        s = key.split("_")
        if len(s) == 3:
            # Format: s08_02b_029657-029952
            utterance_segs[key] = tuple([int(i) for i in s[-1].split("-")])
        elif len(s) == 2:
            # Format: s08_02b
            utterance_segs[key] = (0, input_npz[key].shape[0])

    # Create target segments dict
    print("Reading segments:", segments_fn)
    target_segs = {}  # target_segs["years_s01_01a_004951-005017"]
                      # is ("s01_01a", 4951, 5017)
    for line in open(segments_fn):
        line_split = line.split("_")
        utterance = line_split[-3] + "_" + line_split[-2]
        start, end = line_split[-1].split("-")
        start = int(start)
        end = int(end)
        target_segs[line.strip()] = (utterance, start, end)

    print("Extracting segments:")
    output_npz = {}
    n_target_segs = 0
    for target_seg_key in tqdm(sorted(target_segs)):
        utterance, target_start, target_end = target_segs[target_seg_key]
        for utterance_key in [
                i for i in utterance_segs.keys() if (i +
                "_").startswith(utterance + "_")]:
                # If like below: "GE008_128" also matches "GE008_12"
                # i for i in utterance_segs.keys() if i.startswith(utterance)]:
            utterance_start, utterance_end = utterance_segs[utterance_key]
            if (target_start >= utterance_start and target_start <
                    utterance_end):
                start = target_start - utterance_start
                end = target_end - utterance_start
                output_npz[target_seg_key] = input_npz[
                    utterance_key
                    ][start:end]
                n_target_segs += 1

                break

    print(
        "Extracted " + str(n_target_segs) + " out of " + str(len(target_segs))
        + " segments"
        )
    print("Writing:", output_npz_fn)
    np.savez(output_npz_fn, **output_npz)


def terms_from_pairs(pairs_fn, output_list_fn):

    print("Reading:", pairs_fn)
    terms = set()
    with open(pairs_fn) as f:
        for line in f:
            line = line.replace("###", " ")
            (cluster, utt1, start1, end1, cluster2, utt2, start2, end2) = (
                line.strip().split(" ")
                )
            start1 = int(start1)
            end1 = int(end1)
            start2 = int(start2)
            end2 = int(end2)
            terms.add((cluster, utt1, start1, end1))
            terms.add((cluster, utt2, start2, end2))

    print("Writing:", output_list_fn)
    with open(output_list_fn, "w") as f:
        for cluster, utt, start, end in terms:
            f.write(
                cluster + "_" + utt + "_" + "%06d" % start + "-" + "%06d" % end
                + "\n"
                )


def format_enno_pairs(enno_pairs_fn, output_pairs_fn):
    print("Reading:", enno_pairs_fn)
    print("Writing:", output_pairs_fn)
    with codecs.open(enno_pairs_fn, "r", "utf-8") as enno_f:
        with codecs.open(output_pairs_fn, "w", "utf-8") as output_f:
            for line in enno_f:
                line = line.replace("###", " ")
                (cluster1, utt1, start1, end1, cluster2, utt2, start2, end2) = (
                    line.strip().split(" ")
                    )
                start1 = int(start1)
                end1 = int(end1)
                start2 = int(start2)
                end2 = int(end2)
                output_f.write(
                    "{}_{}_{:06d}-{:06d} "
                    "{}_{}_{:06d}-{:06d}\n".format(cluster1, utt1, start1,
                    end1, cluster2, utt2, start2, end2)
                    )
