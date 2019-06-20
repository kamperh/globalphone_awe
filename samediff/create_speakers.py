#!/usr/bin/env python

"""
Create a list of the speaker labels from a list of utterance IDs.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014, 2018, 2019
"""

import argparse
import codecs
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("utterance_ids_fn")
    parser.add_argument("labels_fn")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    
    args = check_argv()

    utt_ids = [i.strip() for i in open(args.utterance_ids_fn)]
    labels = []
    for utt_id in utt_ids:
        speaker = utt_id.split("_")[1] #"_".join(utt_id.split("_")[:-2])
        labels.append(speaker)
    with codecs.open(args.labels_fn, "w", "utf-8") as f:
        for label in labels:
            f.write(label + "\n")


if __name__ == "__main__":
    main()
