#!/usr/bin/env python

"""
Downsample a given file using a particular technique and target dimensionality.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2018, 2019
"""

import argparse
import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal
import sys

flatten_order = "C"


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("input_npz_fn", type=str, help="input speech file")
    parser.add_argument(
        "output_npz_fn", type=str, help="output embeddings file"
        )
    parser.add_argument("n", type=int, help="number of samples")
    parser.add_argument(
        "--technique", choices=["interpolate", "resample", "rasanen"],
        default="resample"
        )
    parser.add_argument(
        "--frame_dims", type=int, default=None,
        help="only keep these number of dimensions"
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
    
    print("Reading:", args.input_npz_fn)
    input_npz = np.load(args.input_npz_fn)
    d_frame = input_npz[sorted(input_npz.keys())[0]].shape[1]

    print("Frame dimensionality:", d_frame)
    if args.frame_dims is not None and args.frame_dims < d_frame:
        d_frame = args.frame_dims
        print("Reducing frame dimensionality:", d_frame)

    print("Downsampling:", args.technique)
    output_npz = {}
    for key in input_npz:

        # Limit input dimensionailty
        y = input_npz[key][:, :args.frame_dims].T

        # Downsample
        if args.technique == "interpolate":
            x = np.arange(y.shape[1])
            f = interpolate.interp1d(x, y, kind="linear")
            x_new = np.linspace(0, y.shape[1] - 1, args.n)
            y_new = f(x_new).flatten(flatten_order) #.flatten("F")
        elif args.technique == "resample":
            y_new = signal.resample(
                y, args.n, axis=1
                ).flatten(flatten_order) #.flatten("F")
        elif args.technique == "rasanen":
            # Taken from Rasenen et al., Interspeech, 2015
            n_frames_in_multiple = int(np.floor(y.shape[1] / args.n)) * args.n
            y_new = np.mean(
                y[:, :n_frames_in_multiple].reshape((d_frame, args.n, -1)),
                axis=-1
                ).flatten(flatten_order) #.flatten("F")

        # This was done in Rasenen et al., 2015, but didn't help here
        # last_term = args.n/3. * np.log10(y.shape[1] * 10e-3)
        # Not sure if the above should be in frames or ms
        # y_new = np.hstack([y_new, last_term])
        
        # Save result
        output_npz[key] = y_new

    print(
        "Output dimensionality:",
        output_npz[sorted(output_npz.keys())[0]].shape[0]
        )

    print("Writing:", args.output_npz_fn)
    np.savez_compressed(args.output_npz_fn, **output_npz)


if __name__ == "__main__":
    main()
