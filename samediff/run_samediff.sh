#!/bin/bash

# Calculate distances for same-different evaluation of autoencoder features.
# Herman Kamper, h.kamper@sms.ed.ac.uk, 2014-2015, 2018.

# General setup
n_cpus=3

# Input features
features_npz=$1  # features_npz=../data/mfcc_test.npz
if [ -z $features_npz ]; then
    echo "usage: ${0} features_npz"
    exit 1
fi
if [ ! -f $features_npz ]; then
    echo "Error: $features_npz does not exist"
    exit 1
fi

# Files and directories
basename=`basename $features_npz`
basename="${basename%.*}"
samediff_dir=exp/$basename
pairs=$samediff_dir/pairs.list
pairs_split_dir=$samediff_dir/pairs_split
labels=$samediff_dir/labels.list
speakers=$samediff_dir/speakers.list
distances_split_dir=$samediff_dir/distances_split
distances=$samediff_dir/distances.dist
samediff_result=$samediff_dir/samediff_result.txt

# Make sure that all the jobs are done
complete=`ls $distances_split_dir/distances.*.log | xargs grep "End time" \
    | wc -l`
echo "Number of splits completed: $complete out of $n_cpus"
if [ "$n_cpus" -ne "$complete" ]; then 
    echo "Error: wait for jobs to complete"
    exit 1
fi

# Concatenate distances
if [ ! -f $distances ]; then
    touch $distances
    for JOB in $(seq 1 $n_cpus); do
        cat $distances_split_dir/distances.$JOB.dist >> $distances
    done
fi

if [ ! -f $samediff_result ]; then
    python ../../src/speech_dtw/utils/samediff.py --binary_dists $labels \
        --speakers_fn $speakers $distances > $samediff_result
    echo
    cat $samediff_result
    echo

    if [ $? -ne 0 ]; then
        echo "Exiting"
        rm $samediff_result
        exit 1
    fi
fi

# Clean directories
read -p "Clean distances (y/n)? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -r $pairs $pairs_split_dir $distances $distances_split_dir
fi
