#!/bin/bash

# Calculate distances for same-different evaluation.
# Herman Kamper, kamperh@gmail.com, 2014-2015, 2018-2019.

set -e

# General setup
n_cpus=3
cmd="python run_local.py"
# cmd="./local/run_sge.py --extraargs -P inf_hcrc_cstr_students"
export PYTHONUNBUFFERED="YOUR_SET"  # flush after every Python print statement


# Input features
features_npz=$1
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
utterance_ids=$samediff_dir/utterance_ids.list
pairs=$samediff_dir/pairs.list
pairs_split_dir=$samediff_dir/pairs_split
labels=$samediff_dir/labels.list
speakers=$samediff_dir/speakers.list
distances_split_dir=$samediff_dir/distances_split
distances=$samediff_dir/distances.dist

# Create samediff dir
[ ! -d $samediff_dir ] && mkdir -p $samediff_dir

# Create utterance IDs and label files
[ ! -f $utterance_ids ] && python get_npz_keys.py $features_npz $utterance_ids
[ ! -f $labels ] && python create_labels.py $utterance_ids $labels
[ ! -f $speakers ] && python create_speakers.py $utterance_ids $speakers

# Generate a list of all possible pairs and split for parallel processing
[ ! -f $pairs ] && python ../../src/speech_dtw/utils/create_pair_file.py \
    $utterance_ids $pairs
[ ! -d $pairs_split_dir ] && ../../src/speech_dtw/utils/split_file.py \
    $pairs $n_cpus $pairs_split_dir

# Calculate DTW distances
if [ ! -d $distances_split_dir ]; then
    mkdir -p $distances_split_dir
    dist_cmd="python ../../src/speech_dtw/utils/calculate_dtw_costs.py \
        --binary_dists --input_fmt npz $pairs_split_dir/pairs.JOB.list \
        $features_npz $distances_split_dir/distances.JOB.dist"
    $cmd 1 $n_cpus $distances_split_dir/distances.JOB.log "$dist_cmd"
fi

echo "Wait to complete, then run run_samediff.sh"
