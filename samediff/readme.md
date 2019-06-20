Same-Different Evaluation
=========================

Overview
--------
Performs same-different evaluation on frame-level features using dynamic time
warping (DTW) alignment.


Evaluation
----------
This needs to be run on a multi-core machine. Change the `n_cpus` variable in
`run_calcdists.sh` and `run_samediff.sh` to the number of CPUs on the machine.

As an example, to evaluate the Spanish development MFCCs:

    ./run_calcdists.sh ../features/mfcc/SP/sp.dev.gt_words.npz  # finish first
    ./run_samediff.sh ../features/mfcc/SP/sp.dev.gt_words.npz


Results
-------
SWDP average precision:

- Spanish dev: 0.19288643880022374
- Spanish eval: 0.29650854702289764

*To-do*
