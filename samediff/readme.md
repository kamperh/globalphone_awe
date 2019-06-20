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

    ./run_calcdists.sh ../features/mfcc/SP/sp.eval.gt_words.npz
    ./run_samediff.sh ../features/mfcc/SP/sp.eval.gt_words.npz


Results
-------
Spanish dev MFFCs:

    Average precision: ??
    Precision-recall breakeven: ??

*To-do*
