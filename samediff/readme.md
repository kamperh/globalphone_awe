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

    ./run_calcdists.sh ../features/mfcc/SP/sp.dev.gt_words.npz
    ./run_samediff.sh ../features/mfcc/SP/sp.dev.gt_words.npz


Results
-------
Spanish eval MFFCs:

    Average precision: 0.430521533704422
    Precision-recall breakeven: 0.48065739075532504
    SWDP average precision: 0.29650854702289764
    SWDP precision-recall breakeven: 0.38123891442546787

*To-do*
