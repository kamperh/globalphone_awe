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

    ./run_calcdists.sh ../features/mfcc/KO/ko.dev.gt_words.npz  # finish first
    ./run_samediff.sh ../features/mfcc/KO/ko.dev.gt_words.npz


Results
-------
SWDP average precision:

- CH dev: 0.15380600
- CR dev: 0.13270483
- HA dev: 0.21368697
- SP dev: 0.19288643
- SW dev: 0.10928384
- TU dev: 0.18624635

- GE dev: 0.22482616
- KO dev: 0.15748395

- SP eval: 0.29650854
