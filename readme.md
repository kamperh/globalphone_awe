Multilingual Acoustic Word Embeddings on GlobalPhone
====================================================

Overview
--------
Multilingual acoustic word embedding approaches are implemented and evaluated
on the GlobalPhone corpus. The experiments are described in:

- H. Kamper, Y. Matusevych, and S.J. Goldwater "Multilingual acoustic word
  embedding models for processing zero-resource languages," in *Proc. ICASSP*,
  2020. [[arXiv](add_link)]

Please cite this paper if you use the code.


Disclaimer
----------
The code provided here is not pretty. But I believe that research should be
reproducible. I provide no guarantees with the code, but please let me know if
you have any problems, find bugs or have general comments.


Download datasets
-----------------
The [GlobalPhone](https://csl.anthropomatik.kit.edu/english/globalphone.php)
corpus and forced alignments of the data needs to be obtained. GlobalPhone
needs to be paid for. If you have proof of payment, we can give you access to
the forced alignments. Save the data and forced alignments in a separate
directory and update the `paths.py` file to point to the data directories.


Install dependencies
--------------------
You will require the following:

- [Python 3](https://www.python.org/downloads/)
- [TensorFlow 1.13.1](https://www.tensorflow.org/)
- [LibROSA](http://librosa.github.io/librosa/)
- [Cython](https://cython.org/)
- [tqdm](https://tqdm.github.io/)
- [speech_dtw](https://github.com/kamperh/speech_dtw/)
- [shorten](http://etree.org/shnutils/shorten/dist/src/shorten-3.6.1.tar.gz)

To install `speech_dtw` (required for same-different evaluation) and `shorten`
(required for processing audio), run `./install_local.sh`.

You can install all the other dependencies in a conda environment by running:

    conda env create -f environment.yml
    conda activate tf1.13


Extract speech features
-----------------------
Update the paths in `paths.py` to point to the data directories. Extract MFCC
features in the `features/` directory as follows:

    cd features
    ./extract_features.py SP

You need to run `extract_features.py` for all languages; run it without any
arguments to see all 16 language codes.

UTD pairs can also be analysed here, by running e.g.:

    ./analyse_utd_pairs.py SP


Evaluate frame-level features using the same-different task
-----------------------------------------------------------
This is optional. To perform frame-level same-different evaluation based on
dynamic time warping (DTW), follow [samediff/readme.md](samediff/readme.md).


Obtain downsampled acoustic word embeddings
-------------------------------------------
Extract and evaluate downsampled acoustic word embeddings by running the steps
in [downsample/readme.md](downsample/readme.md).


Train neural acoustic word embeddings
-------------------------------------
Train and evaluate neural network acoustic word embedding models by running the
steps in [embeddings/readme.md](embeddings/readme.md).


Analyse embedding models
------------------------
Analyse different properties/aspects of the acoustic word embedding models by
running the steps in [blackbox/readme.md](blackbox/readme.md).


Query-by-example search
-----------------------
Perform query-by-example search experiments by running the steps in
[qbe/readme.md](qbe/readme.md).


Unit tests
----------
In the root project directory, run `make test` to run unit tests.


References
----------
- https://github.com/eginhard/cae-utd-utils


Contributors
------------
- [Herman Kamper](http://www.kamperh.com/)
- [Yevgen Matusevych](https://homepages.inf.ed.ac.uk/ymatusev/)
- [Sharon Goldwater](https://homepages.inf.ed.ac.uk/sgwater/)


License
-------
The code is distributed under the Creative Commons Attribution-ShareAlike
license ([CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)).
