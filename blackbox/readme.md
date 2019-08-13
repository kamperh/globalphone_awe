Black-Box Analysis of Embedding Models
======================================

Extract features for analysis
-----------------------------
While the default evaluation data (typically including a `gt_words` tag) are
extracted with a minimum duration of 0.5 seconds at at least 5 characters, it
is useful to do analysis on a larger range of word segments. This is done in
the script below.

Extract features and perform intermediate analysis:

    ./extract_analysis_features.py --analyse RU


Process features with model
---------------------------
The extracted features would typically be applied through a model.

For instance, to obtain downsampled embeddings, run:

    cd ../downsample
    ./downsample.py --technique resample --frame_dims 13 \
        ../blackbox/mfcc/GE/ge.dev.filter1_gt.npz \
        exp/GE/mfcc.dev.filter1_gt.downsample_10.npz 10
    ../embeddings/eval_samediff.py --mvn \
        exp/GE/mfcc.dev.filter1_gt.downsample_10.npz
    cd -

To obtain embeddings from a particular mode, run:

    cd ../embeddings
    ./apply_model_to_npz.py \
        models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ckpt \
        ../blackbox/mfcc/GE/ge.dev.filter1_gt.npz
    ./eval_samediff.py --mvn \
        models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ge.dev.filter1_gt.npz
    cd -


t-SNE visualisation
-------------------
To visualise embeddings, https://projector.tensorflow.org/ can be used. To
generate the input required by this tool, run:

    ./npz_to_tsv.py \
        ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ge.dev.filter1_gt.npz

and load the data into the tool.


Agglomerative clustering
------------------------
Clustering can be applied and visualised by running:

    ./hierarchical_clustering.py --n_samples 1000 \
        ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ge.dev.filter1_gt.npz

Here, the colouring in the labels indicate the speaker for that token.


Classifier analysis
-------------------
Perform speaker classification by training a multi-class logistic regression
classifier on 80% of the data and then test on the remaining 20%:

    ./logreg_speaker.py \
        ../downsample/exp/GE/mfcc.dev.gt_words.downsample_10.npz
    ./logreg_speaker.py \
        ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.GE.val.npz

Perform length (number of phones) classification:

    # To-do: should train and test on different sets here
    ./logreg_pronlength.py \
        ../downsample/exp/GE/mfcc.dev.filter1_gt.downsample_10.npz GE
    ./logreg_pronlength.py \
        ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ge.dev.filter1_gt.npz GE
