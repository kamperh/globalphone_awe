Black-Box Analysis of Embedding Models
======================================

Extract features for analysis
-----------------------------
Extract features and perform intermediate analysis:

    ./extract_analysis_features.py --analyse RU


Process features with model
---------------------------
The extracted features would typically be applied through a model.

For instance, to obtain downsampled embeddings, run:

    cd ../downsample
    ./downsample.py --technique resample --frame_dims 13 \
        ../blackbox/mfcc/RU/ru.dev.filter1_phone.npz \
        exp/RU/mfcc.dev.filter1_phone.downsample_10.npz 10
    ../embeddings/eval_samediff.py --mvn \
        exp/RU/mfcc.dev.filter1_phone.downsample_10.npz
    cd -

To obtain embeddings from a particular mode, run:

    cd ../embeddings
    ./apply_model_to_npz.py \
        models/RU.gt/train_cae_rnn/0bab597d5a/cae.best_val.ckpt \
        ../blackbox/mfcc/RU/ru.dev.filter1_phone.npz
    ./eval_samediff.py --mvn \
        models/RU.gt/train_cae_rnn/0bab597d5a/cae.best_val.ru.dev.filter1_phone.npz
    cd -


t-SNE visualisation
-------------------
To visualise embeddings, https://projector.tensorflow.org/ can be used. To
generate the input required by this tool, run:

    ./npz_to_tsv.py \
        ../downsample/exp/RU/mfcc.dev.filter2_phone.downsample_10.npz auto

and load the data into the tool.


Agglomerative clustering
------------------------
Clustering can be applied and visualised by running:

    ./hierarchical_clustering.py --n_samples 1000 \
        ../embeddings/models/RU.gt/train_cae_rnn/0bab597d5a/cae.best_val.ru.dev.filter2_phone.npz
    






./extract_analysis_features.py --analyse GE
cd ../downsample
./downsample.py --technique resample --frame_dims 13 ../blackbox/mfcc/GE/ge.dev.filter1_gt.npz exp/GE/mfcc.dev.filter1_gt.downsample_10.npz  10
cd -
./npz_to_tsv.py ../downsample/exp/GE/mfcc.dev.filter1_gt.downsample_10.npz auto
cd ../embeddings
./apply_model_to_npz.py models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ckpt ../blackbox/mfcc/GE/ge.dev.filter1_gt.npz
cd -
npz_to_tsv.py ./npz_to_tsv.py ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ge.dev.filter1_gt.npz auto

./extract_analysis_features.py RU
cd ../downsample
./downsample.py --technique resample --frame_dims 13 ../blackbox/mfcc/RU/ru.dev.filter1_phone.npz exp/RU/mfcc.dev.filter1_phone.downsample_10.npz 10
../embeddings/eval_samediff.py --mvn exp/RU/mfcc.dev.filter1_phone.downsample_10.npz
cd ../embeddings
./apply_model_to_npz.py models/RU.gt/train_cae_rnn/0bab597d5a/cae.best_val.ckpt ../blackbox/mfcc/RU/ru.dev.filter1_phone.npz
./eval_samediff.py --mvn models/RU.gt/train_cae_rnn/0bab597d5a/cae.best_val.ru.dev.filter1_phone.npz
cd ../blackbox
./npz_to_tsv.py ../downsample/exp/RU/mfcc.dev.filter1_phone.downsample_10.npz auto
./npz_to_tsv.py ../embeddings/models/RU.gt/train_cae_rnn/0bab597d5a/cae.best_val.ru.dev.filter1_phone.npz auto

./extract_analysis_features.py RU
cd ../downsample
./downsample.py --technique resample --frame_dims 13 ../blackbox/mfcc/RU/ru.dev.filter2_phone.npz exp/RU/mfcc.dev.filter2_phone.downsample_10.npz 10
../embeddings/eval_samediff.py --mvn exp/RU/mfcc.dev.filter2_phone.downsample_10.npz
cd ../embeddings
./apply_model_to_npz.py models/RU.gt/train_cae_rnn/0bab597d5a/cae.best_val.ckpt ../blackbox/mfcc/RU/ru.dev.filter2_phone.npz
./eval_samediff.py --mvn models/RU.gt/train_cae_rnn/0bab597d5a/cae.best_val.ru.dev.filter2_phone.npz
cd ../blackbox
./npz_to_tsv.py ../downsample/exp/RU/mfcc.dev.filter2_phone.downsample_10.npz auto
./npz_to_tsv.py ../embeddings/models/RU.gt/train_cae_rnn/0bab597d5a/cae.best_val.ru.dev.filter2_phone.npz auto

./hierarchical_clustering.py --n_samples 1000 ../embeddings/models/RU.gt/train_cae_rnn/0bab597d5a/cae.best_val.ru.dev.filter2_phone.npz


./analyse_pairs.py --pronunciation GE ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ge.dev.filter1_gt.npz
./analyse_pairs.py --pronunciation GE ../downsample/exp/GE/mfcc.dev.filter1_gt.downsample_10.npz

./hierarchical_clustering.py --n_samples 1000 ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ge.dev.filter1_gt.npz
./hierarchical_clustering.py --n_samples 1000 ../downsample/exp/GE/mfcc.dev.filter1_gt.downsample_10.npz

./hierarchical_clustering.py --n_samples 1000 ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.GE.val.npz
./hierarchical_clustering.py --n_samples 1000 ../downsample/exp/GE/mfcc.dev.gt_words.downsample_10.npz

./logreg_speaker.py ../downsample/exp/GE/mfcc.dev.gt_words.downsample_10.npz
./logreg_speaker.py ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.GE.val.npz
./logreg_speaker.py ../downsample/exp/GE/mfcc.dev.filter1_gt.downsample_10.npz
./logreg_speaker.py ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ge.dev.filter1_gt.npz


./logreg_pronlength.py ../downsample/exp/GE/mfcc.dev.filter1_gt.downsample_10.npz GE
./logreg_pronlength.py ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ge.dev.filter1_gt.npz GE


./extract_analysis_features.py --analyse GE

./npz_to_tsv.py ../downsample/exp/RU/mfcc.dev.gt_words.downsample_10.npz auto
./npz_to_tsv.py ../downsample/exp/GE/mfcc.dev.gt_words.downsample_10.npz auto
./npz_to_tsv.py ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.GE.val.npz auto
./npz_to_tsv.py ../embeddings/models/RU.gt/train_cae_rnn/0bab597d5a/cae.best_val.RU.val.npz auto


./analyse_pairs.py ../embeddings/models/GE.gt/train_cae_rnn/15b3ecce63/cae.best_val.ge.dev.filter1_gt.npz

