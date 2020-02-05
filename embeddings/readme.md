Acoustic Word Embedding Models and Evaluation
=============================================

Overview
--------
The examples below are intended for illustration purposes -- there are many
different language-combinations and other settings which can be adjusted for
the different models. But the most important command-line arguments are
illustrated in the examples here.


Data preparation
----------------
Create links to the MFCC NumPy archives:

    ./link_mfcc.py SP

You need to run `link_mfcc.py` for all languages; run it without any arguments
to see all 16 language codes. Alternatively, links can be greated for all
languages by passing the "all" argument.


Autoencoder RNN
---------------
Train an AE-RNN on Spanish UTD segments:

    ./train_cae_rnn.py --extrinsic_usefinal --ae_n_val_interval 9 \
        --ae_n_epochs 10 --cae_n_epochs 0 --train_tag utd --val_lang SP SP

Train an AE-RNN on seven languages using ground truth segments and validate on
German:

    ./train_cae_rnn.py --ae_n_epochs 25 --cae_n_epochs 0 \
        --n_max_types 1000 --train_tag gt --val_lang GE RU+CZ+FR+PL+TH+PO


Correspondence autoencoder RNN
------------------------------
Train a CAE-RNN on Spanish UTD segments:

    ./train_cae_rnn.py --pretrain_usefinal --extrinsic_usefinal \
        --ae_n_val_interval 14 --ae_n_epochs 15 --cae_n_epochs 3 \
        --cae_batch_size 600 --train_tag utd --val_lang SP SP

Evaluate the model:

    ./apply_model.py \
        models/SP.utd/train_cae_rnn/17b498a959/cae.best_val.ckpt SP val
    ./eval_samediff.py --mvn \
        models/SP.utd/train_cae_rnn/17b498a959/cae.best_val.SP.val.npz

Analyse embeddings:

    ./analyse_embeds.py --normalize --word_type \
        guatemala,presidente,autoridades,candidatos,vicepresidente,social \
        models/SP.utd/train_cae_rnn/17b498a959/cae.best_val.SP.val.npz

All the models trained below can be applied, evaluated and analysed using the
scripts above.

Train a CNN-RNN on Spanish ground truth segments:

    ./train_cae_rnn.py --pretrain_usefinal --n_max_pairs 100000 \
        --ae_n_val_interval 14 --ae_n_epochs 15 --cae_n_epochs 25 \
        --train_tag gt --val_lang SP SP

Train a CAE-RNN jointly on multiple languages, limiting the maximum overall
number of pairs, the maximum number of types per language and requiring a
minimum number of tokens per type:

    ./train_cae_rnn.py --pretrain_usefinal --ae_n_val_interval 14 \
        --ae_n_epochs 15 --cae_n_epochs 10 --n_max_pairs 300000 \
        --n_min_tokens_per_type 2 --n_max_types 1000 --train_tag gt \
        --val_lang GE RU+CZ+FR+PL+TH+PO


Siamese RNN
-----------
Train a Siamese RNN on ground truth segments:

    ./train_siamese_rnn.py --n_epochs 25 --train_tag gt --val_lang SP SP

Train a Siamese RNN ensuring that each batch contains paired data, i.e., no
batch will have a singleton token:

    ./train_siamese_rnn_pairbatch.py --n_epochs 15 --train_tag gt \
        --margin 0.2 --val_lang GE GE


Siamese CNN
-----------
Train a Siamese CNN on ground truth segments:

    ./train_siamese_cnn.py --n_epochs 150 --train_tag gt --n_val_interval 5 SP


Classifier CNN
--------------
Train a word classifier CNN on ground truth segments:

    ./train_cnn.py --n_epochs 100 --train_tag gt --n_val_interval 5 SP


Classifier RNN
--------------
Train a word classifier RNN on ground truth segments:

    ./train_rnn.py --n_epochs 25 --train_tag gt --val_lang SP SP

Train a word classifier RNN jointly on multiple languages:

    ./train_rnn.py --n_epochs 15 --train_tag gt --n_max_types 10000 \
        --n_max_tokens_per_type 20 --val_lang GE RU+CZ+FR+PL+TH+BG+PO

