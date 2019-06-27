Acoustic Word Embedding Models and Evaluation
=============================================

Data preparation
----------------
Create links to the MFCC NumPy archives:

    ./link_mfcc.py SP

You need to run `link_mfcc.py` for all languages; run it without any arguments
to see all 16 language codes. Alternatively, links can be greated for all
languages by giving the "all" argument.


Autoencoder RNN
---------------
Train an AE-RNN on Spanish UTD segments:

    ./train_cae_rnn.py --pretrain_usefinal --extrinsic_usefinal \
        --ae_n_val_interval 9 --ae_n_epochs 10 --cae_n_epochs 0 \
        --train_tag utd --val_lang SP SP


Correspondence autoencoder RNN
------------------------------
Train a CAE-RNN on Spanish UTD segments:

    ./train_cae_rnn.py --pretrain_usefinal --extrinsic_usefinal \
        --ae_n_val_interval 14 --ae_n_epochs 15 --cae_n_epochs 3 \
        --cae_batch_size 600 --train_tag utd --val_lang SP SP

Train a CNN-RNN on Spanish ground truth segments:

    ./train_cae_rnn.py --pretrain_usefinal \
        --ae_n_val_interval 14 --ae_n_epochs 15 --cae_n_epochs 15 \
        --n_max_pairs 100000 --train_tag gt --val_lang SP SP

Evaluate the model:

    ./apply_model.py \
        models/SP.utd/train_cae_rnn/17b498a959/cae.best_val.ckpt SP val
    ./eval_samediff.py --mvn \
        models/SP.utd/train_cae_rnn/17b498a959/cae.best_val.SP.val.npz

All the models trained below can be evaluated using these scripts.

Analyse embeddings:

    ../embeddings/analyse_embeds.py --normalize --word_type \
        guatemala,presidente,autoridades,candidatos,vicepresidente,social \
        models/SP.utd/train_cae_rnn/17b498a959/cae.best_val.SP.val.npz

Train a CAE-RNN jointly on multiple languages:

    ./train_cae_rnn.py --n_max_pairs 100000 --ae_n_epochs 10 --cae_n_epochs 25 --train_tag gt --val_lang SP RU+CZ


Siamese RNN
-----------
Train a Siamese RNN on ground truth segments:

    ./train_siamese_rnn.py --n_epochs 25 --train_tag gt --val_lang SP SP


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
Train a word classifier RNN on ground truth segmens:

    ./train_rnn.py --n_epochs 25 --train_tag gt --val_lang SP SP


Current hyperparameters
-----------------------
CAE-RNN trained on UTD:

    ./train_cae_rnn.py --pretrain_usefinal --ae_n_epochs 5 --cae_n_epochs 25 \
        --train_tag utd --val_lang SP SP

CAE-RNN trained on GT:

    ./train_cae_rnn.py --pretrain_usefinal --ae_n_epochs 5 --cae_n_epochs 25 \
        --n_max_pairs 100000 --train_tag gt --val_lang SP RU
