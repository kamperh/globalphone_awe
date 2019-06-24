Acoustic Word Embedding Models and Evaluation
=============================================

Data preparation
----------------
Create links to the MFCC NumPy archives:

    ./link_mfcc.py SP

You need to run `link_mfcc.py` for all languages; run it without any arguments
to see all 16 language codes. Alternatively, links can be greated for all
languages by giving the "all" argument.


Correspondence autoencoder RNN
------------------------------
Train, validate and test a CAE-RNN on Spanish ground truth segments:

    ./train_cae_rnn.py --ae_n_epochs 10 --cae_n_epochs 3 --val_lang SP SP

Evaluate the model:

    ./apply_model.py \
        models/SP.utd/train_cae_rnn/17b498a959/cae.best_val.ckpt SP val
    ./eval_samediff.py --mvn \
        models/SP.utd/train_cae_rnn/17b498a959/cae.best_val.SP.val.npz

All the models trained below can be evaluated using these scripts.

Analyse embeddings:

    ../embeddings/analyse_embeds.py --normalize --word_type \
        guatemala,presidente,autoridades,candidatos,asesinato,presupuesto,vicepresidente,negociaciones,netanyahu,social,explotaciones \
        models/SP.utd/train_cae_rnn/17b498a959/cae.best_val.SP.val.npz


Siamese RNN
-----------
Train a Siamese RNN on ground truth segments:

    ./train_siamese_rnn.py --n_epochs 50 --train_tag gt --val_lang SP SP


Siamese CNN
-----------
Train a Siamese CNN on ground truth segments:

    ./train_siamese_cnn.py --n_epochs 150 --train_tag gt --n_val_interval 5 SP


Classifier CNN
--------------
Train a word classifier CNN on ground truth segments:

    ./train_cnn.py --n_epochs 150 --train_tag gt --n_val_interval 5 SP


