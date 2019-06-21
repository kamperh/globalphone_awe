Acoustic Word Embedding Models and Evaluation
=============================================

Data preparation
----------------
Create links to the MFCC NumPy archives:

    ./link_mfcc.py SP

You need to run `link_mfcc.py` for all languages; run it without any arguments
to see all 16 language codes. Alternatively, links can be greated for all
languages by giving the "all" argument.


Correspondence autoencoder
--------------------------
Train, validate and test a CAE-RNN on Spanish ground truth segments:

    ./train_cae_rnn.py --ae_n_epochs 10 --cae_n_epochs 3 --val_lang SP SP

Evaluate the model:

    ./apply_model \
        models/SP.utd/train_cae_rnn/17b498a959/cae.best_val.ckpt SP val
