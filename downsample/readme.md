Downsampled Acoustic Word Embeddings
====================================

Overview
--------
MFCCs are downsampled to obtain acoustic word embeddings. These are evaluated
using same-different evaluation.


Downsampling
------------
Perform downsampling on MFCCs without deltas:

    mkdir -p exp/SP
    ./downsample.py --technique resample --frame_dims 13 \
        ../features/mfcc/SP/sp.dev.gt_words.npz \
        exp/SP/mfcc.dev.gt_words.downsample_10.npz \
        10


Evaluation
----------
Evaluate and analyse downsampled MFCCs without deltas:

    ./eval_samediff.py --mvn exp/SP/mfcc.dev.gt_words.downsample_10.npz
    ./analyse_embeds.py --normalize --word_type \
        guatemala,presidente,autoridades,candidatos,asesinato,presupuesto,vicepresidente,negociaciones,netanyahu,social,explotaciones \
        exp/SP/mfcc.dev.gt_words.downsample_10.npz


Results
-------
SWDP average precision:

- Spanish dev: 0.1230192657391301
