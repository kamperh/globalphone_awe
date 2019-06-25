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
        exp/SP/mfcc.dev.gt_words.downsample_10.npz 10


Evaluation
----------
Evaluate and analyse downsampled MFCCs without deltas:

    ../embeddings/eval_samediff.py --mvn \
        exp/SP/mfcc.dev.gt_words.downsample_10.npz
    ../embeddings/analyse_embeds.py --normalize --word_type \
        guatemala,presidente,autoridades,candidatos,asesinato,presupuesto,vicepresidente,negociaciones,netanyahu,social,explotaciones \
        exp/SP/mfcc.dev.gt_words.downsample_10.npz


Results
-------
SWDP average precision:

- CH dev: 0.11420457391777498
- CR dev: 0.11620668424021699
- HA dev: 0.1183197025831254
- SP dev: 0.1230192657391301
- SW dev: 0.06808896675713268
- TU dev: 0.1391460087700688
