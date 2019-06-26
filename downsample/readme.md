Downsampled Acoustic Word Embeddings
====================================

Overview
--------
MFCCs are downsampled to obtain acoustic word embeddings. These are evaluated
using same-different evaluation.


Downsampling
------------
Perform downsampling on MFCCs without deltas:

    mkdir -p exp/TH
    ./downsample.py --technique resample --frame_dims 13 \
        ../features/mfcc/TH/th.dev.gt_words.npz \
        exp/TH/mfcc.dev.gt_words.downsample_10.npz 10


Evaluation
----------
Evaluate and analyse downsampled MFCCs without deltas:

    ../embeddings/eval_samediff.py --mvn \
        exp/TH/mfcc.dev.gt_words.downsample_10.npz
    ../embeddings/analyse_embeds.py --normalize --word_type \
        guatemala,presidente,autoridades,candidatos,asesinato,presupuesto,vicepresidente,negociaciones,netanyahu,social,explotaciones \
        exp/SP/mfcc.dev.gt_words.downsample_10.npz


Results
-------
SWDP average precision:

- CH dev: 0.11420457
- CR dev: 0.11620668
- HA dev: 0.11831970
- SP dev: 0.12301926
- SW dev: 0.06808896
- TU dev: 0.13914600

- GE dev: 0.08031011
- KO dev: 0.13563458
- TH dev: 0.08781202
- VN dev: 0.02734849

