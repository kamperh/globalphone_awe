Query-by-Example Search on Hausa
================================

Overview
--------
Queries are extracted from validation data and the evaluation data is treated
as the search collection.


Prepare and link data
---------------------
Extract features and link the required speech features:

    ./extract_queries_link_search.py HA

Extract the search intervals:

    ./data_prep_dense_seg.py --min_frames 20 --max_frames 60 --step 3 \
        --n_splits 2 HA


DTW-based QbE
-------------
Get QbE costs and write these to file:

    ./get_dtw_costs.py --n_cpus 4 HA

Evaluate QbE performance:

    ./eval_qbe.py HA exp/HA/cost_dict.pkl

HA results:

    ---------------------------------------------------------------------------
    EER:  0.2655, avg: 0.2918, median: 0.2783, max: 0.4505, min: 0.1844
    AUC:  0.8002, avg: 0.7724, median: 0.7960, max: 0.8766, min: 0.5752
    P@10: 0.4139, avg: 0.3468, median: 0.3550, max: 0.5433, min: 0.0933
    P@N:  0.3257, avg: 0.2870, median: 0.2937, max: 0.4471, min: 0.0836
    ---------------------------------------------------------------------------


Embedding-based QbE
-------------------
Apply a CAE-RNN to the dense intervals for the different splits:

    ./apply_model_dense.py ../embeddings/models/HA.utd/train_cae_rnn/5addd62282/cae.best_val.ckpt HA search.0


