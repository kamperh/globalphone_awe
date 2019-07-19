Query-by-Example Search on Hausa
================================

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

    ./eval_qbe.py exp/mfcc.cmvn_dd.test/cost_dict.pkl

HA results:


Embedding-based QbE
-------------------
Apply a CAE-RNN to the dense intervals for the different splits:

    

