#!/bin/zsh
python3 src/train_amplify.py \
    -amp_tr ./data_activity/ecoli_active.fa \
    -non_amp_tr ./data_activity/ecoli_inactive.fa \
    -amp_te ./data_test/union_ecoli_active_test.fa \
    -non_amp_te ./data_test/union_ecoli_inactive.fa \
    -out_dir ./models_amplify_ecoli \
    -model_name amplify_ecoli
