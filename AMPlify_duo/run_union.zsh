python3 src/train_ampliy_union.py \
    -amp_act_tr ./data_union/union_ecoli_active_train.fa \
    -non_amp_act_tr ./data_union/union_ecoli_inactive.fa \
    -amp_tox_tr ./data_union/union_saureus_active_train.fa \
    -non_amp_tox_tr ./union_esch_step/union_saureus_inactive_train.fa \
    -out_dir ./models_amplify_union \
    -model_name amplify_union

# amp_act_tr
# non_amp_act_tr
# -amp_act_te
# -non_act_amp_te
# amp_tox_tr
# non_amp_tox_tr
# - amp_tox_te
# - non_tox_amp_te
# out_dir
# model_name
