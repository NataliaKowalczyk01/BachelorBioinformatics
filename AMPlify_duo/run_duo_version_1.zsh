#!/bin/zsh
python3 src/train_amplify_duo_version_1.py \
    -amp_act_tr ./union_esch_step_new/union_active_Ecoli_tr.fa \
    -non_amp_act_tr ./union_esch_step_new/union_inactive_Ecoli_tr.fa \
    -amp_tox_tr ./union_esch_step_new/union_active_Step_tr.fa \
    -non_amp_tox_tr ./union_esch_step_new/union_inactive_Step_tr.fa \
    -out_dir ./experiment_duo \
    -model_name amplify_duo_constrative_loss