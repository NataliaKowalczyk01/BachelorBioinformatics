#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 2: amplify_union
This script is for model training, and testing the performance if a test set is specified
"""
import argparse
from textwrap import dedent
from Bio import SeqIO
import numpy as np
import random
from layers import Attention, MultiHeadAttention
from keras.models import Model
from keras.layers import Masking, Dense, LSTM, Bidirectional, Input, Dropout
from keras.callbacks import EarlyStopping
#from keras.optimizers  import Adam
from keras.optimizers.legacy import Adam 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import tensorflow as tf
import csv
import os

MAX_LEN = 100 # max length for input sequences

def one_hot_padding(seq_list,padding):
    """
    Generate features for aa sequences [one-hot encoding with zero padding].
    Input: seq_list: list of sequences,
           padding: padding length, >= max sequence length.
    Output: one-hot encoding of sequences.
    """
    feat_list = []
    one_hot = {}
    aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    for i in range(len(aa)):
        one_hot[aa[i]] = [0]*20
        one_hot[aa[i]][i] = 1
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
            feat.append(one_hot[seq_list[i][j]])
        feat = feat + [[0]*20]*(padding-len(seq_list[i]))
        feat_list.append(feat)
    return(np.array(feat_list))

def map_to_classes(y_union):
    mapping = {
        (1, 0): 1,
        (0, 1): 2,
        (0, 0): 3,
        (1, 1): 4
    }

    classes = np.array([mapping[tuple(row)] for row in y_union])
    return classes

def predict_by_class(scores):
    """
    Turn prediction scores into classes.
    If score > 0.5, label the sample as 1; else 0.
    Input: scores - scores predicted by the model, 1-d array.
    Output: an array of 0s and 1s.
    """
    classes = []
    for i in range(len(scores)):
        if scores[i]>0.5:
            classes.append(1)
        else:
            classes.append(0)
    return np.array(classes)

def build_model_union():
    """
    Build and compile the model for data union.
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True,
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name = 'Dropout_1')(hidden)
    hidden = Attention(name='Attention')(hidden)
    prediction_ecoli = Dense(1, activation='sigmoid', name='Output_1')(hidden)
    prediction_saureus = Dense(1, activation='sigmoid', name='Output_2')(hidden)

    model = Model(inputs=inputs, outputs=[prediction_ecoli, prediction_saureus])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #best
    model.compile(    loss={
        'Output_1': 'binary_crossentropy',  # Strata dla aktywności
        'Output_2': 'binary_crossentropy'   # Strata dla toksyczności
    }, optimizer=adam, metrics=['accuracy'])
    return model

def create_csv_for_results(file_path, train_or_test):
    """
    Creates a CSV file for storing the results of a test if it doesn't already exist.
    
    Args:
        file_path (str): Path to the CSV file.
        test_type (str): Type of the test (e.g., 'Test', 'Validation', 'Train') to label the columns accordingly.
    """
    try:
        with open(file_path, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                f'{train_or_test}_Model_Number',
                f'{train_or_test}_Accuracy_ecoli',
                f'{train_or_test}_Accuracy_saureus',
                f'{train_or_test}_Accuracy_combined',
                f'{train_or_test}_Sensitivity_ecoli',
                f'{train_or_test}_Specificity_ecoli',
                f'{train_or_test}_F1_Score_ecoli',
                f'{train_or_test}_ROC_AUC_ecoli',
                f'{train_or_test}_Sensitivity_saureus',
                f'{train_or_test}_Specificity_saureus',
                f'{train_or_test}_F1_Score_saureus',
                f'{train_or_test}_ROC_AUC_saureus',
                f'{train_or_test}_Sensitivity_Combined',
                f'{train_or_test}_F1_Score_Combined',
                f'{train_or_test}_ROC_AUC_Combined',
                'TN',
                'FP',
                'FN',
                'TP'
            ])
    except FileExistsError:
        pass

def calculate_metrics(pred_ecoli, pred_saureus, y_train_ecoli, y_train_saureus):
    """
    Calculate accuracy, sensitivity, specificity, F1-score, and ROC AUC for E.coli and S.aureus predictions.

    Args:
        temp_pred (list): List containing two arrays: predicted values for E.coli and S.aureus.
        y_true (numpy.array): Ground truth labels for the test or validation set.
        y_union (numpy.array): Combined true labels for E.coli and S.aureus.

    Returns:
        dict: Dictionary containing calculated metrics for E.coli, S.aureus, and combined results.
    """
    # Convert scores to classes using a threshold (e.g. 0.5)
    pred_classes_ecoli = predict_by_class(pred_ecoli)
    pred_classes_saureus = predict_by_class(pred_saureus)

    # Calculate accuracy for activity and toxicity
    acc_ecoli = accuracy_score(y_train_ecoli, pred_classes_ecoli)
    acc_saureus = accuracy_score(y_train_saureus, pred_classes_saureus)
    acc_combined = (acc_ecoli + acc_saureus) / 2

    # Confusion matrix and metrics for E.coli
    tn_ecoli, fp_ecoli, fn_ecoli, tp_ecoli = confusion_matrix(y_train_ecoli, pred_classes_ecoli).ravel()
    sens_ecoli = tp_ecoli / (tp_ecoli + fn_ecoli)
    spec_ecoli = tn_ecoli / (tn_ecoli + fp_ecoli)
    f1_ecoli = f1_score(y_train_ecoli, pred_classes_ecoli)
    roc_auc_ecoli = roc_auc_score(y_train_ecoli, pred_ecoli)

    # Confusion matrix and metrics for S.aureus
    tn_saureus, fp_saureus, fn_saureus, tp_saureus = confusion_matrix(y_train_saureus, pred_classes_saureus).ravel()
    sens_saureus = tp_saureus / (tp_saureus + fn_saureus)
    spec_saureus = tn_saureus / (tn_saureus + fp_saureus)
    f1_saureus = f1_score(y_train_saureus, pred_classes_saureus)
    roc_auc_saureus = roc_auc_score(y_train_saureus, pred_saureus)

    # Combine metrics
    tn = tn_ecoli + tn_saureus
    fp = fp_ecoli + fp_saureus
    fn = fn_ecoli + fn_saureus
    tp = tp_ecoli + tp_saureus

    sens_combined = (sens_ecoli + sens_saureus) / 2
    f1_combined = (f1_ecoli + f1_saureus) / 2
    roc_auc_combined = (roc_auc_ecoli + roc_auc_saureus) / 2

    return {
        'acc_ecoli': acc_ecoli,
        'acc_saureus': acc_saureus,
        'acc_combined': acc_combined,
        'sens_ecoli': sens_ecoli,
        'spec_ecoli': spec_ecoli,
        'f1_ecoli': f1_ecoli,
        'roc_auc_ecoli': roc_auc_ecoli,
        'sens_saureus': sens_saureus,
        'spec_saureus': spec_saureus,
        'f1_saureus': f1_saureus,
        'roc_auc_saureus': roc_auc_saureus,
        'sens_combined': sens_combined,
        'f1_combined': f1_combined,
        'roc_auc_combined': roc_auc_combined,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

def save_metrics_to_csv(metrics, save_file_num, results_file):
    """
    Save the calculated metrics to a CSV file.

    Args:
        metrics (dict): Dictionary containing calculated metrics.
        save_file_num (int): The model number (to track different models).
        results_file (str): Path to the CSV file to store results.
    """
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            save_file_num,
            metrics['acc_ecoli'],
            metrics['acc_saureus'],
            metrics['acc_combined'],
            metrics['sens_ecoli'],
            metrics['spec_ecoli'],
            metrics['f1_ecoli'],
            metrics['roc_auc_ecoli'],
            metrics['sens_saureus'],
            metrics['spec_saureus'],
            metrics['f1_saureus'],
            metrics['roc_auc_saureus'],
            metrics['sens_combined'],
            metrics['f1_combined'],
            metrics['roc_auc_combined'],
            metrics['tn'],
            metrics['fp'],
            metrics['fn'],
            metrics['tp']
        ])

def main():
    parser = argparse.ArgumentParser(description=dedent('''
        Model union training 
        ------------------------------------------------------
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-amp_ecoli_tr', help="Training activity AMP set, fasta file", required=True)
    parser.add_argument('-non_amp_ecoli_tr', help="Training activity non-AMP set, fasta file", required=True)

    parser.add_argument('-amp_ecoli_te', help="Test AMP set, fasta file (optional)", default=None, required=False)
    parser.add_argument('-non_act_ecoli_te', help="Test non-AMP set, fasta file (optional)", default=None, required=False)

    parser.add_argument('-amp_saureus_tr', help="Training activity AMP set, fasta file", required=True)
    parser.add_argument('-non_amp_saureus_tr', help="Training activity non-AMP set, fasta file", required=True)

    parser.add_argument('-amp_saureus_te', help="Test AMP set, fasta file (optional)", default=None, required=False)
    parser.add_argument('-non_amp_saureus_te', help="Test non-AMP set, fasta file (optional)", default=None, required=False)

    parser.add_argument('-out_dir', help="Output directory", required=True)
    parser.add_argument('-model_name', help="File name of trained model weights", required=True)

    args = parser.parse_args()

    amp_ecoli_train = []
    non_amp_ecoli_train = []
    for seq_record in SeqIO.parse(args.amp_ecoli_tr, 'fasta'):
        amp_ecoli_train.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(args.non_amp_ecoli_tr, 'fasta'):
        non_amp_ecoli_train.append(str(seq_record.seq))

    train_ecoli_seq = amp_ecoli_train + non_amp_ecoli_train
    y_ecoli_train = np.array([1]*len(amp_ecoli_train) + [0]*len(non_amp_ecoli_train))

    # shuffle training set
    train_ecoli = list(zip(train_ecoli_seq, y_ecoli_train))
    random.Random(123).shuffle(train_ecoli)
    train_ecoli_seq, y_ecoli_train = zip(*train_ecoli)
    train_ecoli_seq = list(train_ecoli_seq)
    y_ecoli_train = np.array((y_ecoli_train))

    amp_saureus_train = []
    non_amp_saureus_train = []
    for seq_record in SeqIO.parse(args.amp_saureus_tr, 'fasta'):
        amp_saureus_train.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(args.non_amp_saureus_tr, 'fasta'):
        non_amp_saureus_train.append(str(seq_record.seq))

    train_saureus_seq = amp_saureus_train + non_amp_saureus_train
    y_saureus_train = np.array([0]*len(train_saureus_seq))

    i=0
    for seq in train_ecoli_seq:
        if seq in amp_saureus_train:
            y_saureus_train[i] = 1
        i+=1

    y_union = np.array(list(zip(y_ecoli_train, y_saureus_train)))

    # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_union = one_hot_padding(train_ecoli_seq, MAX_LEN)

    indv_pred_train = [] # list of predicted scores for individual models on the training set

    # Set file paths for training and test results
    train_results_file = os.path.join(args.out_dir, 'model_results_duo.csv')
    val_results_file = os.path.join(args.out_dir, 'val_model_results_duo.csv')

    # Create CSV files if not already present
    create_csv_for_results(train_results_file, 'Train')
    create_csv_for_results(val_results_file, 'Test')

    if args.amp_ecoli_te is not None and args.non_ecoli_amp_te is not None and args.amp_saureus_te is not None and args.non_saureus_amp_te is not None:
        amp_ecoli_test = []
        non_amp_ecoli_test = []
        for seq_record in SeqIO.parse(args.amp_ecoli_te, 'fasta'):
            amp_ecoli_test.append(str(seq_record.seq))
        for seq_record in SeqIO.parse(args.non_amp_ecoli_te, 'fasta'):
            non_amp_ecoli_test.append(str(seq_record.seq))

        test_ecoli_seq = amp_ecoli_test + non_amp_ecoli_test

        # set labels for training sequences
        y_ecoli_test = np.array([1]*len(amp_ecoli_test) + [0]*len(non_amp_ecoli_test))

        # shuffle training set
        test_ecoli = list(zip(test_ecoli_seq, y_ecoli_test))
        random.Random(123).shuffle(test_ecoli)
        test_ecoli_seq, y_ecoli_test = zip(*test_ecoli)
        test_ecoli_seq = list(test_ecoli_seq)
        y_ecoli_test = np.array((y_ecoli_test))

        amp_saureus_test = []
        non_amp_saureus_test = []
        for seq_record in SeqIO.parse(args.amp_saureus_te, 'fasta'):
            amp_saureus_test.append(str(seq_record.seq))
        for seq_record in SeqIO.parse(args.non_amp_saureus_te, 'fasta'):
            non_amp_saureus_test.append(str(seq_record.seq))

        test_saureus_seq = amp_saureus_train + non_amp_saureus_test

        y_saureus_test = np.array([0]*len(test_ecoli_seq))

        i=0
        for seq in test_ecoli_seq:
            if seq in amp_saureus_test:
                y_saureus_test[i] = 1
            i+=1

        y_union_test = np.array(list(zip(y_ecoli_test, y_saureus_test)))

        # generate one-hot encoding input and pad sequences into MAX_LEN long
        X_test = one_hot_padding(test_ecoli_seq, MAX_LEN)

        # Set file paths for test results
        test_results_file = os.path.join(args.out_dir, 'model_results_duo_test.csv')

        # Create CSV files if not already present
        create_csv_for_results(test_results_file, 'Test')


    ensemble_number = 5 # number of training subsets for ensemble
    ensemble = StratifiedKFold(n_splits=ensemble_number, shuffle=True, random_state=50)
    save_file_num = 0

    y_classes = map_to_classes(y_union)

    for tr_ens, te_ens in ensemble.split(X_union, y_classes):

        model = build_model_union()

        early_stopping = EarlyStopping(monitor='val_accuracy',  min_delta=0.001, patience=50, restore_best_weights=True)
        model.fit(X_union[tr_ens], [y_ecoli_train[tr_ens],y_saureus_train[tr_ens]], epochs=2, batch_size=32,
                      validation_data=(X_union[te_ens], [y_ecoli_train[te_ens],y_saureus_train[te_ens]] ), verbose=2, initial_epoch=0, callbacks=[early_stopping])


        save_file_num = save_file_num + 1
        save_dir = args.out_dir + '/' + args.model_name + '_' + str(save_file_num) + '.h5'
        save_dir_wt = args.out_dir + '/' + args.model_name + '_weights_' + str(save_file_num) + '.h5'
        model.save(save_dir) #save
        model.save_weights(save_dir_wt) #save

        # Predicting on the training set
        temp_pred_train = model.predict(X_union[tr_ens])
        # temp_pred_train contains two arrays: one for ecoli and one for saureus
        train_pred_ecoli = temp_pred_train[0].flatten()
        train_pred_saureus = temp_pred_train[1].flatten()
        metrics_train = calculate_metrics(train_pred_ecoli,train_pred_saureus, y_ecoli_train[tr_ens],y_saureus_train[tr_ens])
        save_metrics_to_csv(metrics_train, save_file_num, train_results_file)

        # Predicting on the validation set
        temp_pred_val = model.predict(X_union[te_ens])
        # temp_pred_train contains two arrays: one for ecoli and one for saureus
        val_pred_ecoli = temp_pred_val[0].flatten()
        val_pred_saureus = temp_pred_val[1].flatten()
        metrics_val = calculate_metrics(val_pred_ecoli,val_pred_saureus, y_ecoli_train[te_ens],y_saureus_train[te_ens])
        save_metrics_to_csv(metrics_val, save_file_num, val_results_file)

        # Predicting on the test set if provided
        if args.amp_ecoli_te is not None and args.non_ecoli_amp_te is not None and args.amp_saureus_te is not None and args.non_saureus_amp_te is not None:
            temp_pred_test = model.predict([X_test, y_ecoli_test, y_saureus_test])
            metrics_test = calculate_metrics(temp_pred_test,y_ecoli_test, y_saureus_test)

            # Save metrics for test set to test results CSV
            save_metrics_to_csv(metrics_test, save_file_num, test_results_file)

if __name__ == "__main__":
    main()
