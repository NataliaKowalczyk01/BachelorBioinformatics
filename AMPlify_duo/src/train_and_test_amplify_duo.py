#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 3: amplify_duo
This script is for model training, and testing the performance if a test set is specified
"""
import os
import argparse
from textwrap import dedent
from Bio import SeqIO
import numpy as np
import random
from layers import Attention, MultiHeadAttention
from keras.models import Model
from keras.layers import Masking, Dense, LSTM, Bidirectional, Input, Dropout, Concatenate, Layer
from keras.callbacks import EarlyStopping
from keras.optimizers  import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

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

def contrastive_loss(attention_ecoli, attention_saureus, y_true_ecoli, y_true_saureus, margin=1.0):
    # checking the labels
    y = tf.cast(tf.not_equal(y_true_ecoli, y_true_saureus), dtype=tf.float32)
    """
    tf.print("attention_ecoli shape:", tf.shape(attention_ecoli))
    tf.print("attention_saureus shape:", tf.shape(attention_saureus))
    tf.print("y_true_ecoli shape:", tf.shape(y_true_ecoli))
    tf.print("y_true_saureus shape:", tf.shape(y_true_saureus))
    tf.print("y shape (after cast):", tf.shape(y))
    """

    euclidean_distance = tf.sqrt(tf.reduce_sum(tf.square(attention_ecoli - attention_saureus), axis=1))
    """
    tf.print("euclidean_distance shape:", tf.shape(euclidean_distance))
    tf.print("euclidean_distance values:", euclidean_distance)
    """
    # Contrastive loss
    loss_same_class = (1 - y) * tf.square(euclidean_distance)  # same labek for ecoli and saureus
    loss_diff_class = y * tf.square(tf.maximum(margin - euclidean_distance, 0))  # different label for ecoli and saureus

    loss = tf.reduce_mean(loss_same_class + loss_diff_class)

    # tf.print("Final loss value:", loss)

    return loss
def build_attention():
    """
    Build the model architecture for attention output
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True,
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name = 'Dropout_1')(hidden)
    hidden = Attention(return_attention=True, name='Attention')(hidden)
    model = Model(inputs=inputs, outputs=hidden)
    return model

def build_amplify_architecture_attention():
    """
    Build the complete model architecture
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True,
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name = 'Dropout_1')(hidden)
    attention = Attention(name='Attention', return_attention=False)(hidden)
    model = Model(inputs=inputs, outputs=attention)
    return model

def load_multi_model(model_dir_list, architecture):
    """
    Load multiple models with the same architecture in one function.
    Input: list of saved model weights files.
    Output: list of loaded models.
    """
    model_list = []
    for i in range(len(model_dir_list)):
        model = architecture()
        model.load_weights(model_dir_list[i], by_name=True)
        model_list.append(model)
    return model_list

'''
def load_base_model():


    model_amplify_ecoli =r'/Users/nataliakowalczyk/projects/bcs/AMPlify/models/balanced/AMPlify_balanced_model_weights_1.h5'
    model_amplify_saureus =r'/Users/nataliakowalczyk/projects/bcs/AMPlify/models/balanced/AMPlify_balanced_model_weights_1.h5'
    models_amplify=[model_amplify_ecoli, model_amplify_saureus]

    out_models = load_multi_model(models_amplify, build_amplify_architecture_attention)
    return out_models
'''

def load_base_model():

    current_dir = os.getcwd()

    model_amplify_ecoli = os.path.join(current_dir, 'models_amplify_ecoli', 'amplify_ecoli_weights_1.h5')
    model_amplify_saureus = os.path.join(current_dir, 'models_amplify_saureus', 'amplify_saureus_weights_1.h5')

    models_amplify = [model_amplify_ecoli, model_amplify_saureus]

    out_models = load_multi_model(models_amplify, build_amplify_architecture_attention)
    return out_models


class BaseModelLayer(Layer):
    """Stack of Linear layers with a sparsity regularization loss."""

    def __init__(self):
        super().__init__()
        self.model_amplify_ecoli = load_base_model()[0]
        self.model_amplify_saureus = load_base_model()[1]

        for layer in self.model_amplify_ecoli.layers:
            layer.trainable = False
        for layer in self.model_amplify_saureus.layers:
            layer.trainable = False

    def call(self, inputs, y_true_ecoli, y_true_saureus):

        input_data = inputs
        attention_ecoli = self.model_amplify_ecoli(input_data)
        attention_saureus = self.model_amplify_saureus(input_data)

        concatenated = Concatenate(name='Concatenate', axis=-1)([attention_ecoli, attention_saureus])

        self.add_loss(contrastive_loss(attention_ecoli, attention_saureus, y_true_ecoli, y_true_saureus))

        return concatenated

    def get_config(self):
        config = super().get_config()
        return config

def build_duo_model_with_custom_layer():
    """
    Build the model architecture using BaseModelLayer.
    """
    inputs_data = Input(shape=(MAX_LEN, 20), name='Input_duo')

    y_true_ecoli = Input(shape=(), name='y_true_ecoli', dtype=tf.float32)
    y_true_saureus = Input(shape=(), name='y_true_saureus', dtype=tf.float32)

    base_layer_output = BaseModelLayer()(inputs_data, y_true_ecoli, y_true_saureus)

    # Dense layers for prediction
    prediction_ecoli = Dense(1, activation='sigmoid', name='Output_ecoli')(base_layer_output)
    prediction_saureus = Dense(1, activation='sigmoid', name='Output_saureus')(base_layer_output)

    duo_model = Model(inputs=[inputs_data, y_true_ecoli, y_true_saureus], outputs=[prediction_ecoli, prediction_saureus])

    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
    duo_model.compile(loss={
        'Output_ecoli': 'binary_crossentropy',  # Strata dla aktywności
        'Output_saureus': 'binary_crossentropy'}, optimizer=adam, metrics=['accuracy'])
    print("duo model", duo_model.summary())
    return duo_model


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


def plot_roc_curve(y_true_ecoli, y_pred_ecoli, y_true_saureus, y_pred_saureus, save_path):
    """
    Plot and save ROC curves for both E.coli and S.aureus models.

    Args:
        y_true_ecoli (array): True labels for E.coli.
        y_pred_ecoli (array): Predicted probabilities for E.coli.
        y_true_saureus (array): True labels for S.aureus.
        y_pred_saureus (array): Predicted probabilities for S.aureus.
        save_path (str): Path where the ROC curve image will be saved.
    """
    # Calculate ROC curve for E.coli
    fpr_ecoli, tpr_ecoli, _ = roc_curve(y_true_ecoli, y_pred_ecoli)

    # Calculate ROC curve for S.aureus
    fpr_saureus, tpr_saureus, _ = roc_curve(y_true_saureus, y_pred_saureus)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr_ecoli, tpr_ecoli, color='blue', lw=2, label='E.coli ROC')
    plt.plot(fpr_saureus, tpr_saureus, color='green', lw=2, label='S.aureus ROC')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the figure
    plt.savefig(save_path)
    plt.close()
def main():
    parser = argparse.ArgumentParser(description=dedent('''
        Siamense training
        ------------------------------------------------------
        Given training sets with two labels: AMP and non-AMP,
        train the AMP prediction model.
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-amp_ecoli_tr', help="Training activity AMP set, fasta file", required=True)
    parser.add_argument('-non_amp_ecoli_tr', help="Training activity non-AMP set, fasta file", required=True)

    parser.add_argument('-amp_ecoli_te', help="Test AMP set, fasta file (optional)", default=None, required=False)
    parser.add_argument('-non_amp_ecoli_te', help="Test non-AMP set, fasta file (optional)", default=None, required=False)

    parser.add_argument('-amp_saureus_tr', help="Training activity AMP set, fasta file", required=True)
    parser.add_argument('-non_amp_saureus_tr', help="Training activity non-AMP set, fasta file", required=True)

    parser.add_argument('-amp_saureus_te', help="Test AMP set, fasta file (optional)", default=None, required=False)
    parser.add_argument('-non_amp_saureus_te', help="Test non-AMP set, fasta file (optional)", default=None, required=False)

    parser.add_argument('-out_dir', help="Output directory", required=True)
    parser.add_argument('-model_name', help="File name of trained model weights", required=True)

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    amp_ecoli_train = []
    non_amp_ecoli_train = []
    for seq_record in SeqIO.parse(args.amp_ecoli_tr, 'fasta'):
        amp_ecoli_train.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(args.non_amp_ecoli_tr, 'fasta'):
        non_amp_ecoli_train.append(str(seq_record.seq))

    # sequences for training sets (activity)
    train_ecoli_seq = amp_ecoli_train + non_amp_ecoli_train

    # set labels for training sequences
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

    y_saureus_train = np.array([0]*len(train_ecoli_seq))

    i=0
    for seq in train_ecoli_seq:
        if seq in amp_saureus_train:
            y_saureus_train[i] = 1
        i+=1

    y_union = np.array(list(zip(y_ecoli_train, y_saureus_train)))

    # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_union = one_hot_padding(train_ecoli_seq, MAX_LEN)

    # Set file paths for training and test results
    train_results_file = os.path.join(args.out_dir, 'model_results_duo_margin_1.csv')
    val_results_file = os.path.join(args.out_dir, 'val_model_results_duo_margin_1.csv')

    # Create CSV files if not already present
    create_csv_for_results(train_results_file, 'Train')
    create_csv_for_results(val_results_file, 'Test')


    # Split data into training and validation sets (80% training, 20% validation)
    split_idx = int(0.8 * len(X_union))
    X_train, X_val = X_union[:split_idx], X_union[split_idx:]
    y_train_ecoli, y_val_ecoli = y_ecoli_train[:split_idx], y_ecoli_train[split_idx:]
    y_train_saureus, y_val_saureus = y_saureus_train[:split_idx], y_saureus_train[split_idx:]


    if args.amp_ecoli_te is not None and args.non_amp_ecoli_te is not None and args.amp_saureus_te is not None and args.non_amp_saureus_te is not None:
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
        test_results_file = os.path.join(args.out_dir, 'model_results_duo_test_margine_1.csv')

        # Create CSV files if not already present
        create_csv_for_results(test_results_file, 'Test')


    # Model training
    model = build_duo_model_with_custom_layer()

    early_stopping = EarlyStopping(monitor='val_accuracy',  min_delta=0.001, patience=50, restore_best_weights=True)

    model.fit([X_train, y_train_ecoli, y_train_saureus],
            [y_train_ecoli, y_train_saureus],
            epochs=1000, batch_size=32,
            validation_data=([X_val, y_val_ecoli, y_val_saureus],
                            [y_val_ecoli, y_val_saureus]),
            verbose=2, initial_epoch=0, callbacks=[early_stopping])

    save_file_num = 1
    save_dir_wt = os.path.join(args.out_dir, f'{args.model_name}_weights_{save_file_num}.tf')
    if os.path.exists(save_dir_wt):
        os.remove(save_dir_wt)
    model.save_weights(save_dir_wt, save_format='tf')
    print(f"Zapisano wagi: {save_dir_wt}")

    # Predicting on the training set
    temp_pred_train = model.predict([X_train, y_train_ecoli, y_train_saureus])
    # temp_pred_train contains two arrays: one for ecoli and one for saureus
    train_pred_ecoli = temp_pred_train[0].flatten()
    train_pred_saureus = temp_pred_train[1].flatten()
    metrics_train = calculate_metrics(train_pred_ecoli,train_pred_saureus, y_train_ecoli, y_train_saureus)
    save_metrics_to_csv(metrics_train, save_file_num, train_results_file)

    # Predicting on the validation set
    temp_pred_val = model.predict([[X_val, y_val_ecoli, y_val_saureus]])
    # temp_pred_train contains two arrays: one for ecoli and one for saureus
    val_pred_ecoli = temp_pred_val[0].flatten()
    val_pred_saureus = temp_pred_val[1].flatten()
    metrics_val = calculate_metrics(val_pred_ecoli,val_pred_saureus, y_val_ecoli, y_val_saureus)
    save_metrics_to_csv(metrics_val, save_file_num, val_results_file)

    # Predicting on the test set if provided
    if args.amp_ecoli_te is not None and args.non_amp_ecoli_te is not None and args.amp_saureus_te is not None and args.non_amp_saureus_te is not None:
        temp_pred_test = model.predict([X_test, y_ecoli_test, y_saureus_test])
        test_pred_ecoli = temp_pred_test[0].flatten()
        test_pred_saureus = temp_pred_test[1].flatten()
        metrics_test = calculate_metrics(test_pred_ecoli,test_pred_saureus ,y_ecoli_test, y_saureus_test)

        # Save metrics for test set to test results CSV
        save_metrics_to_csv(metrics_test, save_file_num, test_results_file)
        plot_roc_curve(y_ecoli_test, test_pred_ecoli, y_saureus_test, test_pred_saureus, os.path.join(args.out_dir, 'roc_curve_test_margin1.png'))


if __name__ == "__main__":
    main()
