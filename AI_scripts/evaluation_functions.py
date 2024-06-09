import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_curve, auc, precision_recall_curve, precision_score, f1_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mplcatppuccin.palette import load_color
import re
import pandas as pd


def plot_metrics(metrics, fig_name_pdf):
    df = pd.DataFrame(metrics)
    df_melt = df.melt('Metrics', var_name='Metric', value_name='Values')
    plt.figure(figsize=(12, 12))
    ax = sns.barplot(x='Metrics', y='Values', hue='Metric', data=df_melt, palette='pastel')

    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{0:.2f}'.format(p.get_height()), 
                fontsize=15, color='black', ha='center', va='bottom')

    # Set x and y label font sizes
    ax.set_xlabel(ax.get_xlabel(), fontsize=20, labelpad=-10)  # Adjust labelpad as needed
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)

    # Set legend font size and position it closer to the plot on the right, in the middle
    plt.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0., fontsize=15)
    ncol = df['Metrics'].nunique()
    plt.legend(ncol=ncol, bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=15)

    # Adjust the legend to be in one row and place it below the plot
    legend_ncol = len(df_melt['Metric'].unique())
    ax.legend(ncol=legend_ncol, bbox_to_anchor=(0.5, -0.13), loc='lower center', fontsize=15, frameon=False)
    plt.ylim(0, 1)

    sns.despine()
    # ax.get_legend().remove()

    # Set tick label font size
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(fig_name_pdf, bbox_inches="tight")

    plt.show()


def get_labels_probs_cnn_labels(cnn_output, ground_truth_dict):
    
    cnn_labels = []
    cnn_probabilities = []
    ground_truth_labels = []

    for row in cnn_output:
        read_id = row[0]
        pattern = re.compile(r"_[0-9]+$")
        read_id = pattern.sub("", read_id)
        if ground_truth_dict.get(read_id)== None:
            #print("none")
            #print("no ground truth label", read_id_w_sig)
            continue
        else:
            ground_truth_labels.append(ground_truth_dict.get(read_id))
            cnn_labels.append(row[1])
            cnn_probabilities.append(row[2])
        
    cnn_labels = (np.array(cnn_labels)).astype(float)
    cnn_probabilities = (np.array(cnn_probabilities)).astype(float)
    ground_truth_labels = np.array(ground_truth_labels)

    return cnn_labels, cnn_probabilities, ground_truth_labels


def get_inf_res(dir):
    cnn_output = []
    for filename in os.listdir(dir):
        with open(os.path.join(dir, filename), 'r') as f:
            cnn_output += [line.strip().split('\t') for line in f]
    return cnn_output

def get_groundtruthdict(dir, label_type):
    with open(dir, 'r') as f:
        dict_groundtruth = {}
        next(f)  # skip the header
        for line in f:
            line = line.strip().split('\t')
            #chunk_id = line[1].split('_')[1]
            if label_type == "neg":
                dict_groundtruth[line[0]] = 0
            else:
                dict_groundtruth[line[0]] = 1
    return dict_groundtruth


def calc_all_metrics(ground_truth_labels, cnn_labels, cnn_probabilities):
    accuracy = accuracy_score(ground_truth_labels, cnn_labels)

    recall = recall_score(ground_truth_labels, cnn_labels, pos_label=1)

    tn, fp, fn, tp = confusion_matrix(ground_truth_labels, cnn_labels).ravel()
    specificity = tn / (tn + fp)

    fpr, tpr, _ = roc_curve(ground_truth_labels, cnn_probabilities, pos_label=1)
    roc_auc = auc(fpr, tpr)

    precision_pr, recall_pr, thresholds = precision_recall_curve(ground_truth_labels, cnn_probabilities, pos_label=1)
    aupr = auc(recall_pr, precision_pr)

    precision = precision_score(ground_truth_labels, cnn_labels, pos_label=1)

    f1 = f1_score(ground_truth_labels, cnn_labels, pos_label=1)

    return accuracy, recall, specificity, roc_auc, aupr, precision, f1, fpr, tpr, recall_pr, precision_pr



def count_ratio(ground_truth_labels):

    # total number of elements in the array
    total_elements = ground_truth_labels.size

    # count number of 1s and calculate percentage
    num_ones = np.count_nonzero(ground_truth_labels == 1)
    ratio_ones = (num_ones / total_elements)

    # count number of 0s and calculate percentage
    num_zeros = np.count_nonzero(ground_truth_labels == 0)
    ratio_zeros = (num_zeros / total_elements)

    print(f'Total number of reads: {total_elements}')
    print(f'The percentage of 1s: {ratio_ones}')
    print(f'The percentage of 0s: {ratio_zeros}')
    
    return ratio_zeros, ratio_ones


def calculate_metrics_4_thresholds(cnn_prob, ground_truth_labels, figure_dir, data_name):
    thresholds = np.arange(0.0, 1.01, 0.1)
    sns.set_palette("pastel")

    # Initialize metrics lists
    accuracies, precisions, f1_scores, recalls, specificities = [], [], [], [], []

    # Calculate metrics for each threshold
    for threshold in thresholds:
        cnn_labels = np.where(cnn_prob >= threshold, 1, 0)  # Assign 1 if condition is True, else 1
        accuracies.append(accuracy_score(ground_truth_labels, cnn_labels))
        precisions.append(precision_score(ground_truth_labels, cnn_labels, pos_label=1, zero_division=0))
        f1_scores.append(f1_score(ground_truth_labels, cnn_labels, pos_label=1))
        recalls.append(recall_score(ground_truth_labels, cnn_labels))
        tn, fp, fn, tp = confusion_matrix(ground_truth_labels, cnn_labels).ravel()
        specificities.append(tn / (tn + fp))

    # Determine the thresholds for maximum F1 Score and Accuracy
    max_f1_threshold = thresholds[f1_scores.index(max(f1_scores))]
    max_accuracy_threshold = thresholds[accuracies.index(max(accuracies))]

    line_width = 4

    # Plotting
    plt.figure(figsize=(12, 12))

    plt.plot(thresholds, accuracies, marker='.', linewidth=line_width)  # Adjusted line width
    plt.plot(thresholds, f1_scores, marker='.', linewidth=line_width)  # Adjusted line width
    plt.plot(thresholds, recalls, marker='.', linewidth=line_width)  # Adjusted line width
    plt.plot(thresholds, specificities, marker='.', linewidth=line_width)  # Adjusted line width
    plt.plot(thresholds, precisions, marker='.', linewidth=line_width)  # Adjusted line width

    # Add vertical lines for the best F1 Score and Accuracy
    plt.axvline(x=max_accuracy_threshold, linestyle='--', color='gray', label='Best Accuracy')

    # plt.title(f'Metrics vs. Threshold ({data_name})', fontsize=16)
    plt.xlabel('Threshold', fontsize=20)
    plt.ylabel('Values', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    
    sns.despine()  # Remove the top and right spines
    plt.tight_layout()
    plt.savefig(figure_dir + '/Metrics_vs_threshold_' + data_name + ".pdf", bbox_inches='tight')
    plt.show()


def get_predicted_labels_probs(cnn_output):
    cnn_labels = []
    cnn_probabilities = []

    for row in cnn_output:
        cnn_labels.append(row[1])
        cnn_probabilities.append(row[2])
        
    cnn_labels = (np.array(cnn_labels)).astype(float)
    cnn_probabilities = (np.array(cnn_probabilities)).astype(float)

    return cnn_labels, cnn_probabilities


def plot_combined_aupr_auroc(recall_pr, precision_pr, aupr, fpr, tpr, roc_auc, fig_name_pdf, prevalance=0.5):
    fig, ax_roc = plt.subplots(figsize=(12, 12))

    # Plot the ROC curve
    ax_roc.plot(fpr, tpr, label=f'AUROC = {roc_auc:.2f}', color='green')
    ax_roc.plot([0, 1], [0, 1], color='green', linestyle='--')  # Diagonal line for ROC
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.0])
    ax_roc.set_xlabel('1-Specificity', fontsize=20, color='green')
    ax_roc.set_ylabel('Recall', fontsize=20, color='green')  # Sensitivity is the same as Recall
    ax_roc.tick_params(axis='both', labelsize=15, colors='green')
    ax_roc.spines['bottom'].set_color('black')  # Keep bottom x-axis green
    ax_roc.spines['left'].set_color('black')  # Make left y-axis magenta
    ax_roc.legend(loc="lower right", fontsize=15, frameon=False)

    # PR curve with precision on the right y-axis
    ax_precision = ax_roc.twinx()
    ax_precision.plot(recall_pr, precision_pr, label=f'AUPR = {aupr:.2f}', color='magenta')  # Plot as usual
    ax_precision.set_ylim([0.0, 1.0])  # Correct the limits for precision
    ax_precision.set_ylabel('Precision', fontsize=20, color='magenta')
    ax_precision.tick_params(axis='y', labelsize=15, colors='magenta')
    ax_precision.spines['right'].set_color('black')  # Keep right y-axis magenta
    ax_precision.legend(loc="lower right", fontsize=15, frameon=False, bbox_to_anchor=(1, 0.04))
    ax_precision.axhline(y=prevalance, color='magenta', linestyle='--')

    # Top x-axis for precision
    ax_precision_top = ax_roc.twiny()
    ax_precision_top.set_xlim([0.0, 1.0])  # Set limits for precision on top x-axis
    ax_precision_top.set_xlabel('Recall', fontsize=20, color='magenta')
    ax_precision_top.tick_params(axis='x', labelsize=15, colors='magenta')
    ax_precision_top.spines['top'].set_color('black')  # Make top x-axis magenta

    # Save the figure
    plt.savefig(fig_name_pdf, bbox_inches="tight")

    # Show the plot
    plt.show()




