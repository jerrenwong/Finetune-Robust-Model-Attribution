#!/usr/bin/env python3

import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_per_model(file_path, model_type):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        result = json.load(f)
    per_model = result.get('per_model', {})
    if isinstance(per_model, dict):
        return per_model.get(model_type, None)
    return None

def get_test_0_full_accuracies(classifier_dir, results_dir, model_type):
    dir_patterns = ['test_0_full', 'full_0_test']

    qwen_acc = None
    llama_acc = None

    for dir_pattern in dir_patterns:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, f'evaluation_results_{model_type}_full_0_test.json')
        if os.path.exists(file_path):
            qwen_acc = load_per_model(file_path, 'qwen')
            llama_acc = load_per_model(file_path, 'llama')
            if qwen_acc is not None and llama_acc is not None:
                break

    if qwen_acc is None:
        qwen_acc = 0
    if llama_acc is None:
        llama_acc = 0

    qwen_response_acc = qwen_acc * 100
    llama_response_acc = llama_acc * 100
    overall_acc = (qwen_response_acc + llama_response_acc) / 2

    return qwen_response_acc, llama_response_acc, overall_acc

def get_bert_test_0_full_accuracies(classifier_dir, results_dir):
    dir_patterns = ['test_0_full', 'full_0_test']

    qwen_acc = None
    llama_acc = None

    for dir_pattern in dir_patterns:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_bert_full_0_test.json')
        if os.path.exists(file_path):
            qwen_acc = load_per_model(file_path, 'qwen')
            llama_acc = load_per_model(file_path, 'llama')
            if qwen_acc is not None and llama_acc is not None:
                break

    if qwen_acc is None:
        qwen_acc = 0
    if llama_acc is None:
        llama_acc = 0

    qwen_response_acc = qwen_acc * 100
    llama_response_acc = llama_acc * 100
    overall_acc = (qwen_response_acc + llama_response_acc) / 2

    return qwen_response_acc, llama_response_acc, overall_acc

def main():
    results_dir = 'classifiers/results'

    if not os.path.exists(results_dir):
        print(f"Error: Results directory {results_dir} not found!")
        sys.exit(1)

    # Map timesteps to classifier names
    timesteps = [0, 75, 225, 375]
    bert_labels = [r'$B_0$', r'$B_{75}$', r'$B_{225}$', r'$B_{375}$']
    qwen_labels = [r'$Q_0$', r'$Q_{75}$', r'$Q_{225}$', r'$Q_{375}$']
    llama_labels = [r'$L_0$', r'$L_{75}$', r'$L_{225}$', r'$L_{375}$']

    bert_classifier_map = {
        0: 'bert_classifier_0',
        75: 'bert_classifier_mixed_0_75',
        225: 'bert_classifier_mixed_0_225',
        375: 'bert_classifier_mixed_0_375'
    }

    qwen_classifier_map = {
        0: 'llm_classifier_qwen_0',
        75: 'llm_classifier_qwen_mixed_0_75',
        225: 'llm_classifier_qwen_mixed_0_225',
        375: 'llm_classifier_qwen_mixed_0_375'
    }

    llama_classifier_map = {
        0: 'llm_classifier_llama_0',
        75: 'llm_classifier_llama_mixed_0_75',
        225: 'llm_classifier_llama_mixed_0_225',
        375: 'llm_classifier_llama_mixed_0_375'
    }

    # Collect data for each classifier type
    bert_qwen_acc = []
    bert_llama_acc = []
    bert_overall_acc = []

    qwen_qwen_acc = []
    qwen_llama_acc = []
    qwen_overall_acc = []

    llama_qwen_acc = []
    llama_llama_acc = []
    llama_overall_acc = []

    print("Collecting data from results directory...")
    print("=" * 70)

    for step in timesteps:
        classifier = bert_classifier_map[step]
        qwen_acc, llama_acc, overall_acc = get_bert_test_0_full_accuracies(classifier, results_dir)
        bert_qwen_acc.append(qwen_acc)
        bert_llama_acc.append(llama_acc)
        bert_overall_acc.append(overall_acc)
        print(f"BERT Step {step} ({classifier}):")
        print(f"  Qwen-response: {qwen_acc:.2f}%, Llama-response: {llama_acc:.2f}%, Overall: {overall_acc:.2f}%")

    print()

    for step in timesteps:
        classifier = qwen_classifier_map[step]
        qwen_acc, llama_acc, overall_acc = get_test_0_full_accuracies(classifier, results_dir, 'llm')
        qwen_qwen_acc.append(qwen_acc)
        qwen_llama_acc.append(llama_acc)
        qwen_overall_acc.append(overall_acc)
        print(f"Qwen LLM Step {step} ({classifier}):")
        print(f"  Qwen-response: {qwen_acc:.2f}%, Llama-response: {llama_acc:.2f}%, Overall: {overall_acc:.2f}%")

    print()

    for step in timesteps:
        classifier = llama_classifier_map[step]
        qwen_acc, llama_acc, overall_acc = get_test_0_full_accuracies(classifier, results_dir, 'llm')
        llama_qwen_acc.append(qwen_acc)
        llama_llama_acc.append(llama_acc)
        llama_overall_acc.append(overall_acc)
        print(f"Llama LLM Step {step} ({classifier}):")
        print(f"  Qwen-response: {qwen_acc:.2f}%, Llama-response: {llama_acc:.2f}%, Overall: {overall_acc:.2f}%")

    csv_file = 'classifiers/results/.data/test_0_full_accuracy_data.csv'
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Classifier', 'Timestep', 'Timestep_Label', 'Qwen_Response_Accuracy',
                         'Llama_Response_Accuracy', 'Overall_Accuracy'])

        for i, step in enumerate(timesteps):
            writer.writerow(['BERT', step, bert_labels[i], bert_qwen_acc[i],
                           bert_llama_acc[i], bert_overall_acc[i]])

        for i, step in enumerate(timesteps):
            writer.writerow(['Qwen_LLM', step, qwen_labels[i], qwen_qwen_acc[i],
                           qwen_llama_acc[i], qwen_overall_acc[i]])

        for i, step in enumerate(timesteps):
            writer.writerow(['Llama_LLM', step, llama_labels[i], llama_qwen_acc[i],
                           llama_llama_acc[i], llama_overall_acc[i]])

    print(f"Data saved to {csv_file}")
    print()

    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'serif'

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    x_pos = np.arange(len(timesteps))
    width = 0.65

    colors = {
        'bert': '#2E86AB',
        'qwen': '#A23B72',
        'llama': '#F18F01'
    }

    ax1 = axes[0]
    bars1 = ax1.bar(x_pos, bert_overall_acc, width, color=colors['bert'],
                    alpha=0.9, edgecolor='white', linewidth=2.0, zorder=3)

    ax1.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_title('BERT Classifier', fontsize=14, fontweight='bold', pad=25)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bert_labels, fontsize=13)
    ax1.set_ylim(0, 100)
    ax1.set_yticks(range(0, 101, 10))
    ax1.tick_params(axis='both', which='major', labelsize=11, pad=8)
    ax1.axhline(y=50, color='#666666', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    ax1.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.8, zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, height + 2,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2 = axes[1]
    bars2 = ax2.bar(x_pos, qwen_overall_acc, width, color=colors['qwen'],
                    alpha=0.9, edgecolor='white', linewidth=2.0, zorder=3)

    ax2.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_title('Qwen LLM Classifier', fontsize=14, fontweight='bold', pad=25)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(qwen_labels, fontsize=13)
    ax2.set_ylim(0, 100)
    ax2.set_yticks(range(0, 101, 10))
    ax2.tick_params(axis='both', which='major', labelsize=11, pad=8)
    ax2.axhline(y=50, color='#666666', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    ax2.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.8, zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, height + 2,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax3 = axes[2]
    bars3 = ax3.bar(x_pos, llama_overall_acc, width, color=colors['llama'],
                    alpha=0.9, edgecolor='white', linewidth=2.0, zorder=3)

    ax3.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax3.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax3.set_title('Llama LLM Classifier', fontsize=14, fontweight='bold', pad=25)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(llama_labels, fontsize=13)
    ax3.set_ylim(0, 100)
    ax3.set_yticks(range(0, 101, 10))
    ax3.tick_params(axis='both', which='major', labelsize=11, pad=8)
    ax3.axhline(y=50, color='#666666', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    ax3.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.8, zorder=0)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_linewidth(1.2)
    ax3.spines['bottom'].set_linewidth(1.2)

    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, height + 2,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.suptitle('Classifier Accuracy on Base Responses',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    output_file = 'classifiers/results/.plots/test_0_full_accuracy_by_timesteps.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print()
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
