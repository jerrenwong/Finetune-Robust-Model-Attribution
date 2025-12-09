#!/usr/bin/env python3

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_eval_metrics(eval_file_path):
    if not eval_file_path or not os.path.exists(eval_file_path):
        return None
    metrics = []
    with open(eval_file_path, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics

def main():
    results_dir = 'classifiers/results'

    if not os.path.exists(results_dir):
        print(f"Error: Results directory {results_dir} not found!")
        sys.exit(1)

    # Define classifiers
    classifiers = {
        'BERT': [
            ('bert_classifier_0', 0, 'bert'),
            ('bert_classifier_mixed_0_75', 75, 'bert'),
            ('bert_classifier_mixed_0_225', 225, 'bert'),
            ('bert_classifier_mixed_0_375', 375, 'bert')
        ],
        'Qwen_LLM': [
            ('llm_classifier_qwen_0', 0, 'llm'),
            ('llm_classifier_qwen_mixed_0_75', 75, 'llm'),
            ('llm_classifier_qwen_mixed_0_225', 225, 'llm'),
            ('llm_classifier_qwen_mixed_0_375', 375, 'llm')
        ],
        'Llama_LLM': [
            ('llm_classifier_llama_0', 0, 'llm'),
            ('llm_classifier_llama_mixed_0_75', 75, 'llm'),
            ('llm_classifier_llama_mixed_0_225', 225, 'llm'),
            ('llm_classifier_llama_mixed_0_375', 375, 'llm')
        ]
    }

    # Collect data
    data = {}

    print("Collecting data from results directory...")
    print("=" * 70)

    for classifier_type, classifier_list in classifiers.items():
        data[classifier_type] = {}
        print(f"\n{classifier_type}:")
        for classifier_name, step, model_type in classifier_list:
            eval_file = os.path.join(results_dir, classifier_name, 'eval_metrics.jsonl')
            metrics_list = load_eval_metrics(eval_file)

            if metrics_list and len(metrics_list) > 0:
                steps = []
                accuracies = []
                for metric in metrics_list:
                    gs = metric.get('global_step')
                    acc = metric.get('overall')
                    if gs is not None and acc is not None:
                        steps.append(gs)
                        accuracies.append(acc * 100)

                steps.insert(0, 0)
                accuracies.insert(0, 50.0)

                data[classifier_type][step] = {
                    'steps': steps,
                    'accuracies': accuracies
                }
                print(f"  Step {step}: {len(steps)} evaluation points, range: {min(steps) if steps else 0}-{max(steps) if steps else 0}")
            else:
                print(f"  Step {step}: eval_metrics.jsonl NOT FOUND")
                data[classifier_type][step] = {
                    'steps': [],
                    'accuracies': []
                }

    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'serif'

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    legend_labels = {
        'BERT': [r'$B_0$', r'$B_{75}$', r'$B_{225}$', r'$B_{375}$'],
        'Qwen_LLM': [r'$Q_0$', r'$Q_{75}$', r'$Q_{225}$', r'$Q_{375}$'],
        'Llama_LLM': [r'$L_0$', r'$L_{75}$', r'$L_{225}$', r'$L_{375}$']
    }
    classifier_steps = [0, 75, 225, 375]

    for idx, (classifier_type, ax) in enumerate(zip(['BERT', 'Qwen_LLM', 'Llama_LLM'], axes)):
        classifier_data = data[classifier_type]

        for line_idx, step in enumerate(classifier_steps):
            if step in classifier_data:
                steps = classifier_data[step]['steps']
                accuracies = classifier_data[step]['accuracies']
                if len(steps) > 0 and len(accuracies) > 0:
                    ax.plot(steps, accuracies,
                           color=colors[line_idx], linewidth=2.5, alpha=0.85,
                           marker='o', markersize=6,
                           label=legend_labels[classifier_type][line_idx], zorder=3)

        ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel('Test Accuracy on 375 Responses (%)', fontsize=13, fontweight='bold', labelpad=10)

        title_map = {
            'BERT': 'Bert Classifier',
            'Qwen_LLM': 'Qwen LLM Classifier',
            'Llama_LLM': 'Llama LLM Classifier'
        }
        ax.set_title(title_map[classifier_type], fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(45, 100)
        ax.set_yticks([50, 60, 70, 80, 90, 100])
        ax.tick_params(axis='both', which='major', labelsize=11, pad=8)
        ax.axhline(y=50, color='#666666', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
        ax.grid(True, alpha=0.15, axis='both', linestyle='-', linewidth=0.8, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.legend(fontsize=10, loc='best', framealpha=0.9)

    fig.suptitle('Test Accuracy by Training Steps',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    output_file = 'classifiers/results/.plots/375_test_accuracy_by_classifier_steps.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print()
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
