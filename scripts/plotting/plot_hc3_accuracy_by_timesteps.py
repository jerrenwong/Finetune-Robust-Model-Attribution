#!/usr/bin/env python3
"""
Plot classifier accuracy on HC3 questions by finetuning steps.

Creates three side-by-side bar charts:
- BERT classifier at different timesteps
- Qwen LLM classifier at different timesteps
- Llama LLM classifier at different timesteps

For each classifier, shows 4 timesteps, each with 3 bars:
- Average accuracy on qwen-responses
- Average accuracy on llama-responses
- Overall average accuracy (average of the two above)
"""

import json
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_per_model(file_path, model_type):
    """Load per_model accuracy from JSON file"""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            result = json.load(f)
        per_model = result.get('per_model', {})
        if isinstance(per_model, dict):
            return per_model.get(model_type, None)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None

def get_hc3_accuracies_individual(classifier_dir, results_dir, model_type='llm'):
    """
    Get HC3 accuracies for individual model types.
    Returns: (finetuned_qwen_acc, finetuned_llama_acc, uncensored_qwen_acc, uncensored_llama_acc)
    """
    # Try different directory name patterns
    dir_patterns = {
        'finetuned_qwen': ['finetuned_qwen_hc3_questions_test', 'finetuned_qwen_hc3'],
        'uncensored_qwen': ['uncensored_qwen_hc3_questions_test', 'uncensored_qwen_hc3'],
        'finetuned_llama': ['finetuned_llama_hc3_questions_test', 'finetuned_llama_hc3'],
        'uncensored_llama': ['uncensored_llama_hc3_questions_test', 'uncensored_llama_hc3']
    }

    # Get finetuned qwen accuracy
    finetune_qwen_qwen = None
    for dir_pattern in dir_patterns['finetuned_qwen']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, f'evaluation_results_{model_type}_hc3_questions_test.json')
        if os.path.exists(file_path):
            finetune_qwen_qwen = load_per_model(file_path, 'qwen')
            if finetune_qwen_qwen is not None:
                break

    # Get uncensored qwen accuracy
    uncensored_qwen_qwen = None
    for dir_pattern in dir_patterns['uncensored_qwen']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, f'evaluation_results_{model_type}_hc3_questions_test.json')
        if os.path.exists(file_path):
            uncensored_qwen_qwen = load_per_model(file_path, 'qwen')
            if uncensored_qwen_qwen is not None:
                break

    # Get finetuned llama accuracy
    finetune_llama_llama = None
    for dir_pattern in dir_patterns['finetuned_llama']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, f'evaluation_results_{model_type}_hc3_questions_test.json')
        if os.path.exists(file_path):
            finetune_llama_llama = load_per_model(file_path, 'llama')
            if finetune_llama_llama is not None:
                break

    # Get uncensored llama accuracy
    uncensored_llama_llama = None
    for dir_pattern in dir_patterns['uncensored_llama']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, f'evaluation_results_{model_type}_hc3_questions_test.json')
        if os.path.exists(file_path):
            uncensored_llama_llama = load_per_model(file_path, 'llama')
            if uncensored_llama_llama is not None:
                break

    # Convert to percentages, default to 0 if None
    finetune_qwen_acc = (finetune_qwen_qwen * 100) if finetune_qwen_qwen is not None else 0
    finetune_llama_acc = (finetune_llama_llama * 100) if finetune_llama_llama is not None else 0
    uncensored_qwen_acc = (uncensored_qwen_qwen * 100) if uncensored_qwen_qwen is not None else 0
    uncensored_llama_acc = (uncensored_llama_llama * 100) if uncensored_llama_llama is not None else 0

    return finetune_qwen_acc, finetune_llama_acc, uncensored_qwen_acc, uncensored_llama_acc

def get_bert_hc3_accuracies_individual(classifier_dir, results_dir):
    """
    Get HC3 accuracies for individual model types for BERT classifier.
    Returns: (finetuned_qwen_acc, finetuned_llama_acc, uncensored_qwen_acc, uncensored_llama_acc)
    """
    # Try different directory name patterns
    dir_patterns = {
        'finetuned_qwen': ['hc3_questions_test_finetuned_qwen', 'finetuned_qwen_hc3'],
        'uncensored_qwen': ['hc3_questions_test_uncensored_qwen', 'uncensored_qwen_hc3'],
        'finetuned_llama': ['hc3_questions_test_finetuned_llama', 'finetuned_llama_hc3'],
        'uncensored_llama': ['hc3_questions_test_uncensored_llama', 'uncensored_llama_hc3']
    }

    # Get finetuned qwen accuracy
    finetune_qwen_qwen = None
    for dir_pattern in dir_patterns['finetuned_qwen']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_bert_hc3_questions_test.json')
        if os.path.exists(file_path):
            finetune_qwen_qwen = load_per_model(file_path, 'qwen')
            if finetune_qwen_qwen is not None:
                break

    # Get uncensored qwen accuracy
    uncensored_qwen_qwen = None
    for dir_pattern in dir_patterns['uncensored_qwen']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_bert_hc3_questions_test.json')
        if os.path.exists(file_path):
            uncensored_qwen_qwen = load_per_model(file_path, 'qwen')
            if uncensored_qwen_qwen is not None:
                break

    # Get finetuned llama accuracy
    finetune_llama_llama = None
    for dir_pattern in dir_patterns['finetuned_llama']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_bert_hc3_questions_test.json')
        if os.path.exists(file_path):
            finetune_llama_llama = load_per_model(file_path, 'llama')
            if finetune_llama_llama is not None:
                break

    # Get uncensored llama accuracy
    uncensored_llama_llama = None
    for dir_pattern in dir_patterns['uncensored_llama']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_bert_hc3_questions_test.json')
        if os.path.exists(file_path):
            uncensored_llama_llama = load_per_model(file_path, 'llama')
            if uncensored_llama_llama is not None:
                break

    # Convert to percentages, default to 0 if None
    finetune_qwen_acc = (finetune_qwen_qwen * 100) if finetune_qwen_qwen is not None else 0
    finetune_llama_acc = (finetune_llama_llama * 100) if finetune_llama_llama is not None else 0
    uncensored_qwen_acc = (uncensored_qwen_qwen * 100) if uncensored_qwen_qwen is not None else 0
    uncensored_llama_acc = (uncensored_llama_llama * 100) if uncensored_llama_llama is not None else 0

    return finetune_qwen_acc, finetune_llama_acc, uncensored_qwen_acc, uncensored_llama_acc

def get_hc3_accuracies(classifier_dir, results_dir):
    """
    Get HC3 accuracies for a classifier.
    Returns: (qwen_response_acc, llama_response_acc, overall_acc)
    """
    # Try different directory name patterns
    dir_patterns = {
        'finetuned_qwen': ['finetuned_qwen_hc3_questions_test', 'finetuned_qwen_hc3'],
        'uncensored_qwen': ['uncensored_qwen_hc3_questions_test', 'uncensored_qwen_hc3'],
        'finetuned_llama': ['finetuned_llama_hc3_questions_test', 'finetuned_llama_hc3'],
        'uncensored_llama': ['uncensored_llama_hc3_questions_test', 'uncensored_llama_hc3']
    }

    # Get qwen-response accuracy
    finetune_qwen_qwen = None
    uncensored_qwen_qwen = None

    for dir_pattern in dir_patterns['finetuned_qwen']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_llm_hc3_questions_test.json')
        if os.path.exists(file_path):
            finetune_qwen_qwen = load_per_model(file_path, 'qwen')
            if finetune_qwen_qwen is not None:
                break

    for dir_pattern in dir_patterns['uncensored_qwen']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_llm_hc3_questions_test.json')
        if os.path.exists(file_path):
            uncensored_qwen_qwen = load_per_model(file_path, 'qwen')
            if uncensored_qwen_qwen is not None:
                break

    # Get llama-response accuracy
    finetune_llama_llama = None
    uncensored_llama_llama = None

    for dir_pattern in dir_patterns['finetuned_llama']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_llm_hc3_questions_test.json')
        if os.path.exists(file_path):
            finetune_llama_llama = load_per_model(file_path, 'llama')
            if finetune_llama_llama is not None:
                break

    for dir_pattern in dir_patterns['uncensored_llama']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_llm_hc3_questions_test.json')
        if os.path.exists(file_path):
            uncensored_llama_llama = load_per_model(file_path, 'llama')
            if uncensored_llama_llama is not None:
                break

    # Calculate accuracies
    if finetune_qwen_qwen is None:
        finetune_qwen_qwen = 0
    if uncensored_qwen_qwen is None:
        uncensored_qwen_qwen = 0
    if finetune_llama_llama is None:
        finetune_llama_llama = 0
    if uncensored_llama_llama is None:
        uncensored_llama_llama = 0

    qwen_response_acc = (finetune_qwen_qwen + uncensored_qwen_qwen) / 2 * 100
    llama_response_acc = (finetune_llama_llama + uncensored_llama_llama) / 2 * 100
    overall_acc = (qwen_response_acc + llama_response_acc) / 2

    return qwen_response_acc, llama_response_acc, overall_acc

def get_bert_hc3_accuracies(classifier_dir, results_dir):
    """
    Get HC3 accuracies for a BERT classifier.
    Returns: (qwen_response_acc, llama_response_acc, overall_acc)
    """
    # Try different directory name patterns
    dir_patterns = {
        'finetuned_qwen': ['hc3_questions_test_finetuned_qwen', 'finetuned_qwen_hc3'],
        'uncensored_qwen': ['hc3_questions_test_uncensored_qwen', 'uncensored_qwen_hc3'],
        'finetuned_llama': ['hc3_questions_test_finetuned_llama', 'finetuned_llama_hc3'],
        'uncensored_llama': ['hc3_questions_test_uncensored_llama', 'uncensored_llama_hc3']
    }

    # Get qwen-response accuracy
    finetune_qwen_qwen = None
    uncensored_qwen_qwen = None

    for dir_pattern in dir_patterns['finetuned_qwen']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_bert_hc3_questions_test.json')
        if os.path.exists(file_path):
            finetune_qwen_qwen = load_per_model(file_path, 'qwen')
            if finetune_qwen_qwen is not None:
                break

    for dir_pattern in dir_patterns['uncensored_qwen']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_bert_hc3_questions_test.json')
        if os.path.exists(file_path):
            uncensored_qwen_qwen = load_per_model(file_path, 'qwen')
            if uncensored_qwen_qwen is not None:
                break

    # Get llama-response accuracy
    finetune_llama_llama = None
    uncensored_llama_llama = None

    for dir_pattern in dir_patterns['finetuned_llama']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_bert_hc3_questions_test.json')
        if os.path.exists(file_path):
            finetune_llama_llama = load_per_model(file_path, 'llama')
            if finetune_llama_llama is not None:
                break

    for dir_pattern in dir_patterns['uncensored_llama']:
        file_path = os.path.join(results_dir, classifier_dir, dir_pattern, 'evaluation_results_bert_hc3_questions_test.json')
        if os.path.exists(file_path):
            uncensored_llama_llama = load_per_model(file_path, 'llama')
            if uncensored_llama_llama is not None:
                break

    # Calculate accuracies
    if finetune_qwen_qwen is None:
        finetune_qwen_qwen = 0
    if uncensored_qwen_qwen is None:
        uncensored_qwen_qwen = 0
    if finetune_llama_llama is None:
        finetune_llama_llama = 0
    if uncensored_llama_llama is None:
        uncensored_llama_llama = 0

    qwen_response_acc = (finetune_qwen_qwen + uncensored_qwen_qwen) / 2 * 100
    llama_response_acc = (finetune_llama_llama + uncensored_llama_llama) / 2 * 100
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

    # Collect data for each classifier type (overall)
    bert_qwen_acc = []
    bert_llama_acc = []
    bert_overall_acc = []

    qwen_qwen_acc = []
    qwen_llama_acc = []
    qwen_overall_acc = []

    llama_qwen_acc = []
    llama_llama_acc = []
    llama_overall_acc = []

    # Collect data for individual model types
    # BERT classifier individual model accuracies
    bert_finetuned_qwen = []
    bert_finetuned_llama = []
    bert_uncensored_qwen = []
    bert_uncensored_llama = []

    # Qwen LLM classifier individual model accuracies
    qwen_finetuned_qwen = []
    qwen_finetuned_llama = []
    qwen_uncensored_qwen = []
    qwen_uncensored_llama = []

    # Llama LLM classifier individual model accuracies
    llama_finetuned_qwen = []
    llama_finetuned_llama = []
    llama_uncensored_qwen = []
    llama_uncensored_llama = []

    print("Collecting data from results directory...")
    print("=" * 70)

    # BERT classifiers
    for step in timesteps:
        classifier = bert_classifier_map[step]
        qwen_acc, llama_acc, overall_acc = get_bert_hc3_accuracies(classifier, results_dir)
        bert_qwen_acc.append(qwen_acc)
        bert_llama_acc.append(llama_acc)
        bert_overall_acc.append(overall_acc)

        # Get individual model accuracies
        fin_qwen, fin_llama, unc_qwen, unc_llama = get_bert_hc3_accuracies_individual(classifier, results_dir)
        bert_finetuned_qwen.append(fin_qwen)
        bert_finetuned_llama.append(fin_llama)
        bert_uncensored_qwen.append(unc_qwen)
        bert_uncensored_llama.append(unc_llama)

        print(f"BERT Step {step} ({classifier}):")
        print(f"  Qwen-response: {qwen_acc:.2f}%, Llama-response: {llama_acc:.2f}%, Overall: {overall_acc:.2f}%")

    print()

    # Qwen LLM classifiers
    for step in timesteps:
        classifier = qwen_classifier_map[step]
        qwen_acc, llama_acc, overall_acc = get_hc3_accuracies(classifier, results_dir)
        qwen_qwen_acc.append(qwen_acc)
        qwen_llama_acc.append(llama_acc)
        qwen_overall_acc.append(overall_acc)

        # Get individual model accuracies
        fin_qwen, fin_llama, unc_qwen, unc_llama = get_hc3_accuracies_individual(classifier, results_dir, 'llm')
        qwen_finetuned_qwen.append(fin_qwen)
        qwen_finetuned_llama.append(fin_llama)
        qwen_uncensored_qwen.append(unc_qwen)
        qwen_uncensored_llama.append(unc_llama)

        print(f"Qwen LLM Step {step} ({classifier}):")
        print(f"  Qwen-response: {qwen_acc:.2f}%, Llama-response: {llama_acc:.2f}%, Overall: {overall_acc:.2f}%")

    print()

    # Llama LLM classifiers
    for step in timesteps:
        classifier = llama_classifier_map[step]
        qwen_acc, llama_acc, overall_acc = get_hc3_accuracies(classifier, results_dir)
        llama_qwen_acc.append(qwen_acc)
        llama_llama_acc.append(llama_acc)
        llama_overall_acc.append(overall_acc)

        # Get individual model accuracies
        fin_qwen, fin_llama, unc_qwen, unc_llama = get_hc3_accuracies_individual(classifier, results_dir, 'llm')
        llama_finetuned_qwen.append(fin_qwen)
        llama_finetuned_llama.append(fin_llama)
        llama_uncensored_qwen.append(unc_qwen)
        llama_uncensored_llama.append(unc_llama)

        print(f"Llama LLM Step {step} ({classifier}):")
        print(f"  Qwen-response: {qwen_acc:.2f}%, Llama-response: {llama_acc:.2f}%, Overall: {overall_acc:.2f}%")

    # Save all data to CSV
    csv_file = 'classifiers/results/.data/hc3_accuracy_data.csv'
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Classifier', 'Timestep', 'Timestep_Label', 'Qwen_Response_Accuracy',
                         'Llama_Response_Accuracy', 'Overall_Accuracy'])

        # BERT
        for i, step in enumerate(timesteps):
            writer.writerow(['BERT', step, bert_labels[i], bert_qwen_acc[i],
                           bert_llama_acc[i], bert_overall_acc[i]])

        # Qwen LLM
        for i, step in enumerate(timesteps):
            writer.writerow(['Qwen_LLM', step, qwen_labels[i], qwen_qwen_acc[i],
                           qwen_llama_acc[i], qwen_overall_acc[i]])

        # Llama LLM
        for i, step in enumerate(timesteps):
            writer.writerow(['Llama_LLM', step, llama_labels[i], llama_qwen_acc[i],
                           llama_llama_acc[i], llama_overall_acc[i]])

    print(f"Data saved to {csv_file}")
    print()

    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
    plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX fonts for math
    plt.rcParams['font.family'] = 'serif'

    # Create figure with three subplots side by side - only overall accuracy
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    x_pos = np.arange(len(timesteps))
    width = 0.65  # Slightly narrower bars for better spacing

    # Color scheme: 0=black, 75=blue, 225=yellow, 375=red
    timestep_colors = {
        0: '#000000',   # Black
        75: '#137efb',  # Blue
        225: '#fed032', # Yellow
        375: '#fc3042'  # Red
    }
    bar_colors = [timestep_colors[step] for step in timesteps]

    # Plot 1: BERT - Overall only
    ax1 = axes[0]
    bars1 = ax1.bar(x_pos, bert_overall_acc, width, color=bar_colors,
                    alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)

    ax1.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_title('BERT Classifier', fontsize=14, fontweight='bold', pad=20)
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

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, height + 2,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Qwen LLM - Overall only
    ax2 = axes[1]
    bars2 = ax2.bar(x_pos, qwen_overall_acc, width, color=bar_colors,
                    alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)

    ax2.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_title('Qwen LLM Classifier', fontsize=14, fontweight='bold', pad=20)
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

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, height + 2,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 3: Llama LLM - Overall only
    ax3 = axes[2]
    bars3 = ax3.bar(x_pos, llama_overall_acc, width, color=bar_colors,
                    alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)

    ax3.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax3.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax3.set_title('Llama LLM Classifier', fontsize=14, fontweight='bold', pad=20)
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

    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, height + 2,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Set main title
    fig.suptitle('Classifier Accuracy on Wild Model\'s Response towards HC3 Questions',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save the plot
    output_file = 'classifiers/results/.plots/hc3_accuracy_by_timesteps.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print()
    print(f"Plot saved to {output_file}")

    # Create individual model plots
    model_types = [
        ('finetuned_qwen', 'Finetuned Qwen', bert_finetuned_qwen, qwen_finetuned_qwen, llama_finetuned_qwen),
        ('finetuned_llama', 'Finetuned Llama', bert_finetuned_llama, qwen_finetuned_llama, llama_finetuned_llama),
        ('uncensored_qwen', 'Uncensored Qwen', bert_uncensored_qwen, qwen_uncensored_qwen, llama_uncensored_qwen),
        ('uncensored_llama', 'Uncensored Llama', bert_uncensored_llama, qwen_uncensored_llama, llama_uncensored_llama)
    ]

    for model_key, model_title, bert_data, qwen_data, llama_data in model_types:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.patch.set_facecolor('white')

        # Plot 1: BERT
        ax1 = axes[0]
        bars1 = ax1.bar(x_pos, bert_data, width, color=bar_colors,
                        alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)
        ax1.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
        ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
        ax1.set_title('BERT Classifier', fontsize=14, fontweight='bold', pad=20)
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

        # Plot 2: Qwen LLM
        ax2 = axes[1]
        bars2 = ax2.bar(x_pos, qwen_data, width, color=bar_colors,
                        alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)
        ax2.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
        ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
        ax2.set_title('Qwen LLM Classifier', fontsize=14, fontweight='bold', pad=20)
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

        # Plot 3: Llama LLM
        ax3 = axes[2]
        bars3 = ax3.bar(x_pos, llama_data, width, color=bar_colors,
                        alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)
        ax3.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
        ax3.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
        ax3.set_title('Llama LLM Classifier', fontsize=14, fontweight='bold', pad=20)
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

        # Set main title
        fig.suptitle(f'Classifier Accuracy on {model_title} Responses towards HC3 Questions',
                     fontsize=15, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save the plot
        output_file = f'classifiers/results/.plots/hc3_accuracy_by_timesteps_{model_key}.png'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {output_file}")

    # Create combined finetuned plot (average of finetuned_qwen and finetuned_llama)
    bert_finetuned_combined = [(bert_finetuned_qwen[i] + bert_finetuned_llama[i]) / 2 for i in range(len(timesteps))]
    qwen_finetuned_combined = [(qwen_finetuned_qwen[i] + qwen_finetuned_llama[i]) / 2 for i in range(len(timesteps))]
    llama_finetuned_combined = [(llama_finetuned_qwen[i] + llama_finetuned_llama[i]) / 2 for i in range(len(timesteps))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    # Plot 1: BERT
    ax1 = axes[0]
    bars1 = ax1.bar(x_pos, bert_finetuned_combined, width, color=bar_colors,
                    alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)
    ax1.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_title('BERT Classifier', fontsize=14, fontweight='bold', pad=20)
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

    # Plot 2: Qwen LLM
    ax2 = axes[1]
    bars2 = ax2.bar(x_pos, qwen_finetuned_combined, width, color=bar_colors,
                    alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)
    ax2.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_title('Qwen LLM Classifier', fontsize=14, fontweight='bold', pad=20)
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

    # Plot 3: Llama LLM
    ax3 = axes[2]
    bars3 = ax3.bar(x_pos, llama_finetuned_combined, width, color=bar_colors,
                    alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)
    ax3.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax3.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax3.set_title('Llama LLM Classifier', fontsize=14, fontweight='bold', pad=20)
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

    # Set main title
    fig.suptitle('Classifier Accuracy on Fine-tuned Model\'s Response towards HC3 Questions',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save the plot
    output_file = 'classifiers/results/.plots/finetuned_hc3_accuracy_by_timesteps.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")

    # Create combined uncensored plot (average of uncensored_qwen and uncensored_llama)
    bert_uncensored_combined = [(bert_uncensored_qwen[i] + bert_uncensored_llama[i]) / 2 for i in range(len(timesteps))]
    qwen_uncensored_combined = [(qwen_uncensored_qwen[i] + qwen_uncensored_llama[i]) / 2 for i in range(len(timesteps))]
    llama_uncensored_combined = [(llama_uncensored_qwen[i] + llama_uncensored_llama[i]) / 2 for i in range(len(timesteps))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    # Plot 1: BERT
    ax1 = axes[0]
    bars1 = ax1.bar(x_pos, bert_uncensored_combined, width, color=bar_colors,
                    alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)
    ax1.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_title('BERT Classifier', fontsize=14, fontweight='bold', pad=20)
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

    # Plot 2: Qwen LLM
    ax2 = axes[1]
    bars2 = ax2.bar(x_pos, qwen_uncensored_combined, width, color=bar_colors,
                    alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)
    ax2.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_title('Qwen LLM Classifier', fontsize=14, fontweight='bold', pad=20)
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

    # Plot 3: Llama LLM
    ax3 = axes[2]
    bars3 = ax3.bar(x_pos, llama_uncensored_combined, width, color=bar_colors,
                    alpha=0.95, edgecolor='white', linewidth=2.0, zorder=3)
    ax3.set_xlabel('Classifier', fontsize=13, fontweight='bold', labelpad=10)
    ax3.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax3.set_title('Llama LLM Classifier', fontsize=14, fontweight='bold', pad=20)
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

    # Set main title
    fig.suptitle('Classifier Accuracy on Uncensored Model\'s Response towards HC3 Questions',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save the plot
    output_file = 'classifiers/results/.plots/uncensored_hc3_accuracy_by_timesteps.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
