#!/usr/bin/env python3
"""
Visualize training accuracy curves for mixed classifiers.

Creates three separate visualization files:
- One for BERT mixed classifiers (0+75, 0+225, 0+375)
- One for Qwen-based LLM mixed classifiers (0+75, 0+225, 0+375)
- One for Llama-based LLM mixed classifiers (0+75, 0+225, 0+375)

Each graph contains 3 curves (one for each mixed variant) with different line styles.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def load_eval_metrics(eval_file_path):
    """
    Load evaluation metrics from a JSONL file.

    Args:
        eval_file_path: Path to eval_metrics.jsonl file

    Returns:
        List of dictionaries containing metrics, or None if file doesn't exist
    """
    if not os.path.exists(eval_file_path):
        return None

    metrics = []
    with open(eval_file_path, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))

    return metrics


def get_mixed_classifier_data(classifier_type, results_dir):
    """
    Get accuracy data for mixed classifiers of a specific type.

    Args:
        classifier_type: 'bert', 'qwen', or 'llama'
        results_dir: Base directory containing classifier results

    Returns:
        Dictionary mapping mixed_variant to {steps: [], accuracies: []}
    """
    # Define mixed variants
    mixed_variants = ['0_75', '0_225', '0_375']

    # Determine directory pattern based on type
    if classifier_type.lower() == 'bert':
        pattern = 'bert_classifier_mixed'
    elif classifier_type.lower() == 'qwen':
        pattern = 'llm_classifier_qwen_mixed'
    elif classifier_type.lower() == 'llama':
        pattern = 'llm_classifier_llama_mixed'
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    # Collect data for each mixed variant
    data_by_variant = {}

    for variant in mixed_variants:
        # Construct directory name
        dir_name = f"{pattern}_{variant}"
        eval_file = os.path.join(results_dir, dir_name, "eval_metrics.jsonl")

        metrics = load_eval_metrics(eval_file)
        if metrics is None:
            print(f"Warning: No eval_metrics.jsonl found for {dir_name}")
            continue

        # Extract global_step and overall accuracy
        steps = []
        accuracies = []
        for entry in metrics:
            if 'global_step' in entry and 'overall' in entry:
                steps.append(entry['global_step'])
                accuracies.append(entry['overall'])

        if steps and accuracies:
            data_by_variant[variant] = {
                'steps': steps,
                'accuracies': accuracies
            }
            print(f"Loaded {len(steps)} data points for {dir_name}")

    return data_by_variant


def plot_mixed_accuracy_over_time(classifier_type, results_dir, output_dir=None):
    """
    Plot accuracy over time for mixed classifiers of a specific type.

    Args:
        classifier_type: 'bert', 'qwen', or 'llama'
        results_dir: Base directory containing classifier results
        output_dir: Directory to save output plots (default: results_dir)
    """
    if output_dir is None:
        output_dir = results_dir

    # Define mixed variants and line styles
    mixed_variants = ['0_75', '0_225', '0_375']
    line_styles = ['-', '--', '-.']  # solid, dashed, dashdot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # Get data for this classifier type
    data_by_variant = get_mixed_classifier_data(classifier_type, results_dir)

    if not data_by_variant:
        print(f"Warning: No data found for {classifier_type} mixed classifiers")
        return

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot each mixed variant with different line style and color
    for i, variant in enumerate(mixed_variants):
        if variant in data_by_variant:
            data = data_by_variant[variant]
            line_style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]
            # Format label: "0+75" instead of "0_75"
            label_parts = variant.split('_')
            label = f'Mixed {label_parts[0]}+{label_parts[1]}'
            plt.plot(
                data['steps'],
                data['accuracies'],
                linestyle=line_style,
                color=color,
                linewidth=2,
                label=label,
                marker='o',
                markersize=3,
                alpha=0.8
            )

    # Customize plot
    plt.xlabel('Global Step', fontsize=12)
    plt.ylabel('Balanced Accuracy', fontsize=12)

    # Set title based on classifier type
    title_map = {
        'bert': 'BERT Mixed Classifier Training Accuracy',
        'qwen': 'Qwen Mixed Classifier Training Accuracy',
        'llama': 'Llama Mixed Classifier Training Accuracy'
    }
    plt.title(title_map.get(classifier_type.lower(), f'{classifier_type} Mixed Classifier Training Accuracy'),
              fontsize=14, fontweight='bold')

    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'mixed_classifier_accuracy_{classifier_type.lower()}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {output_file}")


def plot_overlay_all_mixed_classifiers(results_dir, output_dir=None):
    """
    Plot accuracy over time for all three mixed classifier types on a single graph.
    Uses different colors for each classifier type and different line styles for mixed variants.

    Args:
        results_dir: Base directory containing classifier results
        output_dir: Directory to save output plots (default: results_dir)
    """
    if output_dir is None:
        output_dir = results_dir

    # NeurIPS color scheme (colorblind-friendly)
    neurips_colors = {
        'bert': '#1f77b4',    # Blue
        'qwen': '#ff7f0e',    # Orange
        'llama': '#2ca02c'    # Green
    }

    # Define mixed variants and line styles
    mixed_variants = ['0_75', '0_225', '0_375']
    line_styles = ['-', '--', '-.']  # solid, dashed, dashdot

    # Get data for all classifier types
    classifier_types = ['bert', 'qwen', 'llama']
    all_data = {}
    for classifier_type in classifier_types:
        all_data[classifier_type] = get_mixed_classifier_data(classifier_type, results_dir)

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Plot each classifier type
    for classifier_type in classifier_types:
        data_by_variant = all_data[classifier_type]
        base_color = neurips_colors[classifier_type]

        if not data_by_variant:
            print(f"Warning: No data found for {classifier_type} mixed classifiers")
            continue

        # Plot each mixed variant with different line style
        for i, variant in enumerate(mixed_variants):
            if variant in data_by_variant:
                data = data_by_variant[variant]
                line_style = line_styles[i % len(line_styles)]
                # Format label: "BERT - Mixed 0+75" instead of "bert_0_75"
                label_parts = variant.split('_')
                label = f'{classifier_type.upper()} - Mixed {label_parts[0]}+{label_parts[1]}'
                plt.plot(
                    data['steps'],
                    data['accuracies'],
                    linestyle=line_style,
                    color=base_color,
                    linewidth=2,
                    label=label,
                    marker='o',
                    markersize=2,
                    alpha=0.8
                )

    # Customize plot
    plt.xlabel('Global Step', fontsize=14)
    plt.ylabel('Balanced Accuracy', fontsize=14)
    plt.title('Mixed Classifier Training Accuracy - All Types', fontsize=16, fontweight='bold')

    plt.legend(loc='best', fontsize=9, ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'mixed_classifier_accuracy_all_overlay.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved overlay visualization to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training accuracy curves for mixed classifiers"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='classifiers/results',
        help='Base directory containing classifier results (default: classifiers/results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save output plots (default: same as results-dir)'
    )
    parser.add_argument(
        '--classifier-type',
        type=str,
        choices=['bert', 'qwen', 'llama', 'all', 'overlay'],
        default='all',
        help='Classifier type to visualize (default: all). Use "overlay" to create combined plot.'
    )
    parser.add_argument(
        '--overlay',
        action='store_true',
        help='Create overlay plot with all classifier types (same as --classifier-type overlay)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist.")
        return

    # Handle overlay option
    if args.overlay or args.classifier_type == 'overlay':
        print(f"\n{'='*60}")
        print("Creating overlay visualization with all mixed classifier types...")
        print(f"{'='*60}")
        plot_overlay_all_mixed_classifiers(args.results_dir, args.output_dir)
        print(f"\n{'='*60}")
        print("Overlay visualization complete!")
        print(f"{'='*60}")
        return

    classifier_types = ['bert', 'qwen', 'llama'] if args.classifier_type == 'all' else [args.classifier_type]

    for classifier_type in classifier_types:
        print(f"\n{'='*60}")
        print(f"Visualizing {classifier_type.upper()} mixed classifiers...")
        print(f"{'='*60}")
        plot_mixed_accuracy_over_time(classifier_type, args.results_dir, args.output_dir)

    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
