#!/usr/bin/env python3
"""
Visualize classifier accuracy over time on the testing file.

Creates three separate visualization files:
- One for BERT classifiers
- One for Qwen-based LLM classifiers
- One for Llama-based LLM classifiers

Each graph contains 4 curves (one for each time step: 0, 75, 225, 375)
with different line types but the same color.
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


def extract_time_step_from_path(path):
    """
    Extract time step from classifier result directory path.

    Args:
        path: Path to classifier result directory

    Returns:
        Time step as integer (0, 75, 225, 375) or None
    """
    path_str = str(path)
    for time_step in [0, 75, 225, 375]:
        if f'_{time_step}_' in path_str or path_str.endswith(f'_{time_step}'):
            return time_step
    return None


def get_classifier_data(classifier_type, results_dir):
    """
    Get accuracy data for a specific classifier type across all time steps.

    Args:
        classifier_type: 'bert', 'qwen', or 'llama'
        results_dir: Base directory containing classifier results

    Returns:
        Dictionary mapping time_step to {steps: [], accuracies: []}
    """
    # Define time steps
    time_steps = [0, 75, 225, 375]

    # Determine classifier pattern based on type
    if classifier_type.lower() == 'bert':
        pattern = 'bert_classifier'
        base_model_suffix = 'with_cls'
    elif classifier_type.lower() == 'qwen':
        pattern = 'llm_classifier'
        base_model_suffix = 'qwen_base'
    elif classifier_type.lower() == 'llama':
        pattern = 'llm_classifier'
        base_model_suffix = 'llama_base'
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    # Collect data for each time step
    data_by_time_step = {}

    for time_step in time_steps:
        # Construct directory name
        if classifier_type.lower() == 'bert':
            dir_name = f"{pattern}_{time_step}_{base_model_suffix}"
        else:
            dir_name = f"{pattern}_{time_step}_{base_model_suffix}"

        eval_file = os.path.join(results_dir, dir_name, "eval_metrics.jsonl")

        metrics = load_eval_metrics(eval_file)
        if metrics is None:
            continue

        # Extract global_step and overall accuracy
        steps = []
        accuracies = []
        for entry in metrics:
            if 'global_step' in entry and 'overall' in entry:
                steps.append(entry['global_step'])
                accuracies.append(entry['overall'])

        if steps and accuracies:
            data_by_time_step[time_step] = {
                'steps': steps,
                'accuracies': accuracies
            }

    return data_by_time_step


def plot_accuracy_over_time(classifier_type, results_dir, output_dir=None):
    """
    Plot accuracy over time for a specific classifier type.

    Args:
        classifier_type: 'bert', 'qwen', or 'llama'
        results_dir: Base directory containing classifier results
        output_dir: Directory to save output plots (default: results_dir)
    """
    if output_dir is None:
        output_dir = results_dir

    # Define time steps and line styles
    time_steps = [0, 75, 225, 375]
    line_styles = ['-', '--', '-.', ':']  # solid, dashed, dashdot, dotted

    # Get data for this classifier type
    data_by_time_step = get_classifier_data(classifier_type, results_dir)

    if not data_by_time_step:
        print(f"Warning: No data found for {classifier_type} classifiers")
        return

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Use a single color for all lines
    color = 'blue'

    # Plot each time step with different line style
    for i, time_step in enumerate(time_steps):
        if time_step in data_by_time_step:
            data = data_by_time_step[time_step]
            line_style = line_styles[i % len(line_styles)]
            label = f'Time Step {time_step}'
            plt.plot(
                data['steps'],
                data['accuracies'],
                linestyle=line_style,
                color=color,
                linewidth=2,
                label=label
            )

    # Customize plot
    plt.xlabel('Global Step', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)

    # Set title based on classifier type
    title_map = {
        'bert': 'BERT Classifier Accuracy Over Time',
        'qwen': 'Qwen-based LLM Classifier Accuracy Over Time',
        'llama': 'Llama-based LLM Classifier Accuracy Over Time'
    }
    plt.title(title_map.get(classifier_type.lower(), f'{classifier_type} Classifier Accuracy Over Time'),
              fontsize=14, fontweight='bold')

    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'accuracy_over_time_{classifier_type.lower()}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {output_file}")


def plot_overlay_all_classifiers(results_dir, output_dir=None):
    """
    Plot accuracy over time for all three classifier types on a single graph.
    Uses NeurIPS color scheme with different colors for each classifier type
    and different line styles for time steps.

    Args:
        results_dir: Base directory containing classifier results
        output_dir: Directory to save output plots (default: results_dir)
    """
    if output_dir is None:
        output_dir = results_dir

    # NeurIPS color scheme (colorblind-friendly)
    # Using colors that are commonly used in NeurIPS papers
    neurips_colors = {
        'bert': '#1f77b4',    # Blue
        'qwen': '#ff7f0e',    # Orange
        'llama': '#2ca02c'    # Green
    }

    # Define time steps and line styles
    time_steps = [0, 75, 225, 375]
    line_styles = ['-', '--', '-.', ':']  # solid, dashed, dashdot, dotted

    # Get data for all classifier types
    classifier_types = ['bert', 'qwen', 'llama']
    all_data = {}
    for classifier_type in classifier_types:
        all_data[classifier_type] = get_classifier_data(classifier_type, results_dir)

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Plot each classifier type
    for classifier_type in classifier_types:
        data_by_time_step = all_data[classifier_type]
        color = neurips_colors[classifier_type]

        if not data_by_time_step:
            print(f"Warning: No data found for {classifier_type} classifiers")
            continue

        # Plot each time step with different line style
        for i, time_step in enumerate(time_steps):
            if time_step in data_by_time_step:
                data = data_by_time_step[time_step]
                line_style = line_styles[i % len(line_styles)]
                label = f'{classifier_type.upper()} - Time Step {time_step}'
                plt.plot(
                    data['steps'],
                    data['accuracies'],
                    linestyle=line_style,
                    color=color,
                    linewidth=2,
                    label=label
                )

    # Customize plot
    plt.xlabel('Global Step', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Classifier Accuracy Over Time - All Types', fontsize=16, fontweight='bold')

    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'accuracy_over_time_all_overlay.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved overlay visualization to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize classifier accuracy over time on testing file"
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
        print("Creating overlay visualization with all classifier types...")
        print(f"{'='*60}")
        plot_overlay_all_classifiers(args.results_dir, args.output_dir)
        print(f"\n{'='*60}")
        print("Overlay visualization complete!")
        print(f"{'='*60}")
        return

    classifier_types = ['bert', 'qwen', 'llama'] if args.classifier_type == 'all' else [args.classifier_type]

    for classifier_type in classifier_types:
        print(f"\n{'='*60}")
        print(f"Visualizing {classifier_type.upper()} classifiers...")
        print(f"{'='*60}")
        plot_accuracy_over_time(classifier_type, args.results_dir, args.output_dir)

    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
