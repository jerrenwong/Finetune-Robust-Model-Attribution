#!/usr/bin/env python3
"""
Plot training loss over time in log scale for all classifiers.

Creates a 4x3 grid:
- Rows: Pure 0, Mixed 0+75, Mixed 0+225, Mixed 0+375
- Columns: BERT, Qwen, Llama
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_loss_data(loss_file_path):
    """
    Load training loss data from a JSONL file.

    Args:
        loss_file_path: Path to training_loss.jsonl file

    Returns:
        Tuple of (steps, losses) arrays, or (None, None) if file doesn't exist
    """
    if not os.path.exists(loss_file_path):
        return None, None

    steps = []
    losses = []
    with open(loss_file_path, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if 'global_step' in entry and 'loss' in entry:
                    steps.append(entry['global_step'])
                    losses.append(entry['loss'])

    return np.array(steps), np.array(losses)


def calculate_log_moving_average(losses, window_size):
    """
    Calculate moving average in log space (geometric mean) for smoothing the trend.
    This is more appropriate for log-scale plots.

    Args:
        losses: Array of loss values (in linear space)
        window_size: Size of the moving average window

    Returns:
        Array of smoothed values (in linear space, same length as input)
    """
    if len(losses) == 0:
        return losses

    # Filter out non-positive values (can't take log of zero or negative)
    # Replace with a small positive value
    losses_positive = np.maximum(losses, 1e-10)

    # Convert to log space
    log_losses = np.log(losses_positive)

    # Use a smaller window if data is too short
    window_size = min(window_size, len(log_losses))
    if window_size < 2:
        return losses

    # Pad the log data at the beginning to handle edge cases
    padded_log = np.concatenate([log_losses[:window_size-1][::-1], log_losses])

    # Calculate moving average in log space
    smoothed_log = np.convolve(padded_log, np.ones(window_size)/window_size, mode='valid')

    # Return the same length as original and convert back to linear space
    smoothed_log = smoothed_log[:len(losses)]
    smoothed = np.exp(smoothed_log)

    return smoothed


def get_classifier_path(classifier_type, training_type, results_dir):
    """
    Get the path to a classifier's results directory.

    Args:
        classifier_type: 'bert', 'qwen', or 'llama'
        training_type: 'pure_0', 'mixed_0_75', 'mixed_0_225', 'mixed_0_375'
        results_dir: Base results directory

    Returns:
        Path to classifier directory
    """
    if training_type == 'pure_0':
        if classifier_type == 'bert':
            return os.path.join(results_dir, 'bert_classifier_0')
        elif classifier_type == 'qwen':
            return os.path.join(results_dir, 'llm_classifier_qwen_0')
        elif classifier_type == 'llama':
            return os.path.join(results_dir, 'llm_classifier_llama_0')
    elif training_type.startswith('mixed_'):
        variant = training_type.replace('mixed_', '')
        if classifier_type == 'bert':
            return os.path.join(results_dir, f'bert_classifier_mixed_{variant}')
        elif classifier_type == 'qwen':
            return os.path.join(results_dir, f'llm_classifier_qwen_mixed_{variant}')
        elif classifier_type == 'llama':
            return os.path.join(results_dir, f'llm_classifier_llama_mixed_{variant}')

    return None


def plot_training_loss_grid(results_dir, output_file=None):
    """
    Plot training loss in a 4x3 grid.

    Args:
        results_dir: Base directory containing classifier results
        output_file: Output file path (default: results_dir/training_loss_grid.png)
    """
    if output_file is None:
        output_file = os.path.join(results_dir, 'training_loss_grid.png')

    # Define grid structure
    training_types = ['pure_0', 'mixed_0_75', 'mixed_0_225', 'mixed_0_375']
    classifier_types = ['bert', 'qwen', 'llama']

    # Training type labels for row titles
    training_labels = {
        'pure_0': 'Pure 0',
        'mixed_0_75': 'Mixed 0+75',
        'mixed_0_225': 'Mixed 0+225',
        'mixed_0_375': 'Mixed 0+375'
    }

    # Classifier type labels for column titles
    classifier_labels = {
        'bert': 'BERT',
        'qwen': 'Qwen',
        'llama': 'Llama'
    }

    # NeurIPS color scheme (colorblind-friendly, professional)
    # Base colors for each classifier type
    neurips_colors = {
        'bert': '#1f77b4',    # Blue
        'qwen': '#ff7f0e',    # Orange
        'llama': '#2ca02c'    # Green
    }

    # Trend line color (darker version of base color)
    trend_colors = {
        'bert': '#0d4a8c',    # Darker blue
        'qwen': '#cc6600',    # Darker orange
        'llama': '#1e7e1e'    # Darker green
    }

    # Create figure with 4 rows x 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Training Loss Over Time (Log Scale)', fontsize=16, fontweight='bold', y=0.995)

    # Plot each combination
    for row_idx, training_type in enumerate(training_types):
        for col_idx, classifier_type in enumerate(classifier_types):
            ax = axes[row_idx, col_idx]

            # Get classifier path
            classifier_path = get_classifier_path(classifier_type, training_type, results_dir)

            if classifier_path is None:
                ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Load loss data
            loss_file = os.path.join(classifier_path, 'training_loss.jsonl')
            steps, losses = load_loss_data(loss_file)

            if steps is None or losses is None or len(steps) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Get colors for this classifier type
            base_color = neurips_colors[classifier_type]
            trend_color = trend_colors[classifier_type]

            # Calculate adaptive window size (~5% of data points, min 50, max 500)
            window_size = max(50, min(500, int(len(losses) * 0.05)))

            # Calculate trend line using log-space moving average
            trend_losses = calculate_log_moving_average(losses, window_size)

            # Plot raw loss data (lighter, thinner)
            ax.plot(steps, losses, linewidth=0.5, alpha=0.3, color=base_color, label='Raw Loss')

            # Plot trend line (darker, thicker)
            ax.plot(steps, trend_losses, linewidth=2, alpha=0.9, color=trend_color, label='Trend')

            # Set log scale on y-axis
            ax.set_yscale('log')

            # Set labels and title
            if row_idx == 3:  # Bottom row
                ax.set_xlabel('Global Step', fontsize=10)
            if col_idx == 0:  # Left column
                ax.set_ylabel('Loss (log)', fontsize=10)

            # Set row title (leftmost column)
            if col_idx == 0:
                ax.text(-0.15, 0.5, training_labels[training_type],
                       transform=ax.transAxes, rotation=90,
                       ha='center', va='center', fontsize=11, fontweight='bold')

            # Set column title (top row)
            if row_idx == 0:
                ax.set_title(classifier_labels[classifier_type], fontsize=12, fontweight='bold', pad=10)

            # Grid
            ax.grid(True, alpha=0.3, which='both')

            # Set reasonable limits
            if len(losses) > 0:
                y_min = max(1e-4, losses.min() * 0.5)
                y_max = losses.max() * 1.5
                ax.set_ylim(y_min, y_max)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle

    # Save the plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved training loss grid to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training loss over time in a 4x3 grid"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='classifiers/results',
        help='Base directory containing classifier results (default: classifiers/results)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file path (default: results_dir/training_loss_grid.png)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist.")
        return

    if args.output_file is None:
        args.output_file = os.path.join(args.results_dir, 'training_loss_grid.png')

    print(f"\n{'='*60}")
    print("Plotting training loss grid...")
    print(f"{'='*60}\n")

    plot_training_loss_grid(args.results_dir, args.output_file)

    print(f"\n{'='*60}")
    print("Plotting complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
