import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_loss_data(loss_file_path):
    """
    Load training loss data from a JSONL file.
    Returns lists of global_step and loss values.
    """
    steps = []
    losses = []

    if not os.path.exists(loss_file_path):
        print(f"Warning: {loss_file_path} does not exist. Skipping.")
        return None, None

    try:
        with open(loss_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    steps.append(data.get('global_step', len(steps) + 1))
                    losses.append(data.get('loss', 0.0))

        if len(steps) == 0:
            print(f"Warning: {loss_file_path} is empty. Skipping.")
            return None, None

        return steps, losses
    except Exception as e:
        print(f"Error loading {loss_file_path}: {e}")
        return None, None


def calculate_log_moving_average(data, window_size):
    """
    Calculate moving average in log space (geometric mean) for smoothing the trend.
    This is more appropriate for log-scale plots.

    Args:
        data: List or array of values (in linear space)
        window_size: Size of the moving average window

    Returns:
        Array of smoothed values (in linear space, same length as input)
    """
    data = np.array(data)
    if len(data) == 0:
        return data

    # Filter out non-positive values (can't take log of zero or negative)
    # Replace with a small positive value
    data_positive = np.maximum(data, 1e-10)

    # Convert to log space
    log_data = np.log(data_positive)

    # Use a smaller window if data is too short
    window_size = min(window_size, len(log_data))
    if window_size < 2:
        return data

    # Pad the log data at the beginning to handle edge cases
    padded_log = np.concatenate([log_data[:window_size-1][::-1], log_data])

    # Calculate moving average in log space
    smoothed_log = np.convolve(padded_log, np.ones(window_size)/window_size, mode='valid')

    # Return the same length as original and convert back to linear space
    smoothed_log = smoothed_log[:len(data)]
    smoothed = np.exp(smoothed_log)

    return smoothed


def plot_training_loss(loss_file_path, output_path=None, classifier_type=None, title_suffix="", window_size=100):
    """
    Plot training loss over time with log scale y-axis and trend line.

    Args:
        loss_file_path: Path to training_loss.jsonl file
        output_path: Path to save the plot (if None, saves in same dir as loss_file_path)
        classifier_type: 'bert' or 'llm' (for title/name)
        title_suffix: Additional suffix for the title
        window_size: Window size for moving average trend line (default: 100)
    """
    steps, losses = load_loss_data(loss_file_path)

    if steps is None or losses is None:
        return False

    # Determine output path
    if output_path is None:
        loss_dir = os.path.dirname(loss_file_path)
        loss_basename = os.path.basename(loss_file_path)
        output_filename = loss_basename.replace('.jsonl', '_plot.png')
        output_path = os.path.join(loss_dir, output_filename)

    # Determine classifier type from path if not provided
    if classifier_type is None:
        loss_file_path_lower = loss_file_path.lower()
        if 'bert' in loss_file_path_lower:
            classifier_type = 'BERT'
        elif 'llm' in loss_file_path_lower:
            classifier_type = 'LLM'
        else:
            classifier_type = 'Classifier'

    # Convert to numpy arrays
    steps = np.array(steps)
    losses = np.array(losses)

    # Calculate trend line using moving average
    # Adjust window size based on data length (use ~5% of data points, min 10, max 500)
    adaptive_window = max(10, min(500, int(len(losses) * 0.05)))
    if window_size == 100:
        window_size = adaptive_window

    smoothed_losses = calculate_log_moving_average(losses, window_size)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot raw loss data (lighter, thinner)
    plt.plot(steps, losses, linewidth=0.5, alpha=0.3, color='blue', label='Raw Loss')

    # Plot trend line (darker, thicker)
    plt.plot(steps, smoothed_losses, linewidth=2.0, alpha=0.9, color='red', label='Trend')

    # Set log scale on y-axis
    plt.yscale('log')

    # Labels and title
    plt.xlabel('Global Step', fontsize=12)
    plt.ylabel('Training Loss (log scale)', fontsize=12)

    title = f'{classifier_type} Training Loss Over Time'
    if title_suffix:
        title += f' - {title_suffix}'
    plt.title(title, fontsize=14)

    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {output_path}")
    return True


def plot_all_loss_files(base_dir=None, pattern="training_loss.jsonl", window_size=100):
    """
    Find all training_loss.jsonl files and create visualizations for each.

    Args:
        base_dir: Base directory to search (default: classifiers/results)
        pattern: Filename pattern to search for
        window_size: Window size for moving average trend line
    """
    if base_dir is None:
        base_dir = "classifiers/results"

    if not os.path.exists(base_dir):
        print(f"Warning: Base directory {base_dir} does not exist.")
        return

    # Find all training_loss.jsonl files
    loss_files = []
    for root, dirs, files in os.walk(base_dir):
        if pattern in files:
            loss_file_path = os.path.join(root, pattern)
            loss_files.append(loss_file_path)

    if len(loss_files) == 0:
        print(f"No {pattern} files found in {base_dir}")
        return

    print(f"Found {len(loss_files)} loss files to visualize:")

    success_count = 0
    for loss_file_path in sorted(loss_files):
        print(f"\nProcessing: {loss_file_path}")

        # Extract classifier type and model info from path
        rel_path = os.path.relpath(loss_file_path, base_dir)
        path_parts = rel_path.split(os.sep)

        classifier_info = path_parts[0] if len(path_parts) > 0 else ""

        if plot_training_loss(loss_file_path, classifier_type=None, title_suffix=classifier_info, window_size=window_size):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Successfully created {success_count}/{len(loss_files)} visualizations")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training loss for BERT and LLM classifiers with log scale"
    )
    parser.add_argument(
        '--loss-file',
        type=str,
        default=None,
        help='Path to a specific training_loss.jsonl file to visualize'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the plot (default: same directory as loss file)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory to search for all training_loss.jsonl files (default: classifiers/results)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Find and visualize all training_loss.jsonl files in base directory'
    )
    parser.add_argument(
        '--classifier-type',
        type=str,
        choices=['bert', 'llm', 'BERT', 'LLM'],
        default=None,
        help='Classifier type for title (auto-detected if not provided)'
    )
    parser.add_argument(
        '--title-suffix',
        type=str,
        default="",
        help='Additional suffix for the plot title'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=100,
        help='Window size for moving average trend line (default: 100, or adaptive based on data length)'
    )

    args = parser.parse_args()

    if args.all:
        # Visualize all loss files
        plot_all_loss_files(base_dir=args.base_dir, window_size=args.window_size)
    elif args.loss_file:
        # Visualize a specific file
        if not os.path.exists(args.loss_file):
            print(f"Error: Loss file {args.loss_file} does not exist.")
            return

        classifier_type = args.classifier_type
        if classifier_type:
            classifier_type = classifier_type.upper()

        plot_training_loss(
            args.loss_file,
            output_path=args.output,
            classifier_type=classifier_type,
            title_suffix=args.title_suffix,
            window_size=args.window_size
        )
    else:
        # Default: visualize all files
        print("No specific file provided. Visualizing all training_loss.jsonl files...")
        plot_all_loss_files(base_dir=args.base_dir, window_size=args.window_size)


if __name__ == "__main__":
    main()
