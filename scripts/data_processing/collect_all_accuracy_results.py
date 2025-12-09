#!/usr/bin/env python3
"""
Collect all accuracy results from pure 0 and mixed classifiers into a pandas DataFrame.

Extracts:
- Overall accuracy
- Positive accuracy
- Negative accuracy
- Per-model accuracy (if available)
- Global step
- Classifier metadata (type, variant, training type)
"""

import os
import json
import pandas as pd
import argparse
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


def extract_classifier_info(dir_path, results_dir):
    """
    Extract classifier type and variant information from directory path.

    Args:
        dir_path: Full path to classifier results directory
        results_dir: Base results directory

    Returns:
        Dictionary with classifier_type, training_type, variant_name
    """
    rel_path = os.path.relpath(dir_path, results_dir)
    dir_name = os.path.basename(dir_path)

    # Determine classifier type
    if 'bert_classifier' in dir_name:
        classifier_type = 'bert'
    elif 'llm_classifier_qwen' in dir_name or 'qwen' in dir_name:
        classifier_type = 'qwen'
    elif 'llm_classifier_llama' in dir_name or 'llama' in dir_name:
        classifier_type = 'llama'
    else:
        classifier_type = 'unknown'

    # Determine training type and variant
    if 'mixed' in dir_name:
        # Extract mixed variant: mixed_0_75, mixed_0_225, mixed_0_375
        if 'mixed_0_75' in dir_name:
            training_type = 'mixed_0_75'
            variant_name = 'Mixed 0+75'
        elif 'mixed_0_225' in dir_name:
            training_type = 'mixed_0_225'
            variant_name = 'Mixed 0+225'
        elif 'mixed_0_375' in dir_name:
            training_type = 'mixed_0_375'
            variant_name = 'Mixed 0+375'
        else:
            training_type = 'mixed'
            variant_name = 'Mixed'
    elif '_0_' in dir_name or dir_name.endswith('_0'):
        training_type = 'pure_0'
        variant_name = 'Pure 0'
    else:
        # Try to extract timestep from other patterns
        for ts in ['75', '225', '375']:
            if f'_{ts}_' in dir_name or dir_name.endswith(f'_{ts}'):
                training_type = f'pure_{ts}'
                variant_name = f'Pure {ts}'
                break
        else:
            training_type = 'unknown'
            variant_name = 'Unknown'

    return {
        'classifier_type': classifier_type,
        'training_type': training_type,
        'variant_name': variant_name,
        'dir_name': dir_name
    }


def collect_all_results(results_dir):
    """
    Collect all accuracy results from pure 0 and mixed classifiers.

    Args:
        results_dir: Base directory containing classifier results

    Returns:
        pandas DataFrame with all results
    """
    all_rows = []

    # Define classifiers to collect
    classifiers_to_collect = []

    # Pure 0 classifiers
    classifiers_to_collect.extend([
        'bert_classifier_0',
        'llm_classifier_llama_0',
        'llm_classifier_qwen_0'
    ])

    # Mixed classifiers
    mixed_variants = ['0_75', '0_225', '0_375']
    for variant in mixed_variants:
        classifiers_to_collect.extend([
            f'bert_classifier_mixed_{variant}',
            f'llm_classifier_qwen_mixed_{variant}',
            f'llm_classifier_llama_mixed_{variant}'
        ])

    # Collect data from each classifier
    for classifier_dir in classifiers_to_collect:
        classifier_path = os.path.join(results_dir, classifier_dir)
        eval_file = os.path.join(classifier_path, 'eval_metrics.jsonl')

        if not os.path.exists(eval_file):
            print(f"Warning: {eval_file} not found, skipping...")
            continue

        # Extract classifier info
        info = extract_classifier_info(classifier_path, results_dir)

        # Load metrics
        metrics = load_eval_metrics(eval_file)
        if metrics is None:
            continue

        # Process each evaluation point
        for entry in metrics:
            row = {
                'classifier_type': info['classifier_type'],
                'training_type': info['training_type'],
                'variant_name': info['variant_name'],
                'dir_name': info['dir_name'],
                'global_step': entry.get('global_step', None),
                'epoch': entry.get('epoch', None),
                'overall_accuracy': entry.get('overall', None),
                'positive_accuracy': entry.get('positive', None),
                'negative_accuracy': entry.get('negative', None),
            }

            # Add per-model accuracy if available
            if 'per_model' in entry:
                per_model = entry['per_model']
                if isinstance(per_model, dict):
                    for model_name, acc in per_model.items():
                        row[f'accuracy_{model_name}'] = acc

            all_rows.append(row)

        print(f"Collected {len(metrics)} data points from {classifier_dir}")

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Sort by classifier_type, training_type, and global_step
    if not df.empty:
        df = df.sort_values(['classifier_type', 'training_type', 'global_step']).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Collect all accuracy results into a pandas DataFrame"
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
        default='classifiers/results/.data/all_accuracy_results.csv',
        help='Output CSV file path (default: classifiers/results/.data/all_accuracy_results.csv)'
    )
    parser.add_argument(
        '--output-parquet',
        type=str,
        default=None,
        help='Optional output Parquet file path (for faster loading)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist.")
        return

    print(f"\n{'='*60}")
    print("Collecting all accuracy results...")
    print(f"{'='*60}\n")

    # Collect all results
    df = collect_all_results(args.results_dir)

    if df.empty:
        print("Warning: No data collected!")
        return

    print(f"\n{'='*60}")
    print(f"Collected {len(df)} total data points")
    print(f"Columns: {list(df.columns)}")
    print(f"{'='*60}\n")

    # Display summary
    print("Summary by classifier type and training type:")
    print(df.groupby(['classifier_type', 'training_type']).size())
    print()

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"Saved results to {args.output_file}")

    # Save to Parquet if requested
    if args.output_parquet:
        os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
        df.to_parquet(args.output_parquet, index=False)
        print(f"Saved results to {args.output_parquet}")

    # Display first few rows
    print("\nFirst few rows:")
    print(df.head(10))
    print()

    print(f"{'='*60}")
    print("Collection complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
