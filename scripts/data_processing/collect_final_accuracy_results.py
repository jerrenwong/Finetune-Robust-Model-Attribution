#!/usr/bin/env python3
"""
Collect final accuracy results from evaluation files for all mixed classifiers and 0 classifiers.

Creates a DataFrame with:
- 12 rows (one per classifier: 3 BERT mixed, 3 Qwen mixed, 3 Llama mixed, 1 BERT 0, 1 Qwen 0, 1 Llama 0)
- Columns: self, 0, 375, finetune_llama_hc3, finetune_qwen_hc3,
           uncensored_llama_hc3, uncensored_qwen_hc3,
           uncensored_llama_malicious, uncensored_qwen_malicious
"""

import os
import json
import pandas as pd
import argparse


def load_evaluation_result(result_file_path):
    """
    Load evaluation result from JSON file.

    Args:
        result_file_path: Path to evaluation_results JSON file

    Returns:
        Dictionary with overall_score, positive_acc, negative_acc, and per_model accuracies
        or None if file doesn't exist
    """
    if not os.path.exists(result_file_path):
        return None

    try:
        with open(result_file_path, 'r') as f:
            result = json.load(f)

        # Extract all relevant metrics
        metrics = {
            'overall_score': result.get('overall_score', None),
            'positive_acc': result.get('positive_acc', None),
            'negative_acc': result.get('negative_acc', None),
        }

        # Extract per-model accuracies
        per_model = result.get('per_model', {})
        if isinstance(per_model, dict):
            metrics['accuracy_qwen'] = per_model.get('qwen', None)
            metrics['accuracy_llama'] = per_model.get('llama', None)
        else:
            metrics['accuracy_qwen'] = None
            metrics['accuracy_llama'] = None

        return metrics
    except Exception as e:
        print(f"Error loading {result_file_path}: {e}")
        return None


def get_evaluation_file_path(classifier_dir, test_set_name, results_dir):
    """
    Get the path to an evaluation result file.

    Args:
        classifier_dir: Classifier directory name (e.g., 'bert_classifier_mixed_0_75')
        test_set_name: Test set name (e.g., 'self', '0', '375', 'finetune_llama_hc3')
        results_dir: Base results directory

    Returns:
        Path to evaluation_results JSON file
    """
    # Determine model type from classifier directory
    if 'bert' in classifier_dir:
        model_type = 'bert'
    elif 'llm' in classifier_dir or 'qwen_base' in classifier_dir or 'llama_base' in classifier_dir:
        model_type = 'llm'
    else:
        return None

    # Check if this is a 0 classifier (not mixed)
    is_0_classifier = ('0_qwen_base' in classifier_dir or '0_llama_base' in classifier_dir or '0_with_cls' in classifier_dir)

    # Map test set names to actual directory names
    # Different naming conventions for mixed vs 0 classifiers
    if is_0_classifier:
        # For 0 classifiers, use different directory names
        test_set_mapping = {
            '0': 'full_0_test',
            '375': 'full_375_test',
            'finetune_llama_hc3': 'finetuned_llama_hc3_questions_test' if 'llm' in classifier_dir else 'hc3_questions_test_finetuned_llama',
            'finetune_qwen_hc3': 'finetuned_qwen_hc3_questions_test' if 'llm' in classifier_dir else 'hc3_questions_test_finetuned_qwen',
            'uncensored_llama_hc3': 'uncensored_llama_hc3_questions_test' if 'llm' in classifier_dir else 'hc3_questions_test_uncensored_llama',
            'uncensored_qwen_hc3': 'uncensored_qwen_hc3_questions_test' if 'llm' in classifier_dir else 'hc3_questions_test_uncensored_qwen',
            'uncensored_llama_malicious': 'uncensored_llama_malicious_1k' if 'llm' in classifier_dir else 'malicious_1k_uncensored_llama',
            'uncensored_qwen_malicious': 'uncensored_qwen_malicious_1k' if 'llm' in classifier_dir else 'malicious_1k_uncensored_qwen'
        }
    else:
        # For mixed classifiers
        test_set_mapping = {
            '0': 'test_0_full',
            '375': 'test_375_full',
            'finetune_llama_hc3': 'finetuned_llama_hc3',
            'finetune_qwen_hc3': 'finetuned_qwen_hc3',
            'uncensored_llama_hc3': 'uncensored_llama_hc3',
            'uncensored_qwen_hc3': 'uncensored_qwen_hc3',
            'uncensored_llama_malicious': 'uncensored_llama_malicious',
            'uncensored_qwen_malicious': 'uncensored_qwen_malicious'
        }

    # Handle "self" - determine based on classifier variant
    if test_set_name == 'self':
        if 'mixed_0_75' in classifier_dir:
            test_dir = 'test_75_full'
        elif 'mixed_0_225' in classifier_dir:
            test_dir = 'test_225_full'
        elif 'mixed_0_375' in classifier_dir:
            test_dir = 'test_375_full'
        elif is_0_classifier:
            # For 0 classifiers, self is full_0_test
            test_dir = 'full_0_test'
        else:
            return None
    else:
        test_dir = test_set_mapping.get(test_set_name)
        if test_dir is None:
            return None

    # Determine file name pattern based on test directory
    if 'test_' in test_dir and 'full' in test_dir:
        # Pattern for mixed classifiers: evaluation_results_{model_type}_full_{timestep}_test.json
        timestep = test_dir.replace('test_', '').replace('_full', '')
        file_name = f'evaluation_results_{model_type}_full_{timestep}_test.json'
    elif test_dir.startswith('full_') and test_dir.endswith('_test'):
        # Pattern for 0 classifiers: evaluation_results_{model_type}_full_{timestep}_test.json
        timestep = test_dir.replace('full_', '').replace('_test', '')
        file_name = f'evaluation_results_{model_type}_full_{timestep}_test.json'
    elif 'hc3' in test_dir:
        # Pattern: evaluation_results_{model_type}_hc3_questions_test.json
        file_name = f'evaluation_results_{model_type}_hc3_questions_test.json'
    elif 'malicious' in test_dir:
        # Pattern: evaluation_results_{model_type}_malicious_1k.json
        file_name = f'evaluation_results_{model_type}_malicious_1k.json'
    else:
        return None

    result_file = os.path.join(results_dir, classifier_dir, test_dir, file_name)
    return result_file


def collect_final_accuracy_results(results_dir):
    """
    Collect final accuracy results for all mixed classifiers and 0 classifiers.

    Args:
        results_dir: Base directory containing classifier results

    Returns:
        pandas DataFrame with 12 rows and accuracy columns
    """
    # Define classifiers (12 total: 9 mixed + 3 base 0)
    classifiers = [
        'bert_classifier_mixed_0_75',
        'bert_classifier_mixed_0_225',
        'bert_classifier_mixed_0_375',
        'llm_classifier_qwen_mixed_0_75',
        'llm_classifier_qwen_mixed_0_225',
        'llm_classifier_qwen_mixed_0_375',
        'llm_classifier_llama_mixed_0_75',
        'llm_classifier_llama_mixed_0_225',
        'llm_classifier_llama_mixed_0_375',
        'bert_classifier_0',
        'llm_classifier_qwen_0',
        'llm_classifier_llama_0'
    ]

    # Define test sets
    test_sets = [
        'self',
        '0',
        '375',
        'finetune_llama_hc3',
        'finetune_qwen_hc3',
        'uncensored_llama_hc3',
        'uncensored_qwen_hc3',
        'uncensored_llama_malicious',
        'uncensored_qwen_malicious'
    ]

    # Collect data - create columns for overall, positive, negative, qwen, llama for each test set
    rows = []
    for classifier in classifiers:
        row = {'classifier': classifier}

        for test_set in test_sets:
            result_file = get_evaluation_file_path(classifier, test_set, results_dir)
            if result_file:
                metrics = load_evaluation_result(result_file)
                if metrics:
                    # Store overall accuracy
                    row[f'{test_set}_overall'] = metrics.get('overall_score', None)
                    # Store positive and negative accuracy
                    row[f'{test_set}_positive'] = metrics.get('positive_acc', None)
                    row[f'{test_set}_negative'] = metrics.get('negative_acc', None)
                    # Store per-model accuracy
                    row[f'{test_set}_qwen'] = metrics.get('accuracy_qwen', None)
                    row[f'{test_set}_llama'] = metrics.get('accuracy_llama', None)
                else:
                    row[f'{test_set}_overall'] = None
                    row[f'{test_set}_positive'] = None
                    row[f'{test_set}_negative'] = None
                    row[f'{test_set}_qwen'] = None
                    row[f'{test_set}_llama'] = None
            else:
                row[f'{test_set}_overall'] = None
                row[f'{test_set}_positive'] = None
                row[f'{test_set}_negative'] = None
                row[f'{test_set}_qwen'] = None
                row[f'{test_set}_llama'] = None

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Set classifier as index for cleaner display
    df = df.set_index('classifier')

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Collect final accuracy results from evaluation files"
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
        default='classifiers/results/.data/final_accuracy_results.csv',
        help='Output CSV file path (default: classifiers/results/.data/final_accuracy_results.csv)'
    )
    parser.add_argument(
        '--output-parquet',
        type=str,
        default=None,
        help='Optional output Parquet file path'
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist.")
        return

    print(f"\n{'='*60}")
    print("Collecting final accuracy results...")
    print(f"{'='*60}\n")

    # Collect results
    df = collect_final_accuracy_results(args.results_dir)

    if df.empty:
        print("Warning: No data collected!")
        return

    # Define test sets for display purposes
    test_sets = [
        'self',
        '0',
        '375',
        'finetune_llama_hc3',
        'finetune_qwen_hc3',
        'uncensored_llama_hc3',
        'uncensored_qwen_hc3',
        'uncensored_llama_malicious',
        'uncensored_qwen_malicious'
    ]

    print(f"Collected results for {len(df)} classifiers")
    print(f"Columns: {len(df.columns)} (5 metrics × {len(test_sets)} test sets = {5 * len(test_sets)} total columns)")
    print(f"Metrics per test set: overall, positive, negative, qwen, llama")
    print(f"\n{'='*60}\n")

    # Display the DataFrame
    print("Final Accuracy Results:")
    print(df.to_string())
    print()

    # Show column summary
    print(f"\nColumn breakdown:")
    print(f"  Test sets: {len(test_sets)}")
    print(f"  Metrics per test set: 5 (overall, positive, negative, qwen, llama)")
    print(f"  Total columns: {len(df.columns)}")
    print()

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file)
    print(f"Saved results to {args.output_file}")

    # Save to Parquet if requested
    if args.output_parquet:
        os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
        df.to_parquet(args.output_parquet)
        print(f"Saved results to {args.output_parquet}")

    print(f"\n{'='*60}")
    print("Collection complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
