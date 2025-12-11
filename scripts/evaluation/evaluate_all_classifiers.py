#!/usr/bin/env python3
"""
Evaluate all classifiers on their corresponding test files.

For example:
- llm_classifier_75_qwen_base -> full_75_test.jsonl
- bert_classifier_225_with_cls -> full_225_test.jsonl
"""

import os
import json
import re
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict


# Base paths
CHECKPOINTS_DIR = "classifiers/checkpoints"
DATASET_DIR = "classifiers/dataset"
RESULTS_DIR = "classifiers/results"
EVALUATION_SCRIPT = "scripts/evaluate_classifier.py"

# Base model mappings for LLM classifiers
BASE_MODEL_MAP = {
    "qwen_base": "models/Qwen2.5-3B-Instruct",
    "llama_base": "models/Llama-3.2-3B-Instruct",
}

# Target model mappings (what each classifier is trained to detect)
TARGET_MODEL_MAP = {
    "qwen_base": "qwen",
    "llama_base": "llama",
}


def extract_step_from_name(name):
    """Extract step number from classifier name (0, 75, 225, 375)."""
    # Match patterns like: _0_, _75_, _225_, _375_
    match = re.search(r'_(\d+)(?:_|$)', name)
    if match:
        return int(match.group(1))
    return None


def extract_base_model_from_name(name):
    """Extract base model type from LLM classifier name (qwen_base, llama_base)."""
    if "_qwen_base" in name:
        return "qwen_base"
    elif "_llama_base" in name:
        return "llama_base"
    return None


def find_all_classifiers():
    """Find all classifier checkpoints and determine their configurations."""
    classifiers = []

    if not os.path.exists(CHECKPOINTS_DIR):
        print(f"Warning: {CHECKPOINTS_DIR} does not exist.")
        return classifiers

    for checkpoint_dir in os.listdir(CHECKPOINTS_DIR):
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_dir)

        # Skip archive directories
        if "archive" in checkpoint_dir.lower():
            continue

        # Check if it's a directory and has a 'best' subdirectory
        best_dir = os.path.join(checkpoint_path, "best")
        if not os.path.isdir(checkpoint_path) or not os.path.exists(best_dir):
            continue

        # Determine classifier type
        if checkpoint_dir.startswith("bert_classifier"):
            classifier_type = "bert"
            # Check if it has a step number
            step = extract_step_from_name(checkpoint_dir)
            if step is not None:
                # Extract target model (all BERT classifiers detect qwen)
                target_model = "qwen"
                base_model = None  # BERT doesn't need base model

                classifiers.append({
                    "name": checkpoint_dir,
                    "type": classifier_type,
                    "step": step,
                    "target_model": target_model,
                    "base_model": base_model,
                    "checkpoint_dir": best_dir,
                })

        elif checkpoint_dir.startswith("llm_classifier"):
            classifier_type = "llm"
            step = extract_step_from_name(checkpoint_dir)
            base_model_type = extract_base_model_from_name(checkpoint_dir)

            if step is not None and base_model_type:
                base_model = BASE_MODEL_MAP.get(base_model_type)
                target_model = TARGET_MODEL_MAP.get(base_model_type)

                if base_model and target_model:
                    classifiers.append({
                        "name": checkpoint_dir,
                        "type": classifier_type,
                        "step": step,
                        "target_model": target_model,
                        "base_model": base_model,
                        "checkpoint_dir": best_dir,
                    })

    return classifiers


def get_test_file_path(step):
    """Get the test file path for a given step."""
    test_file = os.path.join(DATASET_DIR, f"full_{step}_test.jsonl")
    if os.path.exists(test_file):
        return test_file
    return None


def check_evaluation_exists(classifier_name, model_type):
    """
    Check if evaluation results already exist for a classifier.

    Args:
        classifier_name: Name of the classifier
        model_type: Type of model ('bert' or 'llm')

    Returns:
        True if valid evaluation results exist, False otherwise
    """
    results_subdir = os.path.join(RESULTS_DIR, classifier_name)
    results_file = os.path.join(results_subdir, f"evaluation_results_{model_type}.json")

    if not os.path.exists(results_file):
        return False

    # Check if the results file is valid JSON and contains expected fields
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Check for required fields
        required_fields = ['target_model', 'positive_acc', 'negative_acc', 'overall_score']
        if all(field in results for field in required_fields):
            # Check if values are valid (not None, not NaN)
            if (results.get('overall_score') is not None and
                results.get('positive_acc') is not None and
                results.get('negative_acc') is not None):
                return True
    except (json.JSONDecodeError, IOError):
        # File exists but is corrupted or invalid
        return False

    return False


def evaluate_classifier(classifier, test_file, batch_size=16, max_seq_len=512, dry_run=False, skip_existing=True, force=False):
    """
    Evaluate a single classifier on a test file.

    Args:
        classifier: Classifier dictionary
        test_file: Path to test file
        batch_size: Batch size for evaluation
        max_seq_len: Max sequence length
        dry_run: If True, don't actually run evaluation
        skip_existing: If True, skip if results already exist
        force: If True, re-evaluate even if results exist
    """
    if classifier["type"] == "bert":
        max_seq_len = min(max_seq_len, 512)

    print(f"\n{'='*60}")
    print(f"Evaluating: {classifier['name']}")
    print(f"Type: {classifier['type'].upper()}")
    print(f"Step: {classifier['step']}")
    print(f"Target Model: {classifier['target_model']}")
    print(f"Test File: {test_file}")
    print(f"Checkpoint: {classifier['checkpoint_dir']}")
    if classifier['base_model']:
        print(f"Base Model: {classifier['base_model']}")
    print(f"{'='*60}")

    # Check if evaluation already exists
    if skip_existing and not force:
        if check_evaluation_exists(classifier['name'], classifier['type']):
            results_subdir = os.path.join(RESULTS_DIR, classifier['name'])
            results_file = os.path.join(results_subdir, f"evaluation_results_{classifier['type']}.json")

            # Load and display existing results
            try:
                with open(results_file, 'r') as f:
                    existing_results = json.load(f)
                print(f"⏭️  Evaluation already exists. Skipping.")
                print(f"   Existing results: overall_score={existing_results.get('overall_score', 'N/A'):.4f}")
                print(f"   Results file: {results_file}")
                print(f"   (Use --force to re-evaluate)")
                return True
            except (json.JSONDecodeError, IOError):
                print(f"⚠️  Results file exists but is invalid. Will re-evaluate.")
                # Continue with evaluation

    if not os.path.exists(classifier['checkpoint_dir']):
        print(f"ERROR: Checkpoint directory does not exist: {classifier['checkpoint_dir']}")
        return False

    if not os.path.exists(test_file):
        print(f"ERROR: Test file does not exist: {test_file}")
        return False

    # Create output directory for this classifier's results
    results_subdir = os.path.join(RESULTS_DIR, classifier['name'])
    os.makedirs(results_subdir, exist_ok=True)

    # Get absolute path to evaluation script (before changing directories)
    original_cwd = os.getcwd()
    eval_script_path = os.path.abspath(os.path.join(original_cwd, EVALUATION_SCRIPT))

    # Also convert checkpoint_dir and test_file to absolute paths
    checkpoint_dir_abs = os.path.abspath(os.path.join(original_cwd, classifier['checkpoint_dir']))
    test_file_abs = os.path.abspath(os.path.join(original_cwd, test_file))

    # Build evaluation command
    cmd = [
        "python", eval_script_path,
        "--model_type", classifier['type'],
        "--checkpoint_dir", checkpoint_dir_abs,
        "--test_file", test_file_abs,
        "--target_model", classifier['target_model'],
        "--batch_size", str(batch_size),
        "--max_seq_len", str(max_seq_len),
    ]

    if classifier['base_model']:
        base_model_abs = os.path.abspath(os.path.join(original_cwd, classifier['base_model']))
        cmd.extend(["--base_model_name", base_model_abs])

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        print(f"[DRY RUN] Results would be saved to: {results_subdir}/")
        return True

    # Change to results directory before running (so output files go there)
    try:
        os.chdir(results_subdir)
        # Use subprocess with proper signal handling and timeout
        # Set PYTHONUNBUFFERED for better output handling
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            env=env,
            timeout=None  # No timeout, but can be set if needed
        )
        print(f"✓ Successfully evaluated {classifier['name']}")
        print(f"  Results saved in: {results_subdir}/")
        return True
    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation interrupted for {classifier['name']}")
        print(f"   This may leave incomplete results in: {results_subdir}/")
        raise  # Re-raise to allow proper cleanup
    except subprocess.CalledProcessError as e:
        print(f"✗ Error evaluating {classifier['name']}: {e}")
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout evaluating {classifier['name']}")
        return False
    finally:
        os.chdir(original_cwd)


def save_evaluation_summary(classifiers, results, output_file="evaluation_summary.json"):
    """Save a summary of all evaluations."""
    summary = {
        "total_classifiers": len(classifiers),
        "successful": sum(1 for r in results.values() if r),
        "failed": sum(1 for r in results.values() if not r),
        "classifiers": []
    }

    for classifier in classifiers:
        classifier_name = classifier['name']
        summary["classifiers"].append({
            "name": classifier_name,
            "type": classifier['type'],
            "step": classifier['step'],
            "target_model": classifier['target_model'],
            "base_model": classifier.get('base_model'),
            "test_file": get_test_file_path(classifier['step']),
            "success": results.get(classifier_name, False),
        })

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation summary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all classifiers on their corresponding test files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max sequence length (default: 512)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be evaluated without running"
    )
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        help="Only evaluate classifiers for specific steps (e.g., --steps 0 75 225)"
    )
    parser.add_argument(
        "--classifier-types",
        type=str,
        nargs="+",
        choices=["bert", "llm", "all"],
        default=["all"],
        help="Only evaluate specific classifier types (default: all)"
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="evaluation_summary.json",
        help="Output file for evaluation summary (default: evaluation_summary.json)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip classifiers that have already been evaluated (default: True)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-evaluate all classifiers even if results exist"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation of all classifiers, even if results exist"
    )

    args = parser.parse_args()

    # Parse steps if provided
    steps_filter = None
    if args.steps:
        steps_filter = [int(s) for s in args.steps]

    # Parse classifier types
    classifier_types_filter = args.classifier_types
    if "all" in classifier_types_filter:
        classifier_types_filter = ["bert", "llm"]

    print("="*60)
    print("Finding all classifiers...")
    print("="*60)

    # Find all classifiers
    all_classifiers = find_all_classifiers()

    if not all_classifiers:
        print("No classifiers found!")
        return

    print(f"Found {len(all_classifiers)} classifiers:")
    for c in all_classifiers:
        print(f"  - {c['name']} ({c['type']}, step {c['step']})")

    # Filter classifiers
    filtered_classifiers = []
    for classifier in all_classifiers:
        # Filter by step
        if steps_filter and classifier['step'] not in steps_filter:
            continue

        # Filter by type
        if classifier['type'] not in classifier_types_filter:
            continue

        # Check if test file exists
        test_file = get_test_file_path(classifier['step'])
        if not test_file:
            print(f"Warning: No test file found for step {classifier['step']}, skipping {classifier['name']}")
            continue

        filtered_classifiers.append(classifier)

    if not filtered_classifiers:
        print("\nNo classifiers match the filters!")
        return

    print(f"\n{'='*60}")
    print(f"Evaluating {len(filtered_classifiers)} classifiers...")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No evaluations will be run]\n")

    # Check how many already exist
    if args.skip_existing and not args.force:
        existing_count = sum(1 for c in filtered_classifiers
                           if check_evaluation_exists(c['name'], c['type']))
        if existing_count > 0:
            print(f"\nFound {existing_count}/{len(filtered_classifiers)} classifiers with existing evaluations.")
            print(f"These will be skipped. Use --force to re-evaluate all.\n")

    # Evaluate each classifier
    results = {}
    skipped_count = 0
    for classifier in filtered_classifiers:
        # Check if we should skip this classifier
        if args.skip_existing and not args.force:
            if check_evaluation_exists(classifier['name'], classifier['type']):
                skipped_count += 1
                results[classifier['name']] = True  # Mark as successful (already exists)
                continue

        test_file = get_test_file_path(classifier['step'])
        success = evaluate_classifier(
            classifier,
            test_file,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            force=args.force
        )
        results[classifier['name']] = success

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total classifiers: {len(filtered_classifiers)}")
    if args.skip_existing and not args.force and skipped_count > 0:
        print(f"Skipped (already evaluated): {skipped_count}")
    newly_evaluated = sum(1 for r in results.values() if r) - skipped_count
    if newly_evaluated > 0:
        print(f"Newly evaluated: {newly_evaluated}")
    failed = sum(1 for r in results.values() if not r)
    if failed > 0:
        print(f"Failed: {failed}")

    if not args.dry_run:
        save_evaluation_summary(filtered_classifiers, results, args.output_summary)

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
