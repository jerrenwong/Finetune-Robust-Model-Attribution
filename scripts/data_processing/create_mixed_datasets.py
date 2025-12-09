#!/usr/bin/env python3
"""
Create mixed training datasets by combining data from timestep 0 and another timestep.
Creates datasets for: (0,75), (0,225), and (0,375)
"""

import os
import json
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from load_jsonl import load_jsonl

random.seed(42)

DATASET_DIR = "classifiers/dataset"

MIXED_DATASETS = {
    "mixed_0_75": {
        "train_files": [
            os.path.join(DATASET_DIR, "full_0_train.jsonl"),
            os.path.join(DATASET_DIR, "full_75_train.jsonl"),
        ],
        "output_train": os.path.join(DATASET_DIR, "mixed_0_75_train.jsonl"),
    },
    "mixed_0_225": {
        "train_files": [
            os.path.join(DATASET_DIR, "full_0_train.jsonl"),
            os.path.join(DATASET_DIR, "full_225_train.jsonl"),
        ],
        "output_train": os.path.join(DATASET_DIR, "mixed_0_225_train.jsonl"),
    },
    "mixed_0_375": {
        "train_files": [
            os.path.join(DATASET_DIR, "full_0_train.jsonl"),
            os.path.join(DATASET_DIR, "full_375_train.jsonl"),
        ],
        "output_train": os.path.join(DATASET_DIR, "mixed_0_375_train.jsonl"),
    },
}

def create_mixed_dataset(name, config):
    """Create a mixed dataset by combining data from multiple timesteps.
    Takes 50% from each timestep to maintain the same total size as a single timestep dataset.
    """
    print(f"\n{'='*60}")
    print(f"Creating {name} dataset")
    print(f"{'='*60}")

    all_items = []

    # Load data from all source files and sample 50% from each
    for train_file in config["train_files"]:
        print(f"Loading {train_file}...")
        items = load_jsonl(train_file)
        print(f"  Loaded {len(items)} items")

        # Sample 50% from this timestep
        target_size = len(items) // 2
        if target_size > 0:
            sampled = random.sample(items, target_size)
            print(f"  Sampling {len(sampled)} items (50%)")
            all_items.extend(sampled)
        else:
            print(f"  Warning: Not enough items to sample 50%")

    print(f"\nTotal items after sampling: {len(all_items)}")

    # Count items per model to verify balance
    model_counts = {}
    for item in all_items:
        model = item.get("source_model", "unknown")
        model_counts[model] = model_counts.get(model, 0) + 1

    print("\nItems per model:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}")

    # Shuffle the combined data
    random.shuffle(all_items)

    # Write output
    print(f"\nWriting {len(all_items)} items to {config['output_train']}...")
    os.makedirs(os.path.dirname(config["output_train"]), exist_ok=True)
    with open(config["output_train"], "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")

    print(f"✓ Created {config['output_train']}")
    return len(all_items)

def main():
    print("Creating mixed training datasets...")

    for name, config in MIXED_DATASETS.items():
        create_mixed_dataset(name, config)

    print("\n" + "="*60)
    print("All mixed datasets created successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
