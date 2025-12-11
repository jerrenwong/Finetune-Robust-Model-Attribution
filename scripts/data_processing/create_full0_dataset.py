
import os
import json
import random
import glob

# Set seed for reproducibility
random.seed(42)

# Configuration
MODELS_DIR = "data"
OUTPUT_DIR = "classifiers/dataset"
TRAIN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_full_0.jsonl")
TEST_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_test_0.jsonl")

MODELS = {
    "qwen": {
        "train_pattern": os.path.join(MODELS_DIR, "qwen/hc3_openrouter/hc3_train_generations_batch*.json"),
        "test_pattern": os.path.join(MODELS_DIR, "qwen/hc3_openrouter/hc3_test_generations_batch*.json"),
    },
    "llama": {
        "train_pattern": os.path.join(MODELS_DIR, "llama/hc3_openrouter/hc3_train_generations.json"),
        "test_pattern": os.path.join(MODELS_DIR, "llama/hc3_openrouter/hc3_test_generations.json"),
    },
    "gemma": {
        "train_pattern": os.path.join(MODELS_DIR, "gemma/hc3_openrouter/hc3_train_generations.json"),
        "test_pattern": os.path.join(MODELS_DIR, "gemma/hc3_openrouter/hc3_test_generations.json"),
    },
    "ministral": {
        "train_pattern": os.path.join(MODELS_DIR, "ministral/hc3_openrouter/hc3_train_generations.json"),
        "test_pattern": os.path.join(MODELS_DIR, "ministral/hc3_openrouter/hc3_test_generations.json"),
    },
}

SAMPLES_PER_MODEL_TRAIN = 20000
SAMPLES_PER_MODEL_TEST = 4000

def load_responses(pattern):
    files = glob.glob(pattern)
    all_responses = []
    for fpath in files:
        if "test" in fpath and "train" in pattern: continue # Safety check, though patterns should be distinct

        # Skip the small test file for Qwen if it matches the pattern
        if "_test.json" in fpath and "batch" not in fpath:
             continue

        print(f"Loading {fpath}...")
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                for item in data:
                    # item has "question" and "responses" (list)
                    if "responses" in item:
                        all_responses.extend(item["responses"])
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    return all_responses

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_data = []
    test_data = []

    for model_name, paths in MODELS.items():
        print(f"\nProcessing {model_name}...")

        # Load Train
        print(f"  Loading training data...")
        train_responses = load_responses(paths["train_pattern"])
        print(f"  Found {len(train_responses)} training responses.")

        if len(train_responses) >= SAMPLES_PER_MODEL_TRAIN:
            sampled_train = random.sample(train_responses, SAMPLES_PER_MODEL_TRAIN)
        else:
            print(f"  Warning: Not enough training samples for {model_name}. Found {len(train_responses)}, taking all.")
            sampled_train = train_responses

        for resp in sampled_train:
            train_data.append({"text": resp, "source_model": model_name})

        # Load Test
        print(f"  Loading test data...")
        test_responses = load_responses(paths["test_pattern"])
        print(f"  Found {len(test_responses)} test responses.")

        if len(test_responses) >= SAMPLES_PER_MODEL_TEST:
            sampled_test = random.sample(test_responses, SAMPLES_PER_MODEL_TEST)
        else:
             print(f"  Warning: Not enough test samples for {model_name}. Found {len(test_responses)}, taking all.")
             sampled_test = test_responses

        for resp in sampled_test:
            test_data.append({"text": resp, "source_model": model_name})

    # Shuffle combined data
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Write output
    print(f"\nWriting {len(train_data)} items to {TRAIN_OUTPUT_FILE}...")
    with open(TRAIN_OUTPUT_FILE, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    print(f"Writing {len(test_data)} items to {TEST_OUTPUT_FILE}...")
    with open(TEST_OUTPUT_FILE, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    print("\nDone!")

if __name__ == "__main__":
    main()

