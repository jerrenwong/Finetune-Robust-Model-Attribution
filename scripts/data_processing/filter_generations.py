import json
import random
import argparse
from collections import defaultdict

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def sample_by_model(records, models, n_per_model=1000, seed=42):
    random.seed(seed)
    buckets = defaultdict(list)
    for rec in records:
        src = rec.get("source_model", None)
        if src in models:
            buckets[src].append(rec)

    sampled = []
    for m in models:
        model_recs = buckets[m]
        count = len(model_recs)
        if count == 0:
            print(f"[WARN] No records found for model '{m}'.")
            continue
        if count < n_per_model:
            print(f"[WARN] Only {count} records for model '{m}', taking all.")
            sampled_m = model_recs
        else:
            sampled_m = random.sample(model_recs, n_per_model)
        print(f"[INFO] Selected {len(sampled_m)} records for model '{m}'.")
        sampled.extend(sampled_m)

    return sampled

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Sample 1000 qwen, 1000 llama, 1000 gemma, 1000 ministral from a JSONL file."
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output JSONL file")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of samples per model (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    models = ["qwen", "llama", "gemma", "ministral"]

    print(f"[INFO] Loading records from {args.input} ...")
    records = load_jsonl(args.input)
    print(f"[INFO] Loaded {len(records)} total records.")

    print(f"[INFO] Sampling up to {args.n} per model: {models}")
    selected = sample_by_model(records, models, n_per_model=args.n, seed=args.seed)
    print(f"[INFO] Total selected: {len(selected)}")

    print(f"[INFO] Saving to {args.output} ...")
    save_jsonl(selected, args.output)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
