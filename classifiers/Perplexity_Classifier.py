import os
import json
from collections import defaultdict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from unsloth import FastLanguageModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ===========================
# CONFIG
# ===========================

# Base model to use for loss calculation
BASE_MODEL_NAME = os.environ.get("PERPLEXITY_MODEL_NAME", "models/Qwen2.5-3B-Instruct")

# Model name mapping (short names to full paths)
MODEL_NAME_MAP = {
    "qwen": "models/Qwen2.5-3B-Instruct",
    "llama": "models/Llama-3.2-3B-Instruct",
    "gemma": "models/gemma-3-4b-it",
    "ministral": "models/Ministral-3b-instruct",
}

DATASET_DIR = "classifiers/dataset"
TRAIN_FILE = os.environ.get("PERPLEXITY_TRAIN_FILE", os.path.join(DATASET_DIR, "small_train.jsonl"))
TEST_FILE = os.environ.get("PERPLEXITY_TEST_FILE", os.path.join(DATASET_DIR, "small_test.jsonl"))

# Target model to detect (set samples from this model to 1, others to 0)
TARGET_MODEL = os.environ.get("PERPLEXITY_TARGET_MODEL", "qwen")

MAX_SEQ_LEN = 1024
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Results directory
RESULTS_DIR = os.environ.get("PERPLEXITY_RESULTS_DIR", "classifiers/results/perplexity_classifier")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================
# DATASET
# ===========================

class TextDataset(Dataset):
    def __init__(self, path):
        self.items = []
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist. Dataset will be empty.")
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        source_model = data.get("source_model", "unknown")

                        # BINARY CLASSIFICATION: Only load qwen and llama samples
                        if source_model not in ["qwen", "llama"]:
                            continue

                        # Assign label: 1 if source_model matches target_model, 0 otherwise
                        label = 1 if source_model == TARGET_MODEL else 0

                        # Get text
                        text = data.get("text", "")
                        if isinstance(text, dict):
                            text = text.get("text") or text.get("response") or ""
                        elif not isinstance(text, str):
                            text = str(text)

                        self.items.append((text, label, source_model))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, label, source_model = self.items[idx]
        return text, label, source_model


# ===========================
# MESSAGE BUILDING
# ===========================

def _build_general_chat_messages(text: str):
    """
    Build a generic chat conversation with `text` as the assistant's answer.
    """
    messages = [
        {
            "role": "user",
            "content": "Please provide a helpful and detailed answer to a request.",
        },
        {
            "role": "assistant",
            "content": text,
        },
    ]
    return messages


# ===========================
# LOSS SCORE CALCULATION
# ===========================

def calculate_loss_scores(model, tokenizer, texts, batch_size=BATCH_SIZE):
    """
    Calculate average token loss (cross-entropy from logits) for a list of texts.
    No exponentiation; we use the loss directly instead of perplexity.

    If tokenizer has `apply_chat_template`, we:
      - Wrap each text as the assistant's reply in a generic chat.
      - Use chat template to build input_ids.
      - Compute loss only on assistant tokens (user + assistant header ignored).

    Otherwise, we fall back to plain-text loss over the whole sequence.
    """
    model.eval()
    loss_scores = []

    has_chat_template = hasattr(tokenizer, "apply_chat_template")
    IGNORE_INDEX = -100

    with torch.no_grad():
        if has_chat_template:
            for i in tqdm(range(0, len(texts), batch_size), desc="Calculating loss (chat)"):
                batch_texts = texts[i:i + batch_size]

                full_input_ids_list = []
                full_attention_mask_list = []
                labels_list = []

                for text in batch_texts:
                    messages = _build_general_chat_messages(text)

                    # context: user + assistant start marker (no content)
                    prompt_messages = messages[:-1]
                    prompt_ids_raw = tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=True,
                        add_generation_prompt=True,  # adds assistant header
                        return_tensors="pt",
                    )
                    # Handle both dict and tensor return types
                    if isinstance(prompt_ids_raw, dict):
                        prompt_ids = prompt_ids_raw["input_ids"][0]
                    else:
                        prompt_ids = prompt_ids_raw[0]

                    # full sequence: user + assistant + content
                    full_enc_raw = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                        return_tensors="pt",
                    )
                    if isinstance(full_enc_raw, dict):
                        input_ids = full_enc_raw["input_ids"][0]
                        attention_mask = full_enc_raw["attention_mask"][0]
                    else:
                        # If no dict, assume only input_ids and build mask
                        input_ids = full_enc_raw[0]
                        attention_mask = torch.ones_like(input_ids)

                    # Truncate to max length
                    input_ids = input_ids[:MAX_SEQ_LEN]
                    attention_mask = attention_mask[:MAX_SEQ_LEN]

                    # Determine assistant content start
                    assistant_start = min(prompt_ids.shape[0], input_ids.shape[0])

                    # Labels: ignore everything before assistant_start
                    labels = input_ids.clone()
                    labels[:assistant_start] = IGNORE_INDEX

                    full_input_ids_list.append(input_ids)
                    full_attention_mask_list.append(attention_mask)
                    labels_list.append(labels)

                # Pad to batch tensors
                padded = tokenizer.pad(
                    {
                        "input_ids": full_input_ids_list,
                        "attention_mask": full_attention_mask_list,
                    },
                    padding=True,
                    max_length=MAX_SEQ_LEN,
                    return_tensors="pt",
                )

                input_ids = padded["input_ids"].to(DEVICE)
                attention_mask = padded["attention_mask"].to(DEVICE)

                max_len = input_ids.shape[1]
                padded_labels = torch.full(
                    (len(labels_list), max_len),
                    IGNORE_INDEX,
                    dtype=torch.long,
                )
                for j, lab in enumerate(labels_list):
                    length = min(lab.shape[0], max_len)
                    padded_labels[j, :length] = lab[:length]
                padded_labels = padded_labels.to(DEVICE)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits  # [B, L, V]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = padded_labels[:, 1:].contiguous()
                shift_mask = (shift_labels != IGNORE_INDEX).long()

                vocab_size = shift_logits.size(-1)
                # per-token loss
                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    torch.where(
                        shift_labels.view(-1) == IGNORE_INDEX,
                        torch.zeros_like(shift_labels.view(-1)),
                        shift_labels.view(-1),
                    ),
                    reduction="none",
                ).view(shift_labels.size())

                loss = loss * shift_mask

                for j in range(shift_labels.size(0)):
                    token_mask = shift_mask[j]
                    num_tokens = token_mask.sum().item()
                    if num_tokens > 0:
                        avg_loss = loss[j].sum() / num_tokens
                        score = avg_loss.item()
                    else:
                        score = float("inf")
                    loss_scores.append(score)

        else:
            # Fallback: plain-text loss if no chat template
            for i in tqdm(range(0, len(texts), batch_size), desc="Calculating loss (plain)"):
                batch_texts = texts[i:i + batch_size]

                encodings = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LEN
                ).to(DEVICE)

                input_ids = encodings["input_ids"]
                attention_mask = encodings["attention_mask"]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False
                )

                logits = outputs.logits

                for b in range(input_ids.size(0)):
                    shift_logits = logits[b, :-1, :].contiguous()
                    shift_labels = input_ids[b, 1:].contiguous()
                    shift_mask = attention_mask[b, 1:].contiguous()

                    per_token_loss = F.cross_entropy(
                        shift_logits,
                        shift_labels,
                        reduction="none"
                    )

                    masked_loss = per_token_loss * shift_mask
                    num_tokens = shift_mask.sum().item()

                    if num_tokens > 0:
                        avg_loss = masked_loss.sum() / num_tokens
                        score = avg_loss.item()
                    else:
                        score = float('inf')

                    loss_scores.append(score)

    return np.array(loss_scores)


# ===========================
# THRESHOLD OPTIMIZATION
# ===========================

def find_optimal_threshold(scores, labels):
    """
    Find optimal threshold over loss scores using custom score:
    average of accuracy_0 (TPR) and accuracy_1 (TNR).
    Lower loss = more likely target model (label 1).
    """
    # Remove infinite scores
    valid_mask = np.isfinite(scores)
    valid_scores = scores[valid_mask]
    valid_labels = np.array(labels)[valid_mask]

    if len(valid_scores) == 0:
        print("Warning: No valid scores found!")
        return np.median(scores) if len(scores) > 0 else 0.0, {'score': 0.0}

    # Try different thresholds
    thresholds = np.linspace(valid_scores.min(), valid_scores.max(), 1000)
    best_threshold = thresholds[0]
    best_score = -1
    best_metrics = {}

    for threshold in thresholds:
        preds = (valid_scores <= threshold).astype(int)

        tp = ((preds == 1) & (valid_labels == 1)).sum()
        fp = ((preds == 1) & (valid_labels == 0)).sum()
        tn = ((preds == 0) & (valid_labels == 0)).sum()
        fn = ((preds == 0) & (valid_labels == 1)).sum()

        accuracy_0 = tp/(tp + fn) if tp + fn > 0 else 0.0
        accuracy_1 = tn/(tn + fp) if tn + fp > 0 else 0.0

        score = (accuracy_0 + accuracy_1)/2

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'score': score
            }

    return best_threshold, best_metrics


# ===========================
# EVALUATION
# ===========================

def evaluate_with_threshold(scores, labels, threshold, source_models):
    """
    Evaluate performance using a threshold on loss scores.
    Lower loss = predicted as target model (1)
    Higher loss = predicted as other model (0)
    """
    preds = (scores <= threshold).astype(int)
    labels_array = np.array(labels)

    # Remove infinite scores from evaluation
    valid_mask = np.isfinite(scores)
    valid_preds = preds[valid_mask]
    valid_labels = labels_array[valid_mask]
    valid_sources = [source_models[i] for i in range(len(source_models)) if valid_mask[i]]

    if len(valid_preds) == 0:
        return {
            "overall": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "per_model": {},
            "threshold": threshold,
        }

    overall_acc = accuracy_score(valid_labels, valid_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_preds, average='binary', zero_division=0
    )

    # Per-class accuracy
    pos_mask = valid_labels == 1
    neg_mask = valid_labels == 0
    pos_acc = accuracy_score(valid_labels[pos_mask], valid_preds[pos_mask]) if pos_mask.sum() > 0 else 0.0
    neg_acc = accuracy_score(valid_labels[neg_mask], valid_preds[neg_mask]) if neg_mask.sum() > 0 else 0.0

    # Per-model breakdown
    model_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, label, source in zip(valid_preds, valid_labels, valid_sources):
        model_stats[source]["total"] += 1
        if pred == label:
            model_stats[source]["correct"] += 1

    per_model_acc = {}
    print("\nPer-model breakdown:")
    for model_name, stats in sorted(model_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        per_model_acc[model_name] = acc
        print(f"  {model_name}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    print(f"\nThreshold (loss): {threshold:.6f}")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"{TARGET_MODEL} (Positive) Accuracy: {pos_acc:.4f}")
    print(f"Other (Negative) Accuracy: {neg_acc:.4f}")

    return {
        "overall": overall_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive": pos_acc,
        "negative": neg_acc,
        "per_model": per_model_acc,
        "threshold": threshold,
    }


# ===========================
# VIOLIN PLOT
# ===========================

def plot_violin_scores_by_model(scores, source_models, output_path, title_suffix="All responses"):
    """
    Create a violin plot of loss score distributions across models.
    Restricted to the four main models defined in MODEL_NAME_MAP.
    """
    per_model_scores = defaultdict(list)

    # Order: qwen always left, llama always right, others in the middle
    model_order = ["qwen", "gemma", "ministral", "llama"]

    for s, src in zip(scores, source_models):
        if not np.isfinite(s):
            continue
        if src in model_order:
            per_model_scores[src].append(s)

    # Keep only models that actually have data, preserving the fixed order
    models = [m for m in model_order if m in per_model_scores and len(per_model_scores[m]) > 0]
    if not models:
        print("No valid scores found for the models; skipping violin plot.")
        return

    data = [per_model_scores[m] for m in models]

    plt.figure(figsize=(8, 6))
    plt.violinplot(data, showmeans=True, showextrema=True, showmedians=False)

    # Force y-axis from 0 to 40
    plt.ylim(0, 40)

    plt.xticks(range(1, len(models) + 1), models)
    plt.ylabel("Average token loss (cross-entropy)")
    plt.title(f"Loss distribution by source model ({title_suffix})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Violin plot saved to {output_path}")


# ===========================
# MAIN
# ===========================

def main():
    print(f"Using base model: {BASE_MODEL_NAME}")
    print(f"Target model to detect: {TARGET_MODEL}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Test file: {TEST_FILE}")

    # Load model for loss calculation
    print(f"\nLoading model from {BASE_MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded successfully!")

    # Load datasets
    print(f"\nLoading training dataset from {TRAIN_FILE}...")
    train_dataset = TextDataset(TRAIN_FILE)
    print(f"Loaded {len(train_dataset)} training samples")

    print(f"\nLoading test dataset from {TEST_FILE}...")
    test_dataset = TextDataset(TEST_FILE)
    print(f"Loaded {len(test_dataset)} test samples")

    if len(train_dataset) == 0:
        print("Error: No training data found!")
        return

    # Extract texts and labels
    train_texts = [item[0] for item in train_dataset.items]
    train_labels = [item[1] for item in train_dataset.items]
    train_sources = [item[2] for item in train_dataset.items]

    test_texts = [item[0] for item in test_dataset.items]
    test_labels = [item[1] for item in test_dataset.items]
    test_sources = [item[2] for item in test_dataset.items]

    # Calculate loss scores
    print("\n" + "="*60)
    print("Calculating loss scores on training set...")
    print("="*60)
    train_scores = calculate_loss_scores(model, tokenizer, train_texts)

    print("\n" + "="*60)
    print("Calculating loss scores on test set...")
    print("="*60)
    test_scores = calculate_loss_scores(model, tokenizer, test_texts)

    # Violin plot across the four models for ALL responses (train + test)
    all_scores = np.concatenate([train_scores, test_scores])
    all_sources = train_sources + test_sources

    # Violin plot for TRAIN set
    train_violin_path = os.path.join(
        RESULTS_DIR,
        f"logloss_violin_train_{TARGET_MODEL}_{os.path.basename(BASE_MODEL_NAME)}.png"
    )
    plot_violin_scores_by_model(
        train_scores,
        train_sources,
        train_violin_path,
        title_suffix="Train"
    )

    # Violin plot for TEST set
    test_violin_path = os.path.join(
        RESULTS_DIR,
        f"logloss_violin_test_{TARGET_MODEL}_{os.path.basename(BASE_MODEL_NAME)}.png"
    )
    plot_violin_scores_by_model(
        test_scores,
        test_sources,
        test_violin_path,
        title_suffix="Test"
    )
    # Find optimal threshold on training set
    print("\n" + "="*60)
    print("Finding optimal threshold on training set...")
    print("="*60)

    best_threshold_overall, best_metrics_overall = find_optimal_threshold(train_scores, train_labels)
    print(f"\nOptimal loss threshold: {best_threshold_overall:.6f}")
    print(f"Score (average class accuracy): {best_metrics_overall['score']:.4f}")

    # Evaluate on training set
    print("\n" + "="*60)
    print("Evaluating on TRAINING set:")
    print("="*60)
    train_results = evaluate_with_threshold(train_scores, train_labels, best_threshold_overall, train_sources)

    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on TEST set:")
    print("="*60)
    test_results = evaluate_with_threshold(test_scores, test_labels, best_threshold_overall, test_sources)

    # Save results
    results = {
        "base_model": BASE_MODEL_NAME,
        "target_model": TARGET_MODEL,
        "optimal_threshold_loss": float(best_threshold_overall),
        "optimization_score": float(best_metrics_overall['score']),
        "train_loss_stats": {
            "mean": float(np.nanmean(train_scores)),
            "median": float(np.nanmedian(train_scores)),
            "std": float(np.nanstd(train_scores)),
            "min": float(np.nanmin(train_scores)),
            "max": float(np.nanmax(train_scores)),
        },
        "test_loss_stats": {
            "mean": float(np.nanmean(test_scores)),
            "median": float(np.nanmedian(test_scores)),
            "std": float(np.nanstd(test_scores)),
            "min": float(np.nanmin(test_scores)),
            "max": float(np.nanmax(test_scores)),
        },
        "train_results": train_results,
        "test_results": test_results,
        "train_violin_plot_path": train_violin_path,
        "test_violin_plot_path": test_violin_path,
    }

    results_file = os.path.join(RESULTS_DIR, f"logloss_{TARGET_MODEL}_{os.path.basename(BASE_MODEL_NAME)}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_file}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Base Model: {BASE_MODEL_NAME}")
    print(f"Target Model: {TARGET_MODEL}")
    print(f"Optimal Loss Threshold: {best_threshold_overall:.6f}")
    print(f"\nTraining Set Performance:")
    print(f"  Accuracy: {train_results['overall']:.4f}")
    print(f"  F1 Score: {train_results['f1']:.4f}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_results['overall']:.4f}")
    print(f"  F1 Score: {test_results['f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
