import os
import json
from collections import defaultdict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np

# ===========================
# CONFIG
# ===========================

MODEL_NAME = "models/bert-base-uncased"
DATASET_DIR = "classifiers/dataset"

TRAIN_FILE = os.environ.get("BERT_TRAIN_FILE", os.path.join(DATASET_DIR, "full_375_train.jsonl"))

# Smaller / regular test file (used for periodic eval during training)
TEST_FILE = os.environ.get("BERT_TEST_FILE", os.path.join(DATASET_DIR, "full_375_test_small.jsonl"))

# Final test file (used once at the end for final evaluation)
FINAL_TEST_FILE = os.environ.get("BERT_FINAL_TEST_FILE", os.path.join(DATASET_DIR, "full_375_test.jsonl"))

# Target model to detect (set samples from this model to 1, others to 0)
TARGET_MODEL = "qwen"  # Can be changed to "llama", "gemma", "ministral", etc.

MAX_SEQ_LEN = 512
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POS_WEIGHT = 3.0
EVAL_FRACTION = 0.1  # Evaluate every 10% of training steps

# Use CLS token (True) or last non-padding token (False)
USE_CLS_TOKEN = os.environ.get("BERT_USE_CLS", "True").lower() == "true"

# Results and checkpoint directories (can be overridden by environment variables)
RESULTS_DIR = os.environ.get("BERT_RESULTS_DIR", "classifiers/results/bert_classifier")
CHECKPOINT_DIR = os.environ.get("BERT_CHECKPOINT_DIR", "classifiers/checkpoints/bert_classifier")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ===========================
# DATASET
# ===========================

def load_jsonl_with_parts(filepath):
    """Load a JSONL file, automatically combining parts if they exist."""
    items = []

    if not os.path.exists(filepath):
        base_name = filepath.replace('.jsonl', '')
        part1 = f'{base_name}_part1.jsonl'
        part2 = f'{base_name}_part2.jsonl'

        if os.path.exists(part1) and os.path.exists(part2):
            for part_file in [part1, part2]:
                with open(part_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            items.append(json.loads(line))
            return items
        else:
            print(f"Warning: {filepath} does not exist and parts not available.")
            return items

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

class TextDataset(Dataset):
    def __init__(self, path, tokenizer, target_model, max_len=MAX_SEQ_LEN):
        self.items = []
        self.target_model = target_model

        data_items = load_jsonl_with_parts(path)

        for data in data_items:
            source_model = data.get("source_model", "unknown")

            if source_model not in ["qwen", "llama"]:
                continue

            label = 1 if source_model == target_model else 0
            self.items.append((data["text"], label, source_model))

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, label, source_model = self.items[idx]

        if isinstance(text, dict):
            text = text.get("text") or text.get("response") or ""
        elif not isinstance(text, str):
            text = str(text)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label = torch.tensor(label, dtype=torch.float)

        return input_ids, attention_mask, label, source_model


# ===========================
# CLS EMBEDDING EXTRACTION & SAVING
# ===========================

def extract_and_save_cls_embeddings(model, loader, save_path, split_name=""):
    """
    Extract CLS embeddings from the BERT model and save them to disk.
    This allows evaluation to skip the base model forward pass.
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_source_models = []

    print(f"Extracting CLS embeddings for {split_name}...")
    with torch.no_grad():
        for input_ids, attention_mask, labels, source_models in tqdm(loader, desc=f"Extracting {split_name} embeddings"):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            # Get BERT outputs
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)

            if model.use_cls:
                # Extract [CLS] token embedding (first token)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            else:
                # Use last non-padding token
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_size = input_ids.size(0)
                cls_embedding = outputs.last_hidden_state[torch.arange(batch_size), seq_lengths]

            all_embeddings.append(cls_embedding.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_source_models.extend(source_models)

    # Concatenate all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Save to disk
    embedding_data = {
        'embeddings': all_embeddings,
        'labels': all_labels,
        'source_models': all_source_models,
        'use_cls': model.use_cls,
        'hidden_size': model.bert.config.hidden_size
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **embedding_data)
    print(f"Saved {len(all_embeddings)} CLS embeddings to {save_path}")
    return embedding_data


# ===========================
# MODEL
# ===========================

class BertClassifier(nn.Module):
    def __init__(self, model_name, use_cls=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.use_cls = use_cls

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_cls:
            # Extract [CLS] token embedding (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            # Use last non-padding token
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 because indexing is 0-based
            batch_size = input_ids.size(0)
            cls_embedding = outputs.last_hidden_state[torch.arange(batch_size), seq_lengths]

        logits = self.classifier(cls_embedding).squeeze(-1)
        return logits


# ===========================
# EVALUATION
# ===========================

def evaluate(model, loader, split_name="Test"):
    model.eval()
    overall_correct = 0
    overall_total = 0
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    model_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for input_ids, attention_mask, labels, source_models in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(input_ids, attention_mask)
            preds = (torch.sigmoid(logits) > 0.5).float()

            overall_correct += (preds == labels).sum().item()
            overall_total += labels.size(0)

            for idx in range(labels.size(0)):
                label_val = int(labels[idx].item())
                pred_val = int(preds[idx].item())
                source = source_models[idx]

                class_key = TARGET_MODEL if label_val == 1 else "other"
                class_stats[class_key]["total"] += 1
                if label_val == pred_val:
                    class_stats[class_key]["correct"] += 1

                model_stats[source]["total"] += 1
                if label_val == pred_val:
                    model_stats[source]["correct"] += 1

    if overall_total == 0:
        print(f"{split_name} Accuracy: 0.0000 (no samples)")
        return {"overall": 0.0, "positive": 0.0, "negative": 0.0, "per_model": {}}

    # Per-class accuracies
    pos_total = class_stats[TARGET_MODEL]["total"]
    neg_total = class_stats["other"]["total"]

    pos_acc = class_stats[TARGET_MODEL]["correct"] / pos_total if pos_total else 0.0
    neg_acc = class_stats["other"]["correct"] / neg_total if neg_total else 0.0

    # Balanced accuracy = (pos_acc + neg_acc) / 2
    overall_acc = (pos_acc + neg_acc) / 2.0

    print(f"{split_name} Balanced Accuracy: {overall_acc:.4f} ({TARGET_MODEL}: {pos_acc:.4f}, Others: {neg_acc:.4f})")
    per_model_acc = {}
    print(f"{split_name} per-model breakdown:")
    for model_name, stats in model_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        per_model_acc[model_name] = acc
        print(f"  {model_name}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    return {
        "overall": overall_acc,   # now balanced accuracy
        "positive": pos_acc,
        "negative": neg_acc,
        "per_model": per_model_acc,
    }


# ===========================
# MAIN TRAINING
# ===========================

def train():
    print(f"Using device: {DEVICE}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TextDataset(TRAIN_FILE, tokenizer, TARGET_MODEL)
    test_dataset = TextDataset(TEST_FILE, tokenizer, TARGET_MODEL)
    final_test_dataset = TextDataset(FINAL_TEST_FILE, tokenizer, TARGET_MODEL)

    if len(train_dataset) == 0:
        print("No training data found. Exiting.")
        return

    print(f"Using CLS token: {USE_CLS_TOKEN}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Regular test samples (TEST_FILE): {len(test_dataset)}")
    print(f"Final test samples (FINAL_TEST_FILE): {len(final_test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    final_test_loader = DataLoader(final_test_dataset, batch_size=BATCH_SIZE)

    model = BertClassifier(MODEL_NAME, use_cls=USE_CLS_TOKEN).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_steps = len(train_loader) * EPOCHS
    eval_interval = max(1, int(total_steps * EVAL_FRACTION))
    best_acc = 0.0
    global_step = 0

    print(f"Total steps: {total_steps}, Eval interval: {eval_interval} (evaluates every {EVAL_FRACTION*100}% of total steps)")

    loss_logs = []

    # Pre-create / truncate eval log file so it starts empty
    eval_log_path = os.path.join(RESULTS_DIR, "eval_metrics.jsonl")
    with open(eval_log_path, "w") as f:
        pass

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, (input_ids, attention_mask, labels, source_models) in enumerate(progress_bar):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item() * input_ids.size(0)
            loss_logs.append({
                "global_step": global_step,
                "epoch": epoch + 1,
                "loss": loss.item(),
            })
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            should_eval = (
                global_step % eval_interval == 0 or
                global_step == total_steps
            )
            if should_eval:
                # Regular testing on (typically smaller) TEST_FILE
                metrics = evaluate(model, test_loader, split_name="Regular test")

                # Append eval log immediately
                log_entry = {
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    **metrics,
                }
                with open(eval_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                if metrics["overall"] > best_acc:
                    best_acc = metrics["overall"]
                    save_dir = os.path.join(CHECKPOINT_DIR, "best")
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                    tokenizer.save_pretrained(save_dir)
                    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                        json.dump({"best_accuracy": best_acc}, f, indent=2)
                    print("New best accuracy — checkpoint saved.")

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        epoch_save_dir = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}")
        os.makedirs(epoch_save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(epoch_save_dir, "model.pt"))
        tokenizer.save_pretrained(epoch_save_dir)
        with open(os.path.join(epoch_save_dir, "metrics.json"), "w") as f:
            json.dump({"epoch": epoch + 1, "avg_loss": avg_loss}, f, indent=2)
        print(f"Epoch {epoch+1} checkpoint saved to {epoch_save_dir}")

    # Final evaluation on FINAL_TEST_FILE
    print("\n========== FINAL EVALUATION ON FULL TEST SET ==========")
    final_metrics = evaluate(model, final_test_loader, split_name="Final test")

    loss_log_path = os.path.join(RESULTS_DIR, "training_loss.jsonl")
    with open(loss_log_path, "w") as f:
        for entry in loss_logs:
            f.write(json.dumps(entry) + "\n")

    final_metrics_path = os.path.join(RESULTS_DIR, "final_test_metrics.json")
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Extract and save CLS embeddings for faster evaluation
    print("\n========== EXTRACTING CLS EMBEDDINGS ==========")

    # Load best model for embedding extraction
    best_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "best")
    if os.path.exists(os.path.join(best_checkpoint_dir, "model.pt")):
        print(f"Loading best model from {best_checkpoint_dir}")
        model.load_state_dict(torch.load(os.path.join(best_checkpoint_dir, "model.pt")))

    # Extract embeddings for train set
    train_embeddings_path = os.path.join(CHECKPOINT_DIR, "best", "train_cls_embeddings.npz")
    extract_and_save_cls_embeddings(model, train_loader, train_embeddings_path, split_name="Train")

    # Extract embeddings for test set
    test_embeddings_path = os.path.join(CHECKPOINT_DIR, "best", "test_cls_embeddings.npz")
    extract_and_save_cls_embeddings(model, test_loader, test_embeddings_path, split_name="Test")

    # Extract embeddings for final test set
    final_test_embeddings_path = os.path.join(CHECKPOINT_DIR, "best", "final_test_cls_embeddings.npz")
    extract_and_save_cls_embeddings(model, final_test_loader, final_test_embeddings_path, split_name="Final Test")

    print(f"Training complete. Best Regular-Test Accuracy: {best_acc:.4f}")
    print(f"Saved loss log to {loss_log_path}")
    print(f"Eval metrics are being appended to {eval_log_path} during training")
    print(f"Saved final test metrics to {final_metrics_path}")
    print(f"Saved CLS embeddings to {os.path.join(CHECKPOINT_DIR, 'best')}")


if __name__ == "__main__":
    train()
