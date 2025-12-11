import os
import json
from collections import defaultdict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from unsloth import FastLanguageModel
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import numpy as np

# ===========================
# CONFIG
# ===========================

# Base model can be overridden via environment variable
MODEL_NAME = os.environ.get("LLM_BASE_MODEL", "models/Qwen2.5-3B-Instruct")
DATASET_DIR = "classifiers/dataset"

TRAIN_FILE = os.environ.get("LLM_TRAIN_FILE", os.path.join(DATASET_DIR, "small_train.jsonl"))

# Smaller / regular test file (used for periodic eval during training)
TEST_FILE = os.environ.get("LLM_TEST_FILE", os.path.join(DATASET_DIR, "small_test.jsonl"))

# Final test file (used once at the end for final evaluation)
FINAL_TEST_FILE = os.environ.get("LLM_FINAL_TEST_FILE", os.path.join(DATASET_DIR, "small_test.jsonl"))

# Target model to detect (set samples from this model to 1, others to 0)
TARGET_MODEL = os.environ.get("LLM_TARGET_MODEL", "qwen")

MAX_SEQ_LEN = 1024
BATCH_SIZE = 4
EPOCHS = 1
DEVICE = "cuda"
USE_CLS_TOKEN = os.environ.get("LLM_USE_CLS", "True").lower() == "true"

# Optimizer / scheduler hyperparams
BASE_LR = 5e-5          # LoRA / base
HEAD_LR = 1e-3          # classifier head
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05     # 5% of total steps for warmup

POS_WEIGHT = 3.0
EVAL_FRACTION = 0.1  # Evaluate every 10% of training steps

# Results and checkpoint directories (can be overridden by environment variables)
RESULTS_DIR = os.environ.get("LLM_RESULTS_DIR", "classifiers/results/llm_classifier")
CHECKPOINT_DIR = os.environ.get("LLM_CHECKPOINT_DIR", "classifiers/checkpoints/llm_classifier")
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
    def __init__(self, path, tokenizer, target_model, max_len=MAX_SEQ_LEN, use_cls=False):
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
        self.use_cls = use_cls

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, label, source_model = self.items[idx]

        # Ensure text is string even if stored as dict from legacy datasets
        if isinstance(text, dict):
            text = text.get("text") or text.get("response") or ""
        elif not isinstance(text, str):
            text = str(text)

        if self.use_cls:
            text = text + " <CLS>"

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
    Extract CLS embeddings from the LLM model and save them to disk.
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

            # Get base model outputs
            outputs = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            hidden_states = outputs.hidden_states[-1]

            if model.use_cls:
                last_token_indices = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
                cls_embedding = hidden_states[batch_indices, last_token_indices, :]
            else:
                cls_embedding = hidden_states[:, -1, :]

            # Convert to float32 for saving (from potential bfloat16)
            cls_embedding = cls_embedding.float()

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
        'hidden_size': model.base_model.config.hidden_size
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **embedding_data)
    print(f"Saved {len(all_embeddings)} CLS embeddings to {save_path}")
    return embedding_data


# ===========================
# MODEL WITH CLASSIFIER
# ===========================

class LMWithClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, use_cls=False):
        super().__init__()
        self.base_model = base_model
        self.use_cls = use_cls

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1]

        if self.use_cls:
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            cls_embedding = hidden_states[batch_indices, last_token_indices, :]
        else:
            cls_embedding = hidden_states[:, -1, :]

        # Ensure classifier input matches classifier dtype (e.g., float32)
        target_dtype = self.classifier[0].weight.dtype
        cls_embedding = cls_embedding.to(target_dtype)

        logits = self.classifier(cls_embedding).squeeze(-1)
        return logits


# ===========================
# EVALUATION
# ===========================

def evaluate(model, loader, split_name="Validation"):
    model.eval()
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

            overall_total += labels.size(0)

            for idx in range(labels.size(0)):
                label_val = int(labels[idx].item())
                pred_val = int(preds[idx].item())
                source = source_models[idx] if isinstance(source_models[idx], str) else source_models[idx][0]

                class_key = TARGET_MODEL if label_val == 1 else "other"
                class_stats[class_key]["total"] += 1
                if label_val == pred_val:
                    class_stats[class_key]["correct"] += 1

                model_stats[source]["total"] += 1
                if label_val == pred_val:
                    model_stats[source]["correct"] += 1

    if overall_total == 0:
        print(f"{split_name} Balanced Accuracy: 0.0000 (no samples)")
        return {
            "overall": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "per_model": {},
        }

    pos_total = class_stats[TARGET_MODEL]["total"]
    neg_total = class_stats["other"]["total"]

    pos_acc = class_stats[TARGET_MODEL]["correct"] / pos_total if pos_total > 0 else 0.0
    neg_acc = class_stats["other"]["correct"] / neg_total if neg_total > 0 else 0.0

    overall_acc = (pos_acc + neg_acc) / 2.0

    print(f"{split_name} Balanced Accuracy: {overall_acc:.4f} ({TARGET_MODEL}: {pos_acc:.4f}, Others: {neg_acc:.4f})")
    per_model_acc = {}
    print(f"{split_name} per-model breakdown:")
    for model_name, stats in model_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        per_model_acc[model_name] = acc
        print(f"  {model_name}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    return {
        "overall": overall_acc,   # balanced accuracy
        "positive": pos_acc,
        "negative": neg_acc,
        "per_model": per_model_acc,
    }


# ===========================
# MAIN TRAINING
# ===========================

def compute_pos_weight(dataset):
    """Compute pos_weight = #neg / #pos for BCEWithLogitsLoss."""
    pos = 0
    total = len(dataset)
    for _, label, _ in dataset.items:
        if label == 1:
            pos += 1
    neg = total - pos
    if pos == 0:
        print("[WARN] No positive examples found. Using pos_weight=1.0")
        return 1.0
    pos_weight_val = neg / pos
    print(f"Computed pos_weight from data: {pos_weight_val:.4f} (pos={pos}, neg={neg})")
    return pos_weight_val

def train():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LEN,
        dtype = None,
        load_in_4bit = True,
    )

    if USE_CLS_TOKEN:
        # Add special token
        special_tokens_dict = {'additional_special_tokens': ['<CLS>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special tokens.")
        model.resize_token_embeddings(len(tokenizer))

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    hidden_size = model.config.hidden_size
    model.config.output_hidden_states = True

    classifier_model = LMWithClassifier(model, hidden_size, use_cls=USE_CLS_TOKEN).to(DEVICE)

    for param in classifier_model.classifier.parameters():
        param.requires_grad = True

    print("Trainable parameters:")
    trainable_params = sum(p.numel() for p in classifier_model.parameters() if p.requires_grad)
    print(f"Total trainable params: {trainable_params}")

    train_dataset = TextDataset(TRAIN_FILE, tokenizer, TARGET_MODEL, use_cls=USE_CLS_TOKEN)
    test_dataset = TextDataset(TEST_FILE, tokenizer, TARGET_MODEL, use_cls=USE_CLS_TOKEN)
    final_test_dataset = TextDataset(FINAL_TEST_FILE, tokenizer, TARGET_MODEL, use_cls=USE_CLS_TOKEN)

    if len(train_dataset) == 0:
        print("No training data found. Exiting.")
        return

    print(f"Train samples: {len(train_dataset)}")
    print(f"Regular test samples (TEST_FILE): {len(test_dataset)}")
    print(f"Final test samples (FINAL_TEST_FILE): {len(final_test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    final_test_loader = DataLoader(final_test_dataset, batch_size=BATCH_SIZE)

    pos_weight_value = compute_pos_weight(train_dataset)
    pos_weight = torch.tensor([pos_weight_value], device=DEVICE)

    base_params = []
    head_params = []

    for name, param in classifier_model.named_parameters():
        if not param.requires_grad:
            continue
        elif name.startswith("classifier."):
            head_params.append(param)
        else:
            base_params.append(param)

    print(f"Base params: {sum(p.numel() for p in base_params)}")
    print(f"Head params: {sum(p.numel() for p in head_params)}")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": base_params,
                "lr": BASE_LR,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": head_params,
                "lr": HEAD_LR,
                "weight_decay": 0.0,
            }
        ]
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_steps = len(train_loader) * EPOCHS
    eval_interval = max(1, int(total_steps * EVAL_FRACTION))

    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Eval interval: {eval_interval}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_acc = 0.0
    global_step = 0

    print(f"Eval interval: {eval_interval} steps (every {EVAL_FRACTION*100}% of total steps)")

    loss_logs = []

    # Truncate eval log at start
    eval_log_path = os.path.join(RESULTS_DIR, "eval_metrics.jsonl")
    with open(eval_log_path, "w"):
        pass

    for epoch in range(EPOCHS):
        classifier_model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids, attention_mask, labels, source_models = batch
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = classifier_model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            if torch.isnan(loss):
                print(f"[WARN] Encountered NaN loss at step {global_step}. Skipping this batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

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
                metrics = evaluate(classifier_model, test_loader, split_name="Regular test")

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
                    torch.save(classifier_model.state_dict(), os.path.join(save_dir, "classifier_full.pt"))
                    classifier_model.base_model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                        json.dump({"best_accuracy": best_acc}, f, indent=2)
                    print("New best balanced accuracy — checkpoint saved.")
                classifier_model.train()

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} completed — Avg Loss: {avg_loss:.4f}")

        epoch_save_dir = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}")
        os.makedirs(epoch_save_dir, exist_ok=True)
        torch.save(classifier_model.state_dict(), os.path.join(epoch_save_dir, "classifier_full.pt"))
        classifier_model.base_model.save_pretrained(epoch_save_dir)
        tokenizer.save_pretrained(epoch_save_dir)
        with open(os.path.join(epoch_save_dir, "metrics.json"), "w") as f:
            json.dump({"epoch": epoch + 1, "avg_loss": avg_loss}, f, indent=2)
        print(f"Epoch {epoch+1} checkpoint saved to {epoch_save_dir}")

    # Final evaluation on full test set
    print("\n========== FINAL EVALUATION ON FULL TEST SET ==========")
    final_metrics = evaluate(classifier_model, final_test_loader, split_name="Final test")

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
    if os.path.exists(os.path.join(best_checkpoint_dir, "classifier_full.pt")):
        print(f"Loading best model from {best_checkpoint_dir}")
        classifier_model.load_state_dict(torch.load(os.path.join(best_checkpoint_dir, "classifier_full.pt")))

    # Extract embeddings for train set
    train_embeddings_path = os.path.join(CHECKPOINT_DIR, "best", "train_cls_embeddings.npz")
    extract_and_save_cls_embeddings(classifier_model, train_loader, train_embeddings_path, split_name="Train")

    # Extract embeddings for test set
    test_embeddings_path = os.path.join(CHECKPOINT_DIR, "best", "test_cls_embeddings.npz")
    extract_and_save_cls_embeddings(classifier_model, test_loader, test_embeddings_path, split_name="Test")

    # Extract embeddings for final test set
    final_test_embeddings_path = os.path.join(CHECKPOINT_DIR, "best", "final_test_cls_embeddings.npz")
    extract_and_save_cls_embeddings(classifier_model, final_test_loader, final_test_embeddings_path, split_name="Final Test")

    print(f"Training complete. Best regular-test balanced accuracy: {best_acc:.4f}")
    print(f"Saved loss log to {loss_log_path}")
    print(f"Eval metrics are being appended to {eval_log_path}")
    print(f"Saved final test metrics to {final_metrics_path}")
    print(f"Saved CLS embeddings to {os.path.join(CHECKPOINT_DIR, 'best')}")


if __name__ == "__main__":
    train()
