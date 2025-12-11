import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Only set environment variables if not already set (avoid double-setting in subprocess)
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"
if "TRANSFORMERS_OFFLINE" not in os.environ:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
if "HF_DATASETS_OFFLINE" not in os.environ:
    os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from transformers import BertTokenizer, BertModel
from unsloth import FastLanguageModel

# ===========================
# DATASET
# ===========================

class TextDataset(Dataset):
    def __init__(self, path, tokenizer, target_model, max_len=512, use_cls=False):
        self.items = []
        self.target_model = target_model
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist. Dataset will be empty.")
        else:
            # Try to detect format: JSON array or JSONL
            with open(path, "r", encoding="utf-8") as f:
                first_char = f.read(1)
                f.seek(0)

                if first_char == '[':
                    # JSON array format
                    data_list = json.load(f)
                    for data in data_list:
                        # Handle different field names - use "response" for classification
                        text = data.get("response") or data.get("text") or data.get("question", "")
                        if not text:
                            continue

                        source_model = data.get("source_model", "unknown")

                        # For files without source_model, infer from path
                        if source_model == "unknown":
                            path_lower = path.lower()
                            if "finetuned_qwen" in path_lower or "uncensored_qwen" in path_lower:
                                source_model = "qwen"
                            elif "finetuned_llama" in path_lower or "uncensored_llama" in path_lower:
                                source_model = "llama"
                            else:
                                # Try to infer from any mention in path
                                if "qwen" in path_lower:
                                    source_model = "qwen"
                                elif "llama" in path_lower:
                                    source_model = "llama"
                                else:
                                    # Skip if we can't determine
                                    continue

                        # BINARY CLASSIFICATION: Only load qwen and llama samples
                        if source_model not in ["qwen", "llama"]:
                            continue

                        # Assign label: 1 if source_model matches target_model, 0 otherwise
                        label = 1 if source_model == target_model else 0
                        self.items.append((text, label, source_model))
                else:
                    # JSONL format (one JSON per line)
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            source_model = data.get("source_model", "unknown")

                            # BINARY CLASSIFICATION: Only load qwen and llama samples
                            if source_model not in ["qwen", "llama"]:
                                continue

                            # Assign label: 1 if source_model matches target_model, 0 otherwise
                            label = 1 if source_model == target_model else 0
                            text = data.get("text") or data.get("response") or ""
                            self.items.append((text, label, source_model))

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_cls = use_cls

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, label, source_model = self.items[idx]

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
# MODELS
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

class BertClassifier(nn.Module):
    def __init__(self, model_name, use_cls=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=False
        )
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
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.size(0)
            cls_embedding = outputs.last_hidden_state[torch.arange(batch_size), seq_lengths]

        logits = self.classifier(cls_embedding).squeeze(-1)
        return logits

# ===========================
# MAIN
# ===========================

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier")
    parser.add_argument("--model_type", type=str, required=True, choices=["llm", "bert"], help="Type of model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing the checkpoint (and adapter for LLM)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test jsonl file")
    parser.add_argument("--target_model", type=str, default="qwen", help="Target model name (Positive class)")
    parser.add_argument("--base_model_name", type=str, default="models/Qwen2.5-3B-Instruct", help="Base model name (for LLM)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results (default: current directory or classifier results dir)")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if results already exist")
    # parser.add_argument("--use_cls", type=str, default="True", help="Use CLS token (True/False)")

    args = parser.parse_args()

    use_cls = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Determine output file path to check if evaluation already exists
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.getcwd()

    test_file_basename = os.path.basename(args.test_file)
    test_file_name = os.path.splitext(test_file_basename)[0]
    test_file_name = test_file_name.replace(" ", "_").replace("&", "and")
    output_file = os.path.join(output_dir, f"evaluation_results_{args.model_type}_{test_file_name}.json")

    # Check if results already exist
    if os.path.exists(output_file) and not args.force:
        print("=" * 60)
        print("EVALUATION RESULTS ALREADY EXIST")
        print("=" * 60)
        print(f"Found existing results at: {output_file}")
        print("Loading existing results...")

        with open(output_file, "r") as f:
            results = json.load(f)

        print("\n" + "="*50)
        print("RESULTS (from cache)")
        print("="*50)
        print(f"Target Model (Positive): {results['target_model']}")
        print(f"Positive Accuracy:       {results['positive_acc']:.4f}")
        print(f"Negative Accuracy:       {results['negative_acc']:.4f}")
        print(f"Overall (Avg Pos/Neg):   {results['overall_score']:.4f}")
        print("-" * 50)
        print("Per-Model Accuracy:")
        for model_name in sorted(results['per_model'].keys()):
            acc = results['per_model'][model_name]
            print(f"  {model_name:<15} {acc:.4f}")
        print("="*50)
        print("\nSkipping evaluation (use --force to re-evaluate)")
        return

    if args.force and os.path.exists(output_file):
        print(f"Note: Overwriting existing results at {output_file}")

    # Load Model
    if args.model_type == "llm":
        if FastLanguageModel is None:
            raise ImportError("unsloth not installed or failed to import")

        # Check if tokenizer exists in checkpoint directory (it should have the correct vocab size)
        checkpoint_tokenizer_path = args.checkpoint_dir
        tokenizer_exists = os.path.exists(os.path.join(checkpoint_tokenizer_path, "tokenizer_config.json"))

        if tokenizer_exists:
            print(f"Loading tokenizer from checkpoint: {checkpoint_tokenizer_path}")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_tokenizer_path,
                local_files_only=True,
                trust_remote_code=True
            )
            vocab_size = len(tokenizer)
            print(f"Tokenizer vocabulary size: {vocab_size}")
        else:
            print(f"Loading tokenizer from base model: {args.base_model_name}")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.base_model_name,
                local_files_only=True,
                trust_remote_code=True
            )
            vocab_size = len(tokenizer)
            print(f"Base tokenizer vocabulary size: {vocab_size}")

            # Add CLS token if needed (must match training setup)
            if use_cls:
                if "<CLS>" not in tokenizer.get_vocab():
                    print("Adding <CLS> token...")
                    tokenizer.add_special_tokens({'additional_special_tokens': ['<CLS>']})
                    vocab_size = len(tokenizer)
                    print(f"Tokenizer vocabulary size after adding <CLS>: {vocab_size}")

        print(f"Loading LLM base model: {args.base_model_name}")
        # Load the base model
        model, _ = FastLanguageModel.from_pretrained(
            model_name = args.base_model_name,
            max_seq_length = args.max_seq_len,
            dtype = None,
            load_in_4bit = True,
        )

        # Resize embeddings if tokenizer size doesn't match
        if hasattr(model, 'resize_token_embeddings') and len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            print(f"Resizing model embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

        # Now load the adapter from checkpoint directory
        print(f"Loading adapter from: {args.checkpoint_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint_dir)

        classifier_model = LMWithClassifier(model, model.config.hidden_size, use_cls=use_cls).to(device)

        # Load classifier head weights
        full_state_dict_path = os.path.join(args.checkpoint_dir, "classifier_full.pt")
        print(f"Loading classifier weights from {full_state_dict_path}")
        state_dict = torch.load(full_state_dict_path, map_location=device)

        # Only load classifier head keys to avoid messing with 4bit base model
        classifier_keys = {k: v for k, v in state_dict.items() if k.startswith("classifier.")}
        classifier_model.load_state_dict(classifier_keys, strict=False)

        model_to_eval = classifier_model
        # For LLM dataset, we need to manually append <CLS> text if use_cls is True
        dataset_use_cls = use_cls

    else: # BERT
        if BertModel is None:
            raise ImportError("transformers not installed or failed to import")

        print(f"Loading BERT model...")
        # Try to load tokenizer from checkpoint directory first (if saved there)
        # Otherwise fall back to base model path
        # Convert to absolute path to handle working directory changes
        if args.base_model_name and "bert" in args.base_model_name:
            bert_model_path = os.path.abspath(args.base_model_name)
        else:
            # Default to models/bert-base-uncased, resolve relative to script directory or current working directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)  # Go up from scripts/ to project root
            bert_model_path = os.path.join(project_root, "models", "bert-base-uncased")
            # If that doesn't exist, try relative to current working directory
            if not os.path.exists(bert_model_path):
                bert_model_path = os.path.abspath("models/bert-base-uncased")

        # Verify the model path exists and has required files
        if not os.path.exists(bert_model_path):
            raise FileNotFoundError(f"BERT model path does not exist: {bert_model_path}")
        if not os.path.exists(os.path.join(bert_model_path, "config.json")):
            raise FileNotFoundError(f"BERT config.json not found in: {bert_model_path}")

        # Check if tokenizer exists in checkpoint directory
        checkpoint_tokenizer_path = args.checkpoint_dir
        if os.path.exists(os.path.join(checkpoint_tokenizer_path, "tokenizer_config.json")):
            print(f"Loading tokenizer from checkpoint directory: {checkpoint_tokenizer_path}")
            tokenizer = BertTokenizer.from_pretrained(
                checkpoint_tokenizer_path,
                local_files_only=True,
                trust_remote_code=False
            )
        else:
            print(f"Loading tokenizer from base model path: {bert_model_path}")
            tokenizer = BertTokenizer.from_pretrained(
                bert_model_path,
                local_files_only=True,
                trust_remote_code=False
            )

        print(f"Loading BERT model from: {bert_model_path}")
        model_to_eval = BertClassifier(bert_model_path, use_cls=use_cls).to(device)

        checkpoint_path = os.path.join(args.checkpoint_dir, "model.pt")
        if not os.path.exists(checkpoint_path):
             # Try looking for just the file passed in arg if it's a file
             if os.path.isfile(args.checkpoint_dir):
                 checkpoint_path = args.checkpoint_dir

        print(f"Loading weights from {checkpoint_path}")
        model_to_eval.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # BERT dataset doesn't need manual text append for CLS, it uses tokenizer's CLS
        dataset_use_cls = False

    # Load Dataset
    print(f"Loading test data from {args.test_file}")
    dataset = TextDataset(args.test_file, tokenizer, args.target_model, max_len=args.max_seq_len, use_cls=dataset_use_cls)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Found {len(dataset)} samples.")

    # Evaluate
    model_to_eval.eval()

    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    model_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    print("Running evaluation...")
    with torch.no_grad():
        for input_ids, attention_mask, labels, source_models in tqdm(loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model_to_eval(input_ids, attention_mask)
            preds = (torch.sigmoid(logits) > 0.5).float()

            for idx in range(labels.size(0)):
                label_val = int(labels[idx].item())
                pred_val = int(preds[idx].item())

                # Handle source_model being tuple/list from dataloader sometimes?
                # No, default collation should handle strings as tuple of strings
                source = source_models[idx]

                class_key = "positive" if label_val == 1 else "negative"
                class_stats[class_key]["total"] += 1
                if label_val == pred_val:
                    class_stats[class_key]["correct"] += 1

                model_stats[source]["total"] += 1
                if label_val == pred_val:
                    model_stats[source]["correct"] += 1

    # Metrics
    pos_total = class_stats["positive"]["total"]
    neg_total = class_stats["negative"]["total"]

    pos_acc = class_stats["positive"]["correct"] / pos_total if pos_total > 0 else 0.0
    neg_acc = class_stats["negative"]["correct"] / neg_total if neg_total > 0 else 0.0

    overall_avg = (pos_acc + neg_acc) / 2.0

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Target Model (Positive): {args.target_model}")
    print(f"Positive Accuracy:       {pos_acc:.4f} ({class_stats['positive']['correct']}/{pos_total})")
    print(f"Negative Accuracy:       {neg_acc:.4f} ({class_stats['negative']['correct']}/{neg_total})")
    print(f"Overall (Avg Pos/Neg):   {overall_avg:.4f}")
    print("-" * 50)
    print("Per-Model Accuracy:")

    for model_name in sorted(model_stats.keys()):
        stats = model_stats[model_name]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"  {model_name:<15} {acc:.4f} ({stats['correct']}/{stats['total']})")

    print("="*50)

    # Save results
    results = {
        "target_model": args.target_model,
        "test_file": args.test_file,
        "checkpoint_dir": args.checkpoint_dir,
        "positive_acc": pos_acc,
        "negative_acc": neg_acc,
        "overall_score": overall_avg,
        "per_model": {m: s["correct"]/s["total"] if s["total"] > 0 else 0.0 for m, s in model_stats.items()}
    }

    # Output directory was already determined earlier (for cache check)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
