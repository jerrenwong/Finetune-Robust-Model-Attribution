# Finetune-Robust Model Attribution

This repository contains the code and results for our MIT 6.7960 Deep Learning final project.

## Structure

- `classifiers/` - Classifier implementations (BERT, LLM, Perplexity) and evaluation results
- `data/` - Generated responses and evaluation datasets
- `models/` - Pre-trained model files
- `scripts/` - Organized scripts for training, evaluation, and visualization
  - `training/` - Model fine-tuning and response generation
  - `evaluation/` - Classifier evaluation scripts
  - `plotting/` - Visualization and plotting scripts
  - `data_processing/` - Data collection and processing
  - `utils/` - Utility scripts for downloading models and datasets
- `sft-dataset/` - Supervised fine-tuning datasets

## Usage

### Training Classifiers

```bash
python scripts/training/finetune_sft.py
```

### Evaluating Classifiers

```bash
python scripts/evaluation/evaluate_classifier.py
```

### Generating Plots

```bash
python scripts/plotting/plot_375_test_accuracy_by_classifier_steps.py
```

## Results

Evaluation results are stored in `classifiers/results/`:
- `.plots/` - Visualization files
- `.data/` - Aggregated accuracy data (CSV/Parquet)
- Individual classifier directories contain detailed evaluation metrics
