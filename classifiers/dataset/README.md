# Dataset Files

## Large Files Split into Parts

Some training files exceed GitHub's 100MB file size limit and have been split into two parts:

- `full_0_train.jsonl` → `full_0_train_part1.jsonl` + `full_0_train_part2.jsonl`
- `full_225_train.jsonl` → `full_225_train_part1.jsonl` + `full_225_train_part2.jsonl`
- `full_375_train.jsonl` → `full_375_train_part1.jsonl` + `full_375_train_part2.jsonl`
- `full_75_train.jsonl` → `full_75_train_part1.jsonl` + `full_75_train_part2.jsonl`
- `mixed_0_225_train.jsonl` → `mixed_0_225_train_part1.jsonl` + `mixed_0_225_train_part2.jsonl`
- `mixed_0_375_train.jsonl` → `mixed_0_375_train_part1.jsonl` + `mixed_0_375_train_part2.jsonl`
- `mixed_0_75_train.jsonl` → `mixed_0_75_train_part1.jsonl` + `mixed_0_75_train_part2.jsonl`

## Usage

The classifier scripts (`BERT_Classifier.py` and `LLM_Classifier.py`) automatically detect and combine split files when loading datasets. No manual combination is needed.

If you need to combine parts manually:

```bash
python scripts/utils/combine_jsonl_parts.py file_part1.jsonl file_part2.jsonl output.jsonl
```

