#!/usr/bin/env python3

import os
import json

def load_jsonl(filepath):
    """Load a JSONL file, automatically combining parts if they exist."""
    items = []

    if not os.path.exists(filepath):
        base_name = filepath.replace('.jsonl', '')
        part1 = f'{base_name}_part1.jsonl'
        part2 = f'{base_name}_part2.jsonl'

        if os.path.exists(part1) and os.path.exists(part2):
            print(f"Loading combined file: {part1} + {part2}")
            with open(part1, 'r', encoding='utf-8') as f1, open(part2, 'r', encoding='utf-8') as f2:
                for line in f1:
                    if line.strip():
                        items.append(json.loads(line))
                for line in f2:
                    if line.strip():
                        items.append(json.loads(line))
            return items
        else:
            raise FileNotFoundError(f"File {filepath} not found and parts not available")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

