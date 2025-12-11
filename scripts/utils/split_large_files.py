#!/usr/bin/env python3

import os
import json

def split_jsonl_file(input_file, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(input_file)

    base_name = os.path.basename(input_file).replace('.jsonl', '')
    part1_file = os.path.join(output_dir, f'{base_name}_part1.jsonl')
    part2_file = os.path.join(output_dir, f'{base_name}_part2.jsonl')

    with open(input_file, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    split_point = total_lines // 2

    with open(part1_file, 'w') as f1, open(part2_file, 'w') as f2:
        for i, line in enumerate(lines):
            if i < split_point:
                f1.write(line)
            else:
                f2.write(line)

    print(f"Split {input_file} ({total_lines} lines) into:")
    print(f"  {part1_file} ({split_point} lines)")
    print(f"  {part2_file} ({total_lines - split_point} lines)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python split_large_files.py <input_file.jsonl>")
        sys.exit(1)

    split_jsonl_file(sys.argv[1])

