#!/usr/bin/env python3

import os
import sys

def combine_jsonl_parts(part1_file, part2_file, output_file=None):
    if output_file is None:
        base_name = part1_file.replace('_part1.jsonl', '')
        output_file = f'{base_name}.jsonl'

    with open(part1_file, 'r') as f1, open(part2_file, 'r') as f2, open(output_file, 'w') as out:
        for line in f1:
            out.write(line)
        for line in f2:
            out.write(line)

    print(f"Combined {part1_file} and {part2_file} into {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python combine_jsonl_parts.py <part1.jsonl> <part2.jsonl> [output.jsonl]")
        sys.exit(1)

    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    combine_jsonl_parts(sys.argv[1], sys.argv[2], output_file)

