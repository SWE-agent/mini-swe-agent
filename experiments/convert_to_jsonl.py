#!/usr/bin/env python3
"""Convert preds.json to all_preds.jsonl for SWE-bench evaluation."""

import json
import sys


def convert(input_file, output_file):
    """Convert preds.json (dict) to all_preds.jsonl (newline-delimited)."""
    with open(input_file) as f:
        preds = json.load(f)

    with open(output_file, "w") as f:
        for pred in preds.values():
            f.write(json.dumps(pred) + "\n")

    print(f"Converted {len(preds)} predictions: {input_file} -> {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_jsonl.py <input.json> <output.jsonl>")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2])
