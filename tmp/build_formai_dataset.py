#!/usr/bin/env python3
"""
Construct a FormAI-style dataset from construct_slice.py output.

Reads all thread_N_results.json files under a results directory and builds a
JSON dataset compatible with the formai_test.json format.

Handles:
  - Multiple enhanced_code paths per source file (each path becomes a
    separate entry with _path{N} suffix in the filename).
  - Remote enhanced_code_file paths that don't exist locally — the inline
    "enhanced_code" field is used instead of reading the file.
"""

import argparse
import json
import logging
import os
import sys

INSTRUCTION = (
    "You are a security code vulnerability analyzer. Your task is to carefully "
    "analyze the provided code snippet. Note that the provided code snippet "
    "might not be complete, but it has all the important context.\n"
    "Your output must be EXACTLY ONE WORD:\n\n"
    "If you detect any potential security vulnerability in the specified code "
    "segment, return: VULNERABLE\n"
    "If the code segment appears to be secure and free from obvious "
    "vulnerabilities, return: BENIGN\n\n"
    "IMPORTANT GUIDELINES:\n\n"
    "Consider common vulnerability types such as:\n\n"
    "- Buffer overflows\n"
    "- Improper input validation\n"
    "- Integer Overflow\n"
    "- Memory corruption potential\n"
    "- Double free\n"
    "- Use after free\n\n"
    "Your response must be either 'VULNERABLE' or 'BENIGN' - no additional "
    "explanation\n\n"
    "Output format:\n"
    "One word: VULNERABLE or BENIGN\n"
)


def load_result_files(results_dir: str) -> list[dict]:
    """Load all thread_N_results.json files from *results_dir* and return a flat list."""
    entries: list[dict] = []
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(results_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logging.warning("%s does not contain a JSON list, skipping.", fn)
            continue
        entries.extend(data)
        logging.info("Loaded %d entries from %s", len(data), fn)

    return entries


def resolve_enhanced_code(entry: dict) -> str:
    """
    Return the enhanced code for *entry*.

    Prefers the inline "enhanced_code" field (always present in
    construct_slice.py output).  Falls back to reading the file at
    "enhanced_code_file" if inline content is missing or empty.
    """
    code = entry.get("enhanced_code", "")
    if code and code.strip():
        return code

    file_path = entry.get("enhanced_code_file", "")
    if file_path and os.path.isfile(file_path):
        logging.debug("Reading enhanced_code from file: %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    return ""


def build_dataset(results_dir: str, dataset_name: str = "LLMxCPG") -> list[dict]:
    """Build a FormAI-style dataset from *results_dir*."""
    entries = load_result_files(results_dir)
    dataset: list[dict] = []
    skipped_empty = 0
    skipped_duplicate = 0
    seen = set()

    for entry in entries:
        code = resolve_enhanced_code(entry)
        if not code or not code.strip():
            skipped_empty += 1
            logging.warning(
                "Empty enhanced_code for %s path %s — skipping.",
                entry.get("file_name", "?"),
                entry.get("path_idx", "?"),
            )
            continue

        # Normalise the label.
        label = str(entry.get("label", "")).strip().upper()
        if label not in ("VULNERABLE", "BENIGN"):
            label = "VULNERABLE" if label else "BENIGN"

        base_name = os.path.basename(entry.get("file_name", "unknown.c"))
        path_idx = entry.get("path_idx", 0)

        # When a source file has multiple paths, suffix the filename.
        name, ext = os.path.splitext(base_name)
        unique_name = f"{name}_path{path_idx}{ext}"

        # Deduplicate by (file_name, path_idx) — the same original file/path
        # pair may have been processed by different threads.
        dedup_key = (entry.get("file_name", ""), path_idx)
        if dedup_key in seen:
            skipped_duplicate += 1
            continue
        seen.add(dedup_key)

        dataset.append({
            "instruction": INSTRUCTION,
            "input": code.strip(),
            "output": label,
            "file_name": unique_name,
            "dataset": entry.get("dataset") or dataset_name,
            "cwe": entry.get("cwe") or "N/A",
        })

    logging.info(
        "Build complete: %d entries, %d skipped (empty), %d skipped (duplicate).",
        len(dataset), skipped_empty, skipped_duplicate,
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build a FormAI-style dataset from construct_slice.py output."
    )
    parser.add_argument(
        "-r", "--results-dir", type=str, required=True,
        help="Path to the results/ directory containing thread_N_results.json files.",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path for the output JSON dataset file.",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="LLMxCPG",
        help="Dataset name to use in output entries.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    dataset = build_dataset(args.results_dir, args.dataset_name)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logging.info("Wrote %d entries to %s", len(dataset), args.output)

    # Print label distribution
    labels: dict[str, int] = {}
    for d in dataset:
        labels[d["output"]] = labels.get(d["output"], 0) + 1
    logging.info("Label distribution: %s", labels)


if __name__ == "__main__":
    main()
