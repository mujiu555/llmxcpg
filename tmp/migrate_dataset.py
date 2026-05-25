#!/usr/bin/env python3
"""
Migrate old pipeline output to the new-format detection dataset.

Joins data from up to three sources to recover all original dataset fields
without re-running Joern or LLM queries:

  1. generate_and_run_queries.py output  (has ``details`` with full original sample)
  2. construct_slice.py output           (has ``enhanced_code`` per path)
  3. Original dataset JSON               (optional — supplementary fields)

Usage:
    # Full join (recommended):
    python migrate_dataset.py \
        -q query_results/ \
        -s slice_results/ \
        -d original_dataset.json \
        -o output.json

    # Minimal: slice results only (construct_slice.py):
    python migrate_dataset.py -s slice_results/ -o output.json
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Optional

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


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_json_list(path: str) -> list[dict]:
    """Load a JSON file that must contain a list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} does not contain a JSON list")
    return data


def load_result_dir(results_dir: str) -> list[dict]:
    """Load all thread_N_results.json files from a directory, return flat list."""
    entries: list[dict] = []
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(results_dir, fn)
        data = load_json_list(path)
        entries.extend(data)
        logging.info("Loaded %d entries from %s", len(data), fn)
    return entries


# ---------------------------------------------------------------------------
# Name normalisation — strip index prefix so old and new formats match
# ---------------------------------------------------------------------------

def resolve_join_key(entry: dict, *, prefer: str = "file_name") -> str:
    """
    Return a canonical join key for *entry*.

    Uses *prefer* first, then falls back to other common keys.
    Strips a leading ``{digits}_`` index prefix so that old (un-indexed)
    and new (indexed) file names match.
    """
    key = entry.get(prefer) or entry.get("file_name") or entry.get(
        "original_file_name", ""
    )
    base = os.path.basename(key)
    # Strip leading index prefix  e.g. "0_foo.c" → "foo.c"
    while True:
        idx, sep, rest = base.partition("_")
        if idx.isdigit() and sep:
            base = rest
        else:
            break
    return base


# ---------------------------------------------------------------------------
# Enhanced code
# ---------------------------------------------------------------------------

def resolve_enhanced_code(entry: dict) -> str:
    """Return enhanced code, preferring inline field over file read."""
    code = entry.get("enhanced_code", "")
    if code and code.strip():
        return code
    file_path = entry.get("enhanced_code_file", "")
    if file_path and os.path.isfile(file_path):
        logging.debug("Reading enhanced_code from file: %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_dataset(
    slice_dir: str,
    query_dir: Optional[str] = None,
    original_dataset_path: Optional[str] = None,
    dataset_name: str = "LLMxCPG",
) -> list[dict]:
    """
    Build a new-format detection dataset.

    Parameters
    ----------
    slice_dir:
        Path to construct_slice.py output directory (thread_N_results.json).
    query_dir:
        Path to generate_and_run_queries.py output directory (thread_N_results.json).
        Entries here carry the ``details`` field with all original dataset fields.
    original_dataset_path:
        Path to the original dataset JSON (e.g. formai_query_generation.json).
        Used as a supplement for any fields missing from both query and slice results.
    dataset_name:
        Fallback dataset name.
    """

    # --- load slice results (primary source of enhanced_code) ---
    slice_entries = load_result_dir(slice_dir)
    logging.info("Slice entries total: %d", len(slice_entries))

    # --- load query results (source of ``details``) ---
    query_lookup: dict[str, dict] = {}
    if query_dir:
        query_entries = load_result_dir(query_dir)
        for qe in query_entries:
            key = resolve_join_key(qe)
            query_lookup[key] = qe
        logging.info("Query entries: %d  (unique join keys: %d)",
                     len(query_entries), len(query_lookup))

    # --- load original dataset (supplementary source) ---
    original_lookup: dict[str, dict] = {}
    if original_dataset_path:
        orig_data = load_json_list(original_dataset_path)
        for od in orig_data:
            key = resolve_join_key(od)
            original_lookup[key] = od
        logging.info("Original dataset entries: %d  (unique join keys: %d)",
                     len(orig_data), len(original_lookup))

    # --- assign indices ---
    # Gather all unique original file identities across slice entries.
    file_groups: dict[str, list[dict]] = defaultdict(list)
    for se in slice_entries:
        key = resolve_join_key(se)
        file_groups[key].append(se)

    # Prefer index from query results, then original dataset, then assign
    # sequentially based on sorted join key order (deterministic).
    index_map: dict[str, int] = {}
    seq = 0
    for key in sorted(file_groups.keys()):
        idx = None
        # Try query results
        qe = query_lookup.get(key, {})
        idx = qe.get("index") or qe.get("details", {}).get("index")
        # Try original dataset
        if idx is None:
            oe = original_lookup.get(key, {})
            idx = oe.get("index")
        if idx is None:
            idx = seq
        index_map[key] = int(idx)
        seq += 1

    # --- build detection entries ---
    dataset: list[dict] = []
    skipped_empty = 0
    skipped_duplicate = 0
    seen = set()

    for se in slice_entries:
        code = resolve_enhanced_code(se)
        if not code or not code.strip():
            skipped_empty += 1
            logging.warning(
                "Empty enhanced_code for %s path %s — skipping.",
                se.get("file_name", "?"),
                se.get("path_idx", "?"),
            )
            continue

        join_key = resolve_join_key(se)
        sample_index = index_map.get(join_key, 0)

        # Resolve the richest ``details`` payload
        qe = query_lookup.get(join_key, {})
        oe = original_lookup.get(join_key, {})

        # Start with query-level details (has full original sample)
        details = dict(qe.get("details", {}))
        # Merge any extra fields from the original dataset
        for k, v in oe.items():
            if k not in details:
                details[k] = v
        # Merge slice-level fields (enhanced_code, path, etc.)
        for k, v in se.items():
            if k not in details:
                details[k] = v

        # Normalise the label
        label = str(se.get("label") or qe.get("label") or details.get("label", "")).strip().upper()
        if label not in ("VULNERABLE", "BENIGN"):
            label = "VULNERABLE" if label else "BENIGN"

        # Resolve CWE — prefer deeper sources
        cwe = (
            se.get("cwe")
            or details.get("cwe")
            or qe.get("details", {}).get("cwe")
            or oe.get("cwe")
            or "N/A"
        )

        # Resolve dataset tag
        ds = (
            se.get("dataset")
            or qe.get("details", {}).get("dataset")
            or oe.get("dataset")
            or dataset_name
        )

        # Original file name (un-indexed basename)
        orig_fn = (
            se.get("original_file_name")
            or qe.get("details", {}).get("file_name")
            or oe.get("file_name")
            or join_key
        )
        orig_basename = os.path.basename(orig_fn)

        path_idx = se.get("path_idx", 0)

        # Build indexed file name
        name, ext = os.path.splitext(orig_basename)
        unique_name = f"{sample_index}_{name}_path{path_idx}{ext}"

        # Deduplicate
        dedup_key = (orig_fn, path_idx)
        if dedup_key in seen:
            skipped_duplicate += 1
            continue
        seen.add(dedup_key)

        dataset.append({
            "index": sample_index,
            "instruction": INSTRUCTION,
            "input": code.strip(),
            "output": label,
            "file_name": unique_name,
            "original_file_name": orig_fn,
            "dataset": ds,
            "cwe": cwe,
            "details": details,
        })

    logging.info(
        "Build complete: %d entries, %d skipped (empty), %d skipped (duplicate).",
        len(dataset), skipped_empty, skipped_duplicate,
    )
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Migrate old pipeline output to new-format detection dataset."
    )
    parser.add_argument(
        "-s", "--slice-results-dir", type=str, required=True,
        help="Path to construct_slice.py results/ directory (thread_N_results.json files).",
    )
    parser.add_argument(
        "-q", "--query-results-dir", type=str, default=None,
        help="Path to generate_and_run_queries.py results/ directory (has 'details').",
    )
    parser.add_argument(
        "-d", "--original-dataset", type=str, default=None,
        help="Path to the original dataset JSON (supplementary fields).",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path for the output JSON detection dataset.",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="LLMxCPG",
        help="Fallback dataset name.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    dataset = build_dataset(
        slice_dir=args.slice_results_dir,
        query_dir=args.query_results_dir,
        original_dataset_path=args.original_dataset,
        dataset_name=args.dataset_name,
    )

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logging.info("Wrote %d entries to %s", len(dataset), args.output)

    labels: dict[str, int] = {}
    for d in dataset:
        labels[d["output"]] = labels.get(d["output"], 0) + 1
    logging.info("Label distribution: %s", labels)

    indices = sorted({d["index"] for d in dataset})
    logging.info("Index range: %d – %d (%d unique)", min(indices), max(indices), len(indices))

    # Report on details richness
    detail_keys = set()
    for d in dataset:
        detail_keys.update(d.get("details", {}).keys())
    logging.info("Details fields preserved (%d): %s", len(detail_keys), sorted(detail_keys))


if __name__ == "__main__":
    main()
