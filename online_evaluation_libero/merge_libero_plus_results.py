#!/usr/bin/env python3
"""Merge chunked LIBERO-plus eval JSON files and print category averages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_evaluation_libero.libero_runtime import aggregate_plus_categories


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_root", type=str, default="/data/user_data/ziyinx/LIBERO-plus")
    parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    merged_results = {}
    metadata = {}

    for path in sorted(Path().glob(args.input_glob)):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = {
            key: payload[key]
            for key in (
                "benchmark",
                "libero_root",
                "libero_variant",
                "checkpoint",
                "model_type",
                "num_eval",
                "max_steps",
            )
            if key in payload
        }
        merged_results.update(payload.get("results", {}))

    if not merged_results:
        raise ValueError(f"No eval files matched glob: {args.input_glob}")

    output = dict(metadata)
    output["results"] = merged_results
    output["mean_success_rate"] = sum(
        float(item["success_rate"]) for item in merged_results.values()
    ) / len(merged_results)

    benchmark = metadata.get("benchmark")
    if benchmark:
        category_rates = aggregate_plus_categories(
            benchmark,
            merged_results,
            args.libero_root,
        )
        if category_rates:
            output["category_success_rates"] = category_rates
            output["category_mean_success_rate"] = sum(category_rates.values()) / len(
                category_rates
            )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Wrote merged results to {out_path}")
    print(f"mean_success_rate={output['mean_success_rate']:.4f}")
    if "category_success_rates" in output:
        for category, rate in output["category_success_rates"].items():
            print(f"  {category}: {rate:.4f}")


if __name__ == "__main__":
    main()
