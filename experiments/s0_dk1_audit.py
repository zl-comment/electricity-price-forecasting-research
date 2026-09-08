#!/usr/bin/env python3
"""Run the S0 structural audit (A-D) on the Energinet DK1 tables."""

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from epf_harness.dk1_audit import audit_dataset, collect_findings, common_window


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def environment_fingerprint() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
    }


def write_outputs(summary: dict, findings: list, output: dict) -> Path:
    directory = REPOSITORY_ROOT / output["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / output["summary_json"]).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    with (directory / output["findings_csv"]).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "price_area", "check", "finding", "value"])
        writer.writerows(findings)
    return directory


def main() -> None:
    arguments = parse_arguments()
    config = yaml.safe_load(arguments.config.open(encoding="utf-8"))
    np.random.seed(config["protocol"]["random_seed"])
    root = REPOSITORY_ROOT / config["data"]["root"]
    primary_price_area = config["data"]["primary_price_area"]
    audits = {}
    for name, settings in config["datasets"].items():
        print(f"Auditing {name}", flush=True)
        audits[name] = audit_dataset(root, settings, config)
    findings = collect_findings(audits)
    summary = {
        "random_seed": config["protocol"]["random_seed"],
        "environment_fingerprint": environment_fingerprint(),
        "primary_price_area": primary_price_area,
        "datasets": audits,
        "common_window": common_window(audits, primary_price_area),
        "findings": [
            {"dataset": d, "price_area": a, "check": c, "finding": f, "value": v}
            for d, a, c, f, v in findings
        ],
    }
    directory = write_outputs(summary, findings, config["output"])
    print(json.dumps(summary["common_window"], indent=2, ensure_ascii=False), flush=True)
    print(f"{len(findings)} findings written to {directory}", flush=True)


if __name__ == "__main__":
    main()
