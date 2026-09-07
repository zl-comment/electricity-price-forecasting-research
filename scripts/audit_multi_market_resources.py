"""Audit the multi-market paper and data library without modifying source data."""

import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper/multi_market_energy_reserve"
DATA_DIR = ROOT / "data/multi_market_energy_reserve"


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_manifest(path):
    results = []
    for item in json.loads(path.read_text(encoding="utf-8"))["files"]:
        target = ROOT / item["path"]
        results.append(
            {
                "path": item["path"],
                "exists": target.exists(),
                "sha256_match": target.exists() and file_sha256(target) == item["sha256"],
            }
        )
    return results


def audit_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        missing = Counter()
        rows = 0
        first = None
        last = None
        for row in reader:
            rows += 1
            first = first or row
            last = row
            for key in fields:
                if row.get(key) is None or str(row[key]).strip().lower() in ("", "null", "none", "nan", "na"):
                    missing[key] += 1
        return {
            "path": str(path.relative_to(ROOT)),
            "rows": rows,
            "columns": fields,
            "missing_by_column": dict(missing),
            "all_missing_columns": [key for key, count in missing.items() if count == rows],
            "first_row": first,
            "last_row": last,
        }


def audit_pdfs():
    results = []
    for path in sorted(PAPER_DIR.glob("*.pdf")):
        info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True)
        fields = dict(line.split(":", 1) for line in info.stdout.splitlines() if ":" in line)
        text = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True
        ).stdout
        results.append(
            {
                "path": str(path.relative_to(ROOT)),
                "pages": int(fields["Pages"]),
                "extractable_characters": len(text),
                "opening_text": text[:500],
            }
        )
    return results


def main():
    integrity = []
    for manifest in (DATA_DIR / "download_manifest.json", PAPER_DIR / "download_manifest.json"):
        integrity.extend(audit_manifest(manifest))
    csv_results = [
        audit_csv(path)
        for path in sorted(DATA_DIR.rglob("*.csv"))
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    ]
    pdf_results = audit_pdfs()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity": integrity,
        "csv_files": csv_results,
        "pdf_files": pdf_results,
    }
    (DATA_DIR / "data_audit.json").write_text(
        json.dumps({key: value for key, value in report.items() if key != "pdf_files"}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (PAPER_DIR / "pdf_audit.json").write_text(
        json.dumps(pdf_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert integrity and all(item["exists"] and item["sha256_match"] for item in integrity)
    assert pdf_results and all(item["extractable_characters"] > 100 for item in pdf_results)
    print(
        json.dumps(
            {
                "checksums_verified": len(integrity),
                "csv_audited": len(csv_results),
                "pdf_audited": len(pdf_results),
                "csvs_with_missing_values": [
                    item["path"] for item in csv_results if item["missing_by_column"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
