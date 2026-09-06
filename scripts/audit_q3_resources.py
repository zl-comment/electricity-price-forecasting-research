"""Read-only audit of downloaded inputs; generate JSON reports (stdlib only)."""
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit_csv(path):
    delimiter = ";" if "elia" in path.parts else ","
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        columns = reader.fieldnames or []
        nulls = Counter()
        keys = set()
        times = set()
        categories = {k: set() for k in ("region", "offshoreonshore", "gridconnectiontype") if k in columns}
        rows = bad_width = 0
        first = last = None
        for row in reader:
            rows += 1
            first = first or row
            last = row
            bad_width += int(None in row or any(v is None for v in row.values()))
            for k in columns:
                if row.get(k) is None or str(row[k]).strip().lower() in ("", "na", "nan", "null", "none"):
                    nulls[k] += 1
            for k in categories:
                categories[k].add(row[k])
            if "datetime" in columns:
                key = tuple(row[k] for k in ("datetime", *categories))
                keys.add(key)
                times.add(row["datetime"])
            elif all(k in columns for k in ("Year", "Month", "Day", "Period")):
                keys.add(tuple(row[k] for k in ("Year", "Month", "Day", "Period")))
    report = {"path": str(path.relative_to(ROOT)), "rows": rows, "columns": columns,
              "malformed_width_rows": bad_width, "missing_by_column": dict(nulls),
              "all_missing_columns": [k for k, n in nulls.items() if n == rows],
              "first_row": first, "last_row": last,
              "categories": {k: sorted(v) for k, v in categories.items()}}
    if keys:
        report["duplicate_key_rows"] = rows - len(keys)
    if times:
        parsed = sorted(datetime.fromisoformat(t) for t in times)
        expected = int((parsed[-1] - parsed[0]).total_seconds() // 900) + 1
        report.update(first_timestamp=parsed[0].isoformat(), last_timestamp=parsed[-1].isoformat(),
                      unique_timestamps=len(times), missing_15min_timestamps_between_endpoints=expected-len(times))
        if categories:
            report["note"] = "Timestamp continuity checks union of categories, not completeness of every category."
    return report


def main():
    integrity = []
    for manifest in (ROOT / "data/Q3/download_manifest.json", ROOT / "data/Q3/elia/download_manifest.json",
                     ROOT / "paper/Q3/download_manifest.json"):
        if not manifest.exists():
            continue
        for entry in json.loads(manifest.read_text())["files"]:
            if entry["status"] != "downloaded":
                continue
            path = ROOT / entry["path"]
            ok = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
            integrity.append({"path": entry["path"], "sha256_match": ok})
    csv_results = [audit_csv(p) for p in sorted((ROOT / "data/Q3").rglob("*.csv"))]
    elia_manifest = ROOT / "data/Q3/elia/download_manifest.json"
    if elia_manifest.exists():
        entries = {f["path"]: f for f in json.loads(elia_manifest.read_text())["files"]}
        for result in csv_results:
            entry = entries.get(result["path"], {})
            if "start_inclusive_utc" in entry:
                start = datetime.fromisoformat(entry["start_inclusive_utc"]).replace(tzinfo=timezone.utc)
                end = datetime.fromisoformat(entry["end_exclusive_utc"]).replace(tzinfo=timezone.utc)
                expected = int((end - start).total_seconds() // 900)
                result["requested_window_expected_timestamps"] = expected
                result["missing_timestamps_in_requested_window"] = expected - result["unique_timestamps"]
    data_report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
                   "integrity": integrity, "csv_files": csv_results}
    (ROOT / "data/Q3/data_audit.json").write_text(json.dumps(data_report, indent=2, ensure_ascii=False) + "\n")
    pdf_results = []
    for path in sorted((ROOT / "paper/Q3").glob("*.pdf")):
        info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True)
        fields = dict(line.split(":", 1) for line in info.stdout.splitlines() if ":" in line)
        text = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True).stdout
        pdf_results.append({"path": str(path.relative_to(ROOT)), "pages": int(fields["Pages"]),
                            "extractable_characters": len(text), "opening_text": text[:500]})
    (ROOT / "paper/Q3/pdf_audit.json").write_text(json.dumps(pdf_results, indent=2, ensure_ascii=False) + "\n")
    assert all(r["sha256_match"] for r in integrity), "Checksum mismatch"
    assert all(not r["malformed_width_rows"] for r in csv_results), "Malformed CSV"
    assert all(r["extractable_characters"] > 100 for r in pdf_results), "PDF text extraction failed"
    print(json.dumps({"checksums_verified": len(integrity), "csv_audited": len(csv_results),
                      "pdf_audited": len(pdf_results), "csv_missing_columns": {
                          r["path"]: r["all_missing_columns"] for r in csv_results if r["all_missing_columns"]}}, indent=2))


if __name__ == "__main__":
    main()
