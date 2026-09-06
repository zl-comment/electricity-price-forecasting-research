"""Download official Elia historical data, without changing its values.

Annual 2023 reference for the 2026 reserve paper + latest complete UTC days.
Wind retains all component categories; PV selects only the Belgium total.
"""
import concurrent.futures
import csv
import io
import json
from datetime import datetime, timezone
from urllib.parse import urlencode

from collect_q3_resources import ROOT, download

BASE = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/"


def main():
    items = []
    for dataset in ("ods001", "ods031", "ods032"):
        items.append({"dataset": dataset, "path": f"data/Q3/elia/{dataset}_metadata.json",
                      "urls": [BASE + dataset]})
        for label, start, end in (("2023", "2023-01-01", "2024-01-01"),
                                  ("20260801_20260905", "2026-08-01", "2026-09-06")):
            where = f"datetime >= date'{start}' AND datetime < date'{end}'"
            if dataset == "ods032":
                where += " AND region = 'Belgium'"
            query = urlencode({"where": where, "order_by": "datetime", "timezone": "UTC",
                               "use_labels": "false", "delimiter": ";"})
            items.append({"dataset": dataset, "start_inclusive_utc": start,
                          "end_exclusive_utc": end, "where": where,
                          "path": f"data/Q3/elia/{dataset}_{label}.csv",
                          "urls": [BASE + dataset + "/exports/csv?" + query]})
    items.append({"dataset": "license", "path": "data/Q3/elia/upstream_licence.html",
                  "urls": ["https://opendata.elia.be/pages/licence/"]})
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for result in executor.map(download, items):
            if result["status"] == "downloaded" and result["path"].endswith(".csv"):
                path = ROOT / result["path"]
                rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")), delimiter=";"))
                if not rows or "datetime" not in rows[0]:
                    raise ValueError(f"Invalid or empty CSV: {path}")
                result["rows"] = len(rows)
            results.append(result)
    manifest = ROOT / "data/Q3/elia/download_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"research_cutoff": "2026-09-06",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Historical exports can contain retrospective corrections, not an as-issued vintage archive.",
        "files": results}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"downloaded": sum(r["status"] == "downloaded" for r in results),
                      "unavailable": sum(r["status"] != "downloaded" for r in results),
                      "bytes": sum(r.get("bytes", 0) for r in results)}), flush=True)


if __name__ == "__main__":
    main()
