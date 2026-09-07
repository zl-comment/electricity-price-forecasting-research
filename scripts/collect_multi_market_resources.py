"""Collect target-specific papers and data for energy-reserve multi-market research.

Usage:
    python3 scripts/collect_multi_market_resources.py papers
    python3 scripts/collect_multi_market_resources.py data

Downloads public source files only. Energinet API responses are preserved as JSON;
their ``records`` arrays are also serialized losslessly to CSV for convenient use.
"""

import argparse
import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-09-06"
PAPER_DIR = ROOT / "paper/multi_market_energy_reserve"
DATA_DIR = ROOT / "data/multi_market_energy_reserve"
USER_AGENT = "energy-reserve-multi-market-research/1.0"

PAPERS = [
    {
        "id": "P01",
        "title": "An Integrated Forecasting and Optimization Framework for Battery Energy Storage Participation in Spanish Electricity Markets",
        "date": "2026-07-29",
        "doi": "10.32604/ee.2026.084810",
        "version": "publisher PDF, CC BY 4.0",
        "file": "P01_2026_Integrated_Forecasting_Spanish_Markets.pdf",
        "url": "https://www.techscience.com/energy/online/detail/27739/pdf",
    },
    {
        "id": "P02",
        "title": "Optimizing Multi-Market Participation of Battery and Electrolyser Systems Based on Field Performance",
        "date": "2026-08-17",
        "doi": "10.48550/arXiv.2608.16238",
        "version": "arXiv v1; accepted at IEEE PESGM 2026",
        "file": "P02_2026_Field_Performance_Multi_Market_BESS.pdf",
        "url": "https://arxiv.org/pdf/2608.16238v1",
    },
    {
        "id": "P03",
        "title": "Joint Bidding on Intraday and Frequency Containment Reserve Markets",
        "date": "2025-10-03",
        "doi": "10.48550/arXiv.2510.03209",
        "version": "arXiv v1; working-paper version dated 2026-03-22 is discussed on the paper page",
        "file": "P03_2025_Joint_Intraday_FCR_Bidding.pdf",
        "url": "https://arxiv.org/pdf/2510.03209v1",
    },
    {
        "id": "P04",
        "title": "Storage Participation in Electricity Markets: Time Discretization through Robust Optimization",
        "date": "2026-05-11 revision",
        "doi": "10.48550/arXiv.2510.10856",
        "version": "arXiv v2",
        "file": "P04_2026_Storage_Energy_Ancillary_Services.pdf",
        "url": "https://arxiv.org/pdf/2510.10856v2",
    },
    {
        "id": "P05",
        "title": "Multi-Interval Energy-Reserve Co-Optimization with SoC-Dependent Bids from Battery Storage",
        "date": "2024-08-24 revision",
        "doi": "10.48550/arXiv.2401.15525",
        "version": "arXiv v2; market-clearing foundation",
        "file": "P05_2024_Energy_Reserve_Cooptimization_SoC_Bids.pdf",
        "url": "https://arxiv.org/pdf/2401.15525v2",
    },
    {
        "id": "P06",
        "title": "Conformal Prediction for Electricity Price Forecasting in the Day-Ahead and Real-Time Balancing Market",
        "date": "2025-02-07",
        "doi": "10.48550/arXiv.2502.04935",
        "version": "arXiv v1; probabilistic multi-settlement forecasting baseline",
        "file": "P06_2025_Conformal_DA_Balancing_Price_Forecasting.pdf",
        "url": "https://arxiv.org/pdf/2502.04935v1",
    },
    {
        "id": "P07",
        "title": "A Decision-Focused Predict-then-Bid Framework for Strategic Energy Storage",
        "date": "2025-05-06 revision",
        "doi": "10.48550/arXiv.2505.01551",
        "version": "arXiv v2; decision-focused bidding method bridge",
        "file": "P07_2025_Decision_Focused_Predict_Then_Bid.pdf",
        "url": "https://arxiv.org/pdf/2505.01551v2",
    },
    {
        "id": "P08",
        "title": "Probabilistic Forecasting for Day-ahead Electricity Prices, Battery Trading Strategies and the Economic Evaluation of Predictive Accuracy",
        "date": "2026-04-21",
        "doi": "10.48550/arXiv.2604.19580",
        "version": "arXiv v1; decision-value evaluation foundation",
        "file": "P08_2026_Probabilistic_Forecasting_Battery_Economic_Value.pdf",
        "url": "https://arxiv.org/pdf/2604.19580v1",
    },
]

ENERGINET_DATASETS = [
    ("DayAheadPrices", "day_ahead_prices"),
    ("AfrrReservesNordic", "afrr_capacity_market"),
    ("ImbalancePrice", "imbalance_and_activation"),
    ("Forecasts_Hour", "wind_solar_forecasts"),
    ("ProductionConsumptionSettlement", "production_consumption_settlement"),
]
ENERGINET_START = "2025-10-01"
ENERGINET_END = "2026-09-01"
RTS_REPO = "GridMod/RTS-GMLC"
RTS_COMMIT = "3ece0d3725c844056132393ee252b3083dd4eab4"


def retrieve(url, timeout=120):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.url, response.headers.get("Content-Type", "")


def sha256(body):
    return hashlib.sha256(body).hexdigest()


def write_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def download_papers():
    results = []
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for item in PAPERS:
        target = PAPER_DIR / item["file"]
        body, resolved, content_type = retrieve(item["url"])
        if not body.startswith(b"%PDF-"):
            raise ValueError(f"Not a PDF: {item['url']} ({content_type})")
        write_bytes(target, body)
        result = dict(item)
        result.update(
            path=str(target.relative_to(ROOT)),
            source_url=item["url"],
            resolved_url=resolved,
            content_type=content_type,
            bytes=len(body),
            sha256=sha256(body),
            status="downloaded",
        )
        results.append(result)
        print(f"OK {result['path']} ({len(body)} bytes)", flush=True)
    manifest = {
        "research_cutoff": CUTOFF,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Energy-reserve multi-market forecasting, bidding, co-optimization, and decision-value evaluation.",
        "files": results,
    }
    (PAPER_DIR / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def energinet_url(dataset):
    query = urllib.parse.urlencode(
        {
            "start": ENERGINET_START,
            "end": ENERGINET_END,
            "filter": json.dumps({"PriceArea": ["DK1"]}, separators=(",", ":")),
        }
    )
    return f"https://api.energidataservice.dk/dataset/{dataset}?{query}"


def collect_energinet():
    entries = []
    directory = DATA_DIR / "energinet_dk1"
    directory.mkdir(parents=True, exist_ok=True)
    for dataset, stem in ENERGINET_DATASETS:
        metadata_url = f"https://api.energidataservice.dk/meta/dataset/{dataset}"
        metadata_body, resolved, content_type = retrieve(metadata_url)
        metadata_path = directory / f"{stem}_metadata.json"
        write_bytes(metadata_path, metadata_body)
        entries.append(
            {
                "dataset": dataset,
                "kind": "official_metadata",
                "path": str(metadata_path.relative_to(ROOT)),
                "source_url": metadata_url,
                "resolved_url": resolved,
                "content_type": content_type,
                "bytes": len(metadata_body),
                "sha256": sha256(metadata_body),
                "status": "downloaded",
            }
        )

        source_url = energinet_url(dataset)
        body, resolved, content_type = retrieve(source_url)
        response = json.loads(body)
        records = response.get("records", [])
        if not records:
            raise ValueError(f"No records returned for {dataset}")
        raw_path = directory / f"{stem}_raw.json"
        write_bytes(raw_path, body)
        entries.append(
            {
                "dataset": dataset,
                "kind": "official_api_response",
                "path": str(raw_path.relative_to(ROOT)),
                "source_url": source_url,
                "resolved_url": resolved,
                "content_type": content_type,
                "start_inclusive": ENERGINET_START,
                "end_exclusive": ENERGINET_END,
                "price_area": "DK1",
                "records": len(records),
                "bytes": len(body),
                "sha256": sha256(body),
                "status": "downloaded",
            }
        )

        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        csv_body = output.getvalue().encode("utf-8")
        csv_path = directory / f"{stem}.csv"
        write_bytes(csv_path, csv_body)
        entries.append(
            {
                "dataset": dataset,
                "kind": "lossless_records_to_csv",
                "path": str(csv_path.relative_to(ROOT)),
                "derived_from": str(raw_path.relative_to(ROOT)),
                "transformation": "JSON records serialized to CSV; no filtering, interpolation, aggregation, or value conversion after API response.",
                "records": len(records),
                "bytes": len(csv_body),
                "sha256": sha256(csv_body),
                "status": "generated",
            }
        )
        print(f"OK {dataset}: {len(records)} DK1 records", flush=True)

    license_url = "https://www.energidataservice.dk/terms-and-conditions"
    body, resolved, content_type = retrieve(license_url)
    license_path = directory / "upstream_terms_and_conditions.html"
    write_bytes(license_path, body)
    entries.append(
        {
            "dataset": "Energinet license",
            "kind": "license",
            "path": str(license_path.relative_to(ROOT)),
            "source_url": license_url,
            "resolved_url": resolved,
            "content_type": content_type,
            "bytes": len(body),
            "sha256": sha256(body),
            "status": "downloaded",
            "license": "CC BY 4.0",
        }
    )
    return entries


def collect_finland_afrr():
    entries = []
    directory = DATA_DIR / "finland_afrr_zenodo_17494556"
    directory.mkdir(parents=True, exist_ok=True)
    metadata_url = "https://zenodo.org/api/records/17494556"
    metadata_body, resolved, content_type = retrieve(metadata_url)
    metadata = json.loads(metadata_body)
    metadata_path = directory / "zenodo_metadata.json"
    write_bytes(metadata_path, metadata_body)
    entries.append(
        {
            "dataset": "Finland aFRR Energy Market and Weather Data",
            "kind": "repository_metadata",
            "path": str(metadata_path.relative_to(ROOT)),
            "source_url": metadata_url,
            "resolved_url": resolved,
            "content_type": content_type,
            "bytes": len(metadata_body),
            "sha256": sha256(metadata_body),
            "status": "downloaded",
        }
    )
    files = metadata.get("files", [])
    if len(files) != 1:
        raise ValueError(f"Unexpected Zenodo file count: {len(files)}")
    remote = files[0]
    source_url = remote["links"]["self"]
    archive_body, resolved, content_type = retrieve(source_url)
    archive_path = directory / remote["key"]
    write_bytes(archive_path, archive_body)
    entries.append(
        {
            "dataset": "Finland aFRR Energy Market and Weather Data",
            "kind": "source_archive",
            "path": str(archive_path.relative_to(ROOT)),
            "source_url": source_url,
            "resolved_url": resolved,
            "content_type": content_type,
            "bytes": len(archive_body),
            "sha256": sha256(archive_body),
            "status": "downloaded",
        }
    )
    with zipfile.ZipFile(io.BytesIO(archive_body)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {info.filename}")
            if "__MACOSX" in relative.parts or relative.name.startswith("._"):
                continue
            member_body = archive.read(info)
            target = directory / "extracted" / relative
            write_bytes(target, member_body)
            entries.append(
                {
                    "dataset": "Finland aFRR Energy Market and Weather Data",
                    "kind": "extracted_archive_member",
                    "path": str(target.relative_to(ROOT)),
                    "derived_from": str(archive_path.relative_to(ROOT)),
                    "bytes": len(member_body),
                    "sha256": sha256(member_body),
                    "status": "extracted",
                }
            )
    print(f"OK Zenodo 17494556: {len(entries) - 2} extracted files", flush=True)
    return entries


def rts_items():
    tree_url = f"https://api.github.com/repos/{RTS_REPO}/git/trees/{RTS_COMMIT}?recursive=1"
    body, _, _ = retrieve(tree_url)
    tree = json.loads(body)
    if tree.get("truncated"):
        raise ValueError("Truncated RTS-GMLC repository tree")
    for node in tree["tree"]:
        source_path = node["path"]
        if node["type"] != "blob":
            continue
        take = source_path == "README.md" or (
            source_path.startswith("RTS_Data/SourceData/")
            and source_path.count("/") == 2
            and (source_path.endswith(".csv") or source_path.endswith("README.md"))
        ) or (
            source_path.startswith("RTS_Data/timeseries_data_files/")
            and "/origin/" not in source_path
            and (source_path.endswith(".csv") or source_path.endswith("README.md"))
        )
        if take:
            yield node, source_path


def collect_rts():
    entries = []
    directory = DATA_DIR / "rts_gmlc_system"
    for node, source_path in rts_items():
        target = directory / source_path
        source_url = f"https://raw.githubusercontent.com/{RTS_REPO}/{RTS_COMMIT}/{urllib.parse.quote(source_path)}"
        if target.exists():
            body = target.read_bytes()
            resolved = source_url
            content_type = "existing file revalidated"
        else:
            body, resolved, content_type = retrieve(source_url)
            write_bytes(target, body)
        blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        if blob != node["sha"]:
            raise ValueError(f"RTS-GMLC Git blob mismatch: {source_path}")
        entries.append(
            {
                "dataset": "RTS-GMLC",
                "kind": "pinned_git_blob",
                "repository": RTS_REPO,
                "commit": RTS_COMMIT,
                "upstream_path": source_path,
                "git_blob_sha": node["sha"],
                "path": str(target.relative_to(ROOT)),
                "source_url": source_url,
                "resolved_url": resolved,
                "content_type": content_type,
                "bytes": len(body),
                "sha256": sha256(body),
                "status": "downloaded",
            }
        )
    print(f"OK RTS-GMLC: {len(entries)} pinned files", flush=True)
    return entries


def collect_data():
    entries = []
    entries.extend(collect_energinet())
    entries.extend(collect_finland_afrr())
    entries.extend(collect_rts())
    manifest = {
        "research_cutoff": CUTOFF,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Observed DK1 energy/aFRR/balancing data, a Finland reserve-price forecasting dataset, and a pinned system benchmark for transparent co-clearing experiments.",
        "files": entries,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "entries": len(entries),
                "bytes": sum(item.get("bytes", 0) for item in entries),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=("papers", "data", "all"))
    args = parser.parse_args()
    if args.group in ("papers", "all"):
        download_papers()
    if args.group in ("data", "all"):
        collect_data()


if __name__ == "__main__":
    main()
