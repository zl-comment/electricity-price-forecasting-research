"""Download pinned, public Q3 research resources; record provenance and checksums.

Run from any directory: python scripts/collect_q3_resources.py datasets|papers
Only downloads source files; does not run third-party code or alter raw data.
"""

import argparse
import concurrent.futures
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-09-06"
REPOS = [
    ("GridMod/RTS-GMLC", "3ece0d3725c844056132393ee252b3083dd4eab4", "rts_gmlc"),
    ("citylearn-project/CityLearn", "599a970669daf77f8409ce970aca79e9f03d9047", "citylearn_2022"),
]
PAPERS = [
    {
        "id": "P01", "title": "Decision-focused learning for optimal PV-battery scheduling",
        "doi": "10.1016/j.est.2026.121152", "date": "2026-04-10 (issue); arXiv v1 2026-05-27",
        "version": "author manuscript, arXiv v1; journal article separately verified",
        "path": "P01_2026_DFL_PV_Battery_Scheduling.pdf",
        "urls": ["https://arxiv.org/pdf/2605.28340v1"],
    },
    {
        "id": "P02", "title": "Decision-calibrated prediction sets for robust power system operations",
        "doi": "10.48550/arXiv.2606.02081", "date": "2026-06-01",
        "version": "preprint v1, not a verified peer-reviewed publication",
        "path": "P02_2026_Decision_Calibrated_Robust_Operations.pdf",
        "urls": ["https://arxiv.org/pdf/2606.02081v1"],
    },
    {
        "id": "P03", "title": "Decision focused online learning for real time energy aware scheduling of interconnected data centers with photovoltaic generation and battery storage",
        "doi": "10.1038/s41598-026-67967-z", "date": "2026-08-26",
        "version": "publisher early-access accepted article",
        "path": "P03_2026_Online_DFL_Data_Center_Storage.pdf",
        "urls": ["https://www.nature.com/articles/s41598-026-67967-z_reference.pdf", "https://www.nature.com/articles/s41598-026-67967-z_reference.pdf?error=cookies_not_supported"],
    },
    {
        "id": "P03S", "title": "Supplementary input features for P03",
        "doi": "10.1038/s41598-026-67967-z", "date": "2026-08-26",
        "version": "publisher supplement; not the article or a dataset",
        "path": "P03S_2026_Online_DFL_Supplement.pdf",
        "urls": ["https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-026-67967-z/MediaObjects/41598_2026_67967_MOESM1_ESM.pdf"],
    },
    {
        "id": "P04", "title": "Reserve Deliverability-Constrained Battery Energy Storage Sizing for Renewable Power Plants Considering Cycling Degradation",
        "doi": "10.3390/electronics15153442", "date": "2026-08-03",
        "version": "publisher article",
        "path": "P04_2026_Reserve_Deliverability_BESS.pdf",
        "urls": ["https://www.mdpi.com/2079-9292/15/15/3442/pdf", "https://mdpi-res.com/d_attachment/electronics/electronics-15-03442/article_deploy/electronics-15-03442.pdf"],
    },
    {
        "id": "P05", "title": "Optimal day-ahead scheduling for explicit demand response provision of a renewable energy hub with hot water preparation",
        "doi": "10.1016/j.apenergy.2026.127562", "date": "2026-05-01 (issue)",
        "version": "publisher open-access article",
        "path": "P05_2026_Explicit_DR_Energy_Hub.pdf",
        "urls": ["https://www.sciencedirect.com/science/article/pii/S030626192600214X/pdfft?isDTMRedir=true&download=true"],
    },
    {
        "id": "P06", "title": "Safe Reinforcement Learning for Battery Energy Storage Participation in the Imbalance Settlement",
        "doi": "10.1109/TEMPR.2025.3639758", "date": "2025-12-04 (first online)",
        "version": "accepted author manuscript from institutional repository",
        "path": "P06_2025_Safe_RL_BESS_Imbalance.pdf",
        "urls": ["https://orbi.umons.ac.be/bitstream/20.500.12907/54047/1/Safe_RL_for_BESS_operation_in_the_real_time_market-2.pdf"],
    },
    {
        "id": "P07", "title": "Impact of data for forecasting on performance of model predictive control in buildings with smart energy storage",
        "doi": "10.1016/j.enbuild.2024.114605", "date": "2024 (baseline; not a 2026 paper)",
        "version": "institutional repository full text",
        "path": "P07_2024_Forecast_Data_MPC_CityLearn.pdf",
        "urls": ["https://api.repository.cam.ac.uk/server/api/core/bitstreams/640ba41f-f9a0-430a-b795-8b094af90e2f/content"],
    },
    {
        "id": "B01", "title": "The CityLearn Challenge 2022",
        "doi": None, "date": "2023",
        "version": "PMLR proceedings; dataset background, not a new Q3 method",
        "path": "B01_2023_CityLearn_Challenge_2022.pdf",
        "urls": ["https://proceedings.mlr.press/v220/nweye23a/nweye23a.pdf"],
    },
    {
        "id": "B02", "title": "RTS-GMLC documentation",
        "doi": None, "date": "repository snapshot 2025-10-23",
        "version": "official benchmark documentation, not a new Q3 paper",
        "path": "B02_RTS_GMLC_Documentation.pdf",
        "urls": ["https://raw.githubusercontent.com/GridMod/RTS-GMLC/3ece0d3725c844056132393ee252b3083dd4eab4/RTS-GMLC.pdf"],
    },
]


def retrieve(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Q3-academic-resource-collector/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read(), response.url, response.headers.get("Content-Type", "")


def download(item):
    result = {key: value for key, value in item.items() if key != "urls"}
    result["attempts"] = []
    target = ROOT / item["path"]
    for url in item["urls"]:
        try:
            if target.exists():
                body = target.read_bytes()
                final_url, content_type = url, "existing file, revalidated"
            else:
                body, final_url, content_type = retrieve(url)
            if item.get("kind") == "pdf" and not body.startswith(b"%PDF-"):
                raise ValueError("Response is not a PDF; no file saved")
            if item.get("git_blob_sha"):
                blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
                if blob != item["git_blob_sha"]:
                    raise ValueError("Git blob integrity mismatch")
            if not body:
                raise ValueError("Empty response")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open("xb") as output:
                    output.write(body)
            result.update(status="downloaded", source_url=url, resolved_url=final_url,
                          content_type=content_type, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            print(f"OK {item['path']} ({len(body)} bytes)", flush=True)
            return result
        except Exception as error:
            result["attempts"].append({"url": url, "error": str(error)})
    result["status"] = "unavailable"
    print(f"UNAVAILABLE {item['path']}: {result['attempts'][-1]['error']}", flush=True)
    return result


def dataset_items():
    output = []
    for repo, commit, name in REPOS:
        body, _, _ = retrieve(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1")
        tree = json.loads(body)
        if tree.get("truncated"):
            raise ValueError("Truncated repository tree")
        for node in tree["tree"]:
            source_path = node["path"]
            if node["type"] != "blob":
                continue
            if name == "rts_gmlc":
                take = source_path == "README.md" or (
                    source_path.startswith("RTS_Data/SourceData/") and source_path.count("/") == 2
                    and (source_path.endswith(".csv") or source_path.endswith("README.md"))) or (
                    source_path.startswith("RTS_Data/timeseries_data_files/") and "/origin/" not in source_path
                    and (source_path.endswith(".csv") or source_path.endswith("README.md")))
                relative_path = source_path
            else:
                prefix = "data/datasets/citylearn_challenge_2022_phase_all/"
                take = source_path in ("LICENSE", "README.md") or source_path.startswith(prefix)
                relative_path = source_path[len(prefix):] if source_path.startswith(prefix) else "upstream_" + source_path
            if take:
                output.append({"dataset": name, "repository": repo, "commit": commit,
                               "upstream_path": source_path, "git_blob_sha": node["sha"],
                               "path": f"data/Q3/{name}/{relative_path}",
                               "urls": [f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(source_path)}"]})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["datasets", "papers"])
    args = parser.parse_args()
    if args.group == "datasets":
        items = dataset_items()
        manifest_path = ROOT / "data/Q3/download_manifest.json"
    else:
        items = [{**paper, "path": "paper/Q3/" + paper["path"], "kind": "pdf"} for paper in PAPERS]
        manifest_path = ROOT / "paper/Q3/download_manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = {item["path"]: item for item in json.loads(manifest_path.read_text())["files"]}
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(download, items):
            old = previous.get(result["path"])
            if old and old.get("status") == "downloaded" and old.get("sha256") == result.get("sha256"):
                result = old
            results.append(result)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"research_cutoff": CUTOFF,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "files": results}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"downloaded": sum(r["status"] == "downloaded" for r in results),
                      "unavailable": sum(r["status"] != "downloaded" for r in results),
                      "bytes": sum(r.get("bytes", 0) for r in results)}), flush=True)


if __name__ == "__main__":
    main()
