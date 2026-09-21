#!/usr/bin/env python3
"""Extract market/dataset evidence for the electricity-market forecast corpus.

This is an audit helper, not an automatic fact generator.  It deliberately
records the evidence level so that title/abstract extraction is not cited as if
the paper's data section had been inspected.
"""

import argparse
import csv
import html
import json
import re
from pathlib import Path


MARKETS = {
    "Australia NEM/AEMO": r"Australian National Electricity Market|\bAEMO\b|\bNEM\b|New South Wales|\bNSW\b|South Australia",
    "Nord Pool/Nordic": r"Nord Pool|Nordic (?:electricity|power|energy)|Nordic/Baltic|Elspot",
    "Germany/EPEX-DE": r"German(?:y|-Austrian)?|EPEX-DE|DE-LU|Germany-Austria|Germany-Luxembourg",
    "PJM": r"\bPJM\b|Pennsylvania-New Jersey-Maryland",
    "Spain/OMIE/MIBEL": r"Spain|Spanish|Iberian|\bOMIE\b|\bMIBEL\b",
    "Italy/GME": r"Italian electricity|Italy|\bGME\b",
    "Ontario/IESO": r"Ontario|\bIESO\b",
    "New Zealand": r"New Zealand",
    "Finland/Fingrid": r"Finland|Finnish|Fingrid",
    "Denmark/DK1": r"Denmark|Danish|\bDK1\b|\bDK2\b|Energinet",
    "UK/Great Britain": r"United Kingdom|\bU\.K\.\b|Great Britain|British electricity|EPEX-UK",
    "Ireland/SEM": r"Ireland|Irish|\bI-SEM\b|\bSEMO\b|EirGrid",
    "Brazil/CCEE": r"Brazil|Brazilian|\bCCEE\b|\bPLD\b",
    "China": r"China|Chinese|Guangdong|Shanxi|Jiangsu|Zhejiang",
    "Iran": r"Iran|Iranian",
    "India/IEX": r"India|Indian electricity|\bIEX\b",
    "Mexico": r"Mexic",
    "California/CAISO": r"\bCAISO\b|California",
    "New York/NYISO": r"\bNYISO\b|New York",
    "ISO New England": r"\bISO-NE\b|New England",
    "ERCOT/Texas": r"\bERCOT\b|Texas electricity",
    "Belgium/EPEX-BE": r"Belgium|Belgian|EPEX-BE|\bELIA\b",
    "France/EPEX-FR": r"France|French electricity|EPEX-FR|\bRTE\b",
    "Portugal": r"Portugal|Portuguese",
    "Greece": r"Greece|Greek electricity",
    "Turkey": r"Turkey|Turkish electricity",
    "Alberta/AESO": r"Alberta|\bAESO\b",
    "Chile": r"Chile|Chilean",
    "Poland": r"Poland|Polish electricity",
    "Romania": r"Romania|Romanian",
    "Czech Republic": r"Czech",
    "Korea": r"Korea|Korean electricity|\bKPX\b",
}

PROVIDERS = {
    "EPF benchmark/epftoolbox": r"open-access benchmark|EPF benchmark|epftoolbox",
    "ENTSO-E Transparency": r"ENTSO-E",
    "EPEX SPOT": r"EPEX(?: SPOT)?",
    "Nord Pool": r"Nord Pool",
    "PJM Data Miner": r"PJM Data Miner|PJM website|\bPJM\b",
    "AEMO": r"\bAEMO\b|Australian Energy Market Operator",
    "OMIE": r"\bOMIE\b",
    "IESO": r"\bIESO\b",
    "GME": r"Gestore dei Mercati Energetici|\bGME\b",
    "Energinet": r"Energinet",
    "Fingrid": r"Fingrid",
    "ELIA": r"\bELIA\b",
    "RTE": r"\bRTE\b",
    "SMARD/Bundesnetzagentur": r"\bSMARD\b|Bundesnetzagentur",
    "NYISO": r"\bNYISO\b",
    "CAISO": r"\bCAISO\b",
    "ERCOT": r"\bERCOT\b",
    "ISO-NE": r"\bISO-NE\b",
    "EirGrid/SEMO": r"EirGrid|\bSEMO\b",
    "New Zealand EMI": r"Electricity Market Information|\bEMI\b",
    "IEX": r"Indian Energy Exchange|\bIEX\b",
    "CCEE": r"\bCCEE\b",
    "GEFCom2014": r"GEFCom\s*2014",
    "Kaggle": r"Kaggle",
    "U.S. EIA": r"Energy Information Administration|\bEIA\b",
}

MANUAL_OVERRIDES = {
    "10.1016/j.apenergy.2021.116983": {
        "market_or_region": "Nord Pool/Nordic; PJM; Belgium/EPEX-BE; France/EPEX-FR; Germany/EPEX-DE",
        "data_provider_or_named_dataset": "EPF benchmark/epftoolbox; Nord Pool; PJM Data Miner; ENTSO-E Transparency; ELIA; RTE",
        "data_type": "review_and_open_benchmark", "time_granularity": "hourly",
        "reported_period_and_fields": "Six years per market: NP/PJM 2013-01-01--2018-12-24; BE/FR 2011-01-09--2016-12-31; DE 2012-01-09--2017-12-31. Day-ahead price plus two market-specific day-ahead load/generation covariates.",
    },
    "10.1016/j.ijforecast.2022.03.001": {
        "market_or_region": "Nord Pool/Nordic; PJM; Belgium/EPEX-BE; France/EPEX-FR; Germany/EPEX-DE",
        "data_provider_or_named_dataset": "EPF benchmark/epftoolbox",
        "data_type": "observed_open_benchmark", "time_granularity": "hourly",
        "reported_period_and_fields": "Reuses the five six-year epftoolbox datasets; target is 24 hourly day-ahead prices with two exogenous load/generation forecast series.",
    },
    "10.1016/j.apenergy.2024.124975": {
        "market_or_region": "Germany/EPEX continuous intraday",
        "data_provider_or_named_dataset": "EPEX SPOT; ENTSO-E and listed public/fundamental sources",
        "data_type": "observed_market_data", "time_granularity": "hourly products with 15-minute source covariates",
        "reported_period_and_fields": "2021--2022. Target: end-of-day IDFull index; inputs include live IDFull/ID1/ID3, day-ahead price, load and renewable forecasts, cross-border/fuel/weather/calendar variables.",
    },
    "10.3390/en17122909": {
        "market_or_region": "Denmark/DK1",
        "data_provider_or_named_dataset": "Nord Pool; Energinet; Energi Data Service",
        "data_type": "observed_market_data", "time_granularity": "hourly",
        "reported_period_and_fields": "2019-01-01--2019-04-30. Intraday and day-ahead prices, forecast wind/solar production, and forecast total consumption.",
    },
    "10.1016/j.apenergy.2025.126412": {
        "market_or_region": "Germany; Italian zones NOR/CNOR/CSOU/SOU/SARD/SICI",
        "data_provider_or_named_dataset": "Published German and Italian day-ahead research datasets; accompanying GitHub repository",
        "data_type": "observed_market_data", "time_granularity": "hourly",
        "reported_period_and_fields": "Germany 2015-01-01--2020-12-31; Italy 2015-10-01--2019-08-31. Price, load/renewable forecasts; German case also uses latest gas close.",
    },
    "10.1016/j.egyai.2025.100571": {
        "market_or_region": "Ireland/SEM",
        "data_provider_or_named_dataset": "O'Connor et al. dataset; paper's accompanying GitHub repository",
        "data_type": "observed_open_research_dataset", "time_granularity": "market settlement periods",
        "reported_period_and_fields": "2019--2022. Day-ahead and balancing prices plus TSO wind and demand forecasts; 24-step-ahead DAM case and balancing-market case.",
    },
    "10.1109/tsg.2022.3166791": {
        "market_or_region": "PJM",
        "data_provider_or_named_dataset": "PJM Data Miner",
        "data_type": "observed_market_data", "time_granularity": "hourly",
        "reported_period_and_fields": "Six years of actual electricity prices, temperature, and load; prediction is evaluated through an energy-storage arbitrage objective.",
    },
    "10.1016/j.eneco.2023.106602": {
        "market_or_region": "Nord Pool/Nordic; PJM; Belgium/EPEX-BE; France/EPEX-FR; Germany/EPEX-DE",
        "data_provider_or_named_dataset": "EPF benchmark/epftoolbox",
        "data_type": "observed_open_benchmark", "time_granularity": "hourly",
        "reported_period_and_fields": "Five Lago et al. day-ahead benchmark datasets; point forecasts are postprocessed into multivariate probabilistic paths using the Schaake shuffle.",
    },
    "10.1016/j.eneco.2023.106843": {
        "market_or_region": "Germany/EPEX-DE",
        "data_provider_or_named_dataset": "German day-ahead market research panel",
        "data_type": "observed_market_data", "time_granularity": "hourly",
        "reported_period_and_fields": "Six years of hourly German day-ahead prices and forecasting covariates for distributional neural-network evaluation.",
    },
    "10.1016/j.segan.2023.100996": {
        "market_or_region": "Belgium/EPEX-BE; Germany/EPEX-DE; France/EPEX-FR; Nord Pool/Nordic",
        "data_provider_or_named_dataset": "Four-market day-ahead research panel",
        "data_type": "observed_market_data", "time_granularity": "hourly",
        "reported_period_and_fields": "Day-ahead prices, hourly temperature, and weekday variables; cross-market samples are used for transfer learning.",
    },
    "10.1016/j.epsr.2026.112992": {
        "market_or_region": "Germany/EPEX-DE; France/EPEX-FR; Spain/OMIE/MIBEL",
        "data_provider_or_named_dataset": "EPEX/ENTSO-E and OMIE market panels",
        "data_type": "observed_market_data", "time_granularity": "hourly",
        "reported_period_and_fields": "2015-01-01--2024-12-31. Day-ahead prices with TSO load/renewable forecasts, used to test variance stabilization under volatile regimes.",
    },
}


def doi_filename(doi):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", doi) + ".txt"


def crossref_abstracts(path):
    if not path:
        return {}
    works = json.loads(path.read_text())
    return {
        work.get("DOI", "").lower(): html.unescape(re.sub(r"<[^>]+>", " ", work.get("abstract", "")))
        for work in works
    }


def semantic_abstracts(path):
    if not path:
        return {}
    result = {}
    for work in json.loads(path.read_text()):
        if not work:
            continue
        doi = ((work.get("externalIds") or {}).get("DOI") or "").lower()
        tldr = ((work.get("tldr") or {}).get("text") or "")
        result[doi] = " ".join([work.get("abstract") or "", tldr])
    return result


def data_context(text):
    reference = re.search(r"(?im)^\s*references\s*$", text)
    if reference:
        text = text[:reference.start()]
    heading = re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*)?\.?\s*"
        r"(?:data(?:set| sets?| description| sources?)?|case stud(?:y|ies)|"
        r"empirical (?:study|analysis)|numerical (?:study|results?))\s*$"
    )
    matches = list(heading.finditer(text))
    if matches:
        return " ".join(text[match.start():match.start() + 15000] for match in matches[:3])
    windows = []
    phrase = re.compile(
        r"(?i)data(?:set| set)?\s+(?:is|are|was|were|from|provided|obtained|collected)|"
        r"(?:obtained|collected|downloaded)\s+from|case stud"
    )
    for match in phrase.finditer(text):
        windows.append(text[max(0, match.start() - 800):match.start() + 2500])
    return " ".join(windows[:20])


def evidence_hits(patterns, title, abstract, context):
    hits = []
    for name, pattern in patterns.items():
        score = 5 * bool(re.search(pattern, title, re.I))
        score += 3 * bool(re.search(pattern, abstract, re.I))
        score += min(6, 2 * len(re.findall(pattern, context, re.I)))
        if score >= 2:
            hits.append((score, name))
    return [name for _, name in sorted(hits, reverse=True)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--crossref", type=Path)
    parser.add_argument("--semantic-scholar", type=Path)
    parser.add_argument("--full-text-dir", type=Path)
    args = parser.parse_args()
    with args.corpus.open(encoding="utf-8") as handle:
        corpus = list(csv.DictReader(handle))
    crossref = crossref_abstracts(args.crossref)
    semantic = semantic_abstracts(args.semantic_scholar)
    output = []
    for row in corpus:
        full_path = args.full_text_dir / doi_filename(row["doi"]) if args.full_text_dir else None
        full_text = full_path.read_text(errors="ignore") if full_path and full_path.exists() else ""
        abstract = " ".join([semantic.get(row["doi"], ""), crossref.get(row["doi"], "")])
        context = data_context(full_text) if full_text else ""
        markets = evidence_hits(MARKETS, row["title"], abstract, context)
        providers = evidence_hits(PROVIDERS, row["title"], abstract, context)
        combined = " ".join([row["title"], abstract, context])
        granularities = []
        for name, pattern in [
            ("5-minute", r"5[- ]minute"), ("15-minute", r"15[- ]minute|quarter-hour"),
            ("30-minute", r"30[- ]minute|half-hour"), ("hourly", r"hourly"),
            ("daily", r"\bdaily\b"), ("monthly", r"\bmonthly\b"),
        ]:
            if re.search(pattern, combined, re.I):
                granularities.append(name)
        if row["method_theme"] == "review_benchmark":
            data_type = "review_or_benchmark"
        elif re.search(r"simulat|synthetic|agent-based", combined, re.I):
            data_type = "simulation_or_synthetic"
        elif markets or providers:
            data_type = "observed_market_data"
        else:
            data_type = "not_identified"
        if full_text:
            level = "full_text"
            note = "Automated full-text extraction; verify exact dates and fields before citing."
        elif abstract.strip():
            level = "abstract"
            note = "No full text inspected; based on title/abstract."
        elif markets:
            level = "title"
            note = "Market identified from title only; provider and period remain unverified."
        else:
            level = "metadata_insufficient"
            note = "Available metadata does not disclose the dataset; full-text verification required."
        result = {
            "record_id": row["record_id"], "doi": row["doi"], "title": row["title"],
            "year": row["year"], "venue": row["venue"],
            "market_or_region": "; ".join(markets),
            "data_provider_or_named_dataset": "; ".join(providers),
            "data_type": data_type, "time_granularity": "; ".join(granularities),
            "reported_period_and_fields": "", "evidence_level": level,
            "verification_note": note,
        }
        if row["doi"] in MANUAL_OVERRIDES:
            result.update(MANUAL_OVERRIDES[row["doi"]])
            result["evidence_level"] = "full_text_manually_verified"
            result["verification_note"] = "Manually verified against the paper's data/case-study section."
        output.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {len(output)} records to {args.output}")


if __name__ == "__main__":
    main()
