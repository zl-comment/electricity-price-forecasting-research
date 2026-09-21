#!/usr/bin/env python3
"""Build the 2021--2026 electricity-market forecasting corpus from Crossref metadata.

The input is the deduplicated JSON array returned by the documented Crossref
queries in ELECTRICITY_MARKET_FORECASTING_REVIEW_2021_2026.md.  The script keeps
the curation rules explicit so that false positives and retractions do not enter
the review silently.
"""

import argparse
import csv
import html
import json
import re
from pathlib import Path


INCLUDED_VENUES = {
    "Applied Energy", "Energy", "Energy Economics", "Energy and AI",
    "Energy Strategy Reviews", "International Journal of Forecasting",
    "Journal of Forecasting", "Renewable and Sustainable Energy Reviews",
    "Renewable Energy", "Energy Reports", "Energies",
    "IEEE Transactions on Power Systems", "IEEE Transactions on Smart Grid",
    "IEEE Transactions on Sustainable Energy", "IEEE Power and Energy Magazine",
    "Electric Power Systems Research",
    "International Journal of Electrical Power & Energy Systems",
    "Sustainable Energy, Grids and Networks", "Expert Systems with Applications",
    "Engineering Applications of Artificial Intelligence",
    "Computers & Industrial Engineering", "Applied Soft Computing",
    "Scientific Reports", "IEEE Transactions on Industrial Informatics",
    "IEEE Transactions on Industry Applications",
    "IEEE Transactions on Consumer Electronics", "Knowledge-Based Systems",
    "Journal of Commodity Markets", "Technological Forecasting and Social Change",
    "Journal of Modern Power Systems and Clean Energy",
}

TOP_ANCHORS = {
    "Applied Energy", "International Journal of Forecasting",
    "Renewable and Sustainable Energy Reviews",
    "IEEE Transactions on Power Systems", "IEEE Transactions on Smart Grid",
    "IEEE Transactions on Sustainable Energy",
}

CORE_ENERGY_POWER = {
    "Energy", "Energy Economics", "Energy and AI", "Energy Strategy Reviews",
    "Renewable Energy", "Electric Power Systems Research",
    "International Journal of Electrical Power & Energy Systems",
    "Sustainable Energy, Grids and Networks", "IEEE Power and Energy Magazine",
    "IEEE Transactions on Industrial Informatics",
    "IEEE Transactions on Industry Applications",
    "Journal of Modern Power Systems and Clean Energy",
}

BROAD_ENERGY = {"Energies", "Energy Reports"}

EXCLUDED_DOIS = {
    # Retracted article and its notices.
    "10.1016/j.egyr.2021.04.009", "10.1016/j.egyr.2024.04.004",
    "10.1038/s41598-021-96501-6", "10.1038/s41598-022-06630-9",
    # False positives: another forecast target, rather than a market outcome.
    "10.3390/en18164433", "10.3390/en16093856", "10.3390/en15218057",
    "10.3390/en15197367", "10.1016/j.knosys.2025.114040",
    "10.1038/s41598-026-55296-0", "10.1016/j.engappai.2024.108125",
    "10.1002/for.3146",
}

CURATED_ADDITIONS = [
    {
        "year": 2021,
        "title": "A survey of electricity spot and futures price models for risk management applications",
        "authors": "Thomas Deschatre; Olivier Féron; Pierre Gruet",
        "venue": "Energy Economics", "publication_type": "journal-article",
        "doi": "10.1016/j.eneco.2021.105504", "crossref_cited_by": "",
        "notes": "Curated review; title does not use forecast/predict keyword.",
    },
    {
        "year": 2023,
        "title": "Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx",
        "authors": "Kin G. Olivares; Cristian Challu; Grzegorz Marcjasz; Rafał Weron; Artur Dubrawski",
        "venue": "International Journal of Forecasting", "publication_type": "journal-article",
        "doi": "10.1016/j.ijforecast.2022.03.001", "crossref_cited_by": "",
        "notes": "Curated addition from the existing repository literature library.",
    },
    {
        "year": 2024,
        "title": "Joint forecasting of source-load-price for integrated energy system based on multi-task learning and hybrid attention mechanism",
        "authors": "Li Ke; Mu Yuchen; Yang Fan; Wang Haiyang; Yan Yi; Zhang Chenghui",
        "venue": "Applied Energy", "publication_type": "journal-article",
        "doi": "10.1016/j.apenergy.2024.122821", "crossref_cited_by": "",
        "notes": "Adjacent joint energy-system forecast; not a cross-market reserve forecast.",
    },
    {
        "year": 2026,
        "title": "A novel framework for probabilistic forecasting of electricity forward curves",
        "authors": "Marina Dietze; Davi Valladão; Alexandre Street; Stein-Erik Fleten",
        "venue": "Energy Economics", "publication_type": "journal-article",
        "doi": "10.1016/j.eneco.2026.109293", "crossref_cited_by": "",
        "notes": "Curated online-first record missed by the ranked Crossref candidate query.",
    },
    {
        "year": 2026,
        "title": "Reasoning-enhanced probabilistic electricity price forecasting using parameter-efficient large language models",
        "authors": "Haoxuan Chen; Yinliang Xu; Wenchuan Wu; Hongbin Sun",
        "venue": "Applied Energy", "publication_type": "journal-article",
        "doi": "10.1016/j.apenergy.2026.128712", "crossref_cited_by": "",
        "notes": "Curated online-first record verified on the publisher page.",
    },
    {
        "year": 2021,
        "title": "A Deep Learning Forecaster with Exogenous Variables for Day-Ahead Locational Marginal Price",
        "authors": "Dipanwita Saha; Felipe Lopez",
        "venue": "IEEE Power & Energy Society General Meeting", "publication_type": "proceedings-article",
        "doi": "10.1109/pesgm46819.2021.9638128", "crossref_cited_by": "",
        "notes": "Leading domain conference.",
    },
    {
        "year": 2023,
        "title": "Interpretable Probabilistic Price Forecasting for Energy Markets",
        "authors": "Nandinee Haq; Kai Yuan; Xiaoming Feng",
        "venue": "IEEE Power & Energy Society General Meeting", "publication_type": "proceedings-article",
        "doi": "10.1109/pesgm52003.2023.10252531", "crossref_cited_by": "",
        "notes": "Leading domain conference.",
    },
    {
        "year": 2023,
        "title": "Exploring Models of Electricity Price Forecasting: Case Study on A FCAS Market",
        "authors": "Kenshiro Kato; Koki Iwabuchi; Daichi Watari; Dafang Zhao; Hiroki Nishikawa; Ittetsu Taniguchi; Takao Onoye",
        "venue": "ACM International Conference on Future Energy Systems (companion)",
        "publication_type": "proceedings-article",
        "doi": "10.1145/3599733.3600258", "crossref_cited_by": "",
        "notes": "Leading domain conference companion paper.",
    },
    {
        "year": 2024,
        "title": "Improved Grid Search Algorithm for Optimal LSTM Performance: A Case Study on Australian Electricity Price Forecasting",
        "authors": "Ibrahim Anwar Ibrahim",
        "venue": "IEEE Power & Energy Society General Meeting", "publication_type": "proceedings-article",
        "doi": "10.1109/pesgm51994.2024.10688849", "crossref_cited_by": "",
        "notes": "Leading domain conference.",
    },
    {
        "year": 2025,
        "title": "Multi-Task Decision-Oriented Electricity Price Prediction for Virtual Energy Storage Arbitrage",
        "authors": "Haoxuan Chen; Yinliang Xu; Hongbin Sun",
        "venue": "IEEE Power & Energy Society General Meeting", "publication_type": "proceedings-article",
        "doi": "10.1109/pesgm52009.2025.11225718", "crossref_cited_by": "",
        "notes": "Leading domain conference.",
    },
]


def clean(value):
    return html.unescape(value or "").replace("\t", " ").replace("\n", " ").strip()


def authors(item):
    names = []
    for author in item.get("author", []):
        name = " ".join(part for part in [author.get("given", ""), author.get("family", "")] if part)
        if name:
            names.append(clean(name))
    return "; ".join(names)


def outlet_tier(venue, publication_type):
    if publication_type == "proceedings-article":
        return "leading_domain_conference"
    if venue in TOP_ANCHORS:
        return "top_anchor"
    if venue in CORE_ENERGY_POWER:
        return "core_energy_power"
    if venue in BROAD_ENERGY:
        return "broad_energy"
    return "adjacent_forecasting_ai"


def market_target(title):
    t = title.lower()
    if re.search(r"balanc|imbalance|frequency reserve|afrr|fcas|ancillary|reserve", t):
        return "balancing_reserve_imbalance"
    if re.search(r"locational marginal|\blmp\b|nodal", t):
        return "lmp_nodal"
    if re.search(r"intraday|intra-day|real-time|half-hourly|high frequency", t):
        return "intraday_realtime"
    if re.search(r"forward|futures|long-term|long term|medium-term|medium term|mid-term|mid term|monthly", t):
        return "forward_medium_long_term"
    if re.search(r"day-ahead|day ahead", t):
        return "day_ahead"
    return "generic_spot_or_multi_market"


def forecast_output(title):
    t = title.lower()
    if re.search(r"probabil|quantile|density|interval|confidence|distribution|uncertainty|conformal|scenario", t):
        return "probabilistic"
    if re.search(r"spike|negative price|volatility|jump|tail risk|extreme", t):
        return "tail_event_or_volatility"
    if re.search(r"trading|arbitrage|offering|decision|profit|risk management|hedging", t):
        return "decision_or_economic_value"
    return "point_or_direction"


def method_theme(title):
    t = title.lower()
    if re.search(r"review|survey|bibliometric|state of the art|dawn of", t):
        return "review_benchmark"
    if re.search(r"foundation|large language|\bllm\b|\bgpt\b|chronos|pre-trained|zero-shot", t):
        return "foundation_or_language_model"
    if re.search(r"graph|spatial|hypergraph|cross-border|multi-price zone", t):
        return "spatial_graph_cross_market"
    if re.search(r"transformer|attention|lstm|gru|cnn|deep learning|neural|autoencoder|nbeats", t):
        return "deep_sequence_model"
    if re.search(r"decomposition|wavelet|vmd|ceemdan|spectrum|spectral|filtering|mode decomposition", t):
        return "decomposition_hybrid"
    if re.search(r"decision-focused|decision-oriented|trading|arbitrage|offering|agent-based|game", t):
        return "decision_focused_or_market_simulation"
    if re.search(r"random forest|xgboost|svr|support vector|extreme learning|elm|kernel|machine learning", t):
        return "classical_machine_learning"
    return "statistical_econometric_or_ensemble"


def make_row(item):
    title = clean((item.get("title") or [""])[0])
    venue = clean((item.get("container-title") or [""])[0])
    date_parts = item.get("published", {}).get("date-parts") or [[None]]
    doi = clean(item.get("DOI", "")).lower()
    publication_type = clean(item.get("type", "journal-article"))
    return {
        "year": date_parts[0][0], "title": title, "authors": authors(item),
        "venue": venue, "publication_type": publication_type,
        "outlet_tier": outlet_tier(venue, publication_type),
        "market_target": market_target(title), "forecast_output": forecast_output(title),
        "method_theme": method_theme(title), "doi": doi,
        "url": "https://doi.org/" + doi if doi else clean(item.get("URL", "")),
        "crossref_cited_by": item.get("is-referenced-by-count", ""),
        "notes": "Crossref cited-by count is a retrieval-time field, not a quality tier.",
    }


def relevant(item):
    title = clean((item.get("title") or [""])[0])
    venue = clean((item.get("container-title") or [""])[0])
    doi = clean(item.get("DOI", "")).lower()
    return (
        venue in INCLUDED_VENUES
        and doi not in EXCLUDED_DOIS
        and re.search(r"forecast|predict|nowcast", title, re.I)
        and re.search(r"price|pricing|LMP|locational marginal|spike|volatility|imbalance|balanc|reserve", title, re.I)
        and re.search(r"electric|power|energy|day.?ahead|intraday|spot market|wholesale", title, re.I)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    items = json.loads(args.input.read_text())
    rows = [make_row(item) for item in items if relevant(item)]
    for addition in CURATED_ADDITIONS:
        row = dict(addition)
        row["outlet_tier"] = outlet_tier(row["venue"], row["publication_type"])
        row["market_target"] = market_target(row["title"])
        row["forecast_output"] = forecast_output(row["title"])
        row["method_theme"] = method_theme(row["title"])
        row["url"] = "https://doi.org/" + row["doi"]
        rows.append(row)
    deduped = {row["doi"] or row["title"].casefold(): row for row in rows}
    rows = sorted(deduped.values(), key=lambda row: (int(row["year"]), row["venue"], row["title"]))
    for index, row in enumerate(rows, 1):
        row["record_id"] = f"EMF{index:03d}"
    fields = [
        "record_id", "year", "title", "authors", "venue", "publication_type",
        "outlet_tier", "market_target", "forecast_output", "method_theme", "doi",
        "url", "crossref_cited_by", "notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
