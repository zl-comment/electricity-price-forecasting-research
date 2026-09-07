"""Build the auditable paper-to-data and data-to-paper catalogs."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_CATALOG = ROOT / "paper/multi_market_energy_reserve/catalog.json"
DATA_DIR = ROOT / "data/multi_market_energy_reserve"

DATASETS = {
    "D01_DK1_MULTI_MARKET": {
        "name": "Energinet DK1 energy-aFRR-balancing snapshot",
        "status": "local_observed_market_data",
        "root_path": "data/multi_market_energy_reserve/energinet_dk1",
        "time_scope": "2025-10-01 inclusive to 2026-09-01 exclusive; settlement series ends earlier",
        "fields": ["day-ahead price", "aFRR capacity", "imbalance and activation", "wind/solar forecasts", "production and consumption"],
        "boundary": "No storage bids, awards, SOC, or device telemetry; storage operation is simulated.",
    },
    "D02_FINLAND_AFRR": {
        "name": "Finland aFRR energy market and weather data",
        "status": "local_open_paper_dataset",
        "root_path": "data/multi_market_energy_reserve/finland_afrr_zenodo_17494556",
        "time_scope": "2024-06-20 22:00 UTC to 2025-03-16 22:00 UTC in extended_data_v2.csv",
        "fields": ["aFRR up/down price", "spot price", "capacity price", "weather", "consumption"],
        "boundary": "Exact public dataset for F06; only partial same-market coverage for other Finnish studies.",
    },
    "D03_RTS_GMLC": {
        "name": "RTS-GMLC pinned power-system benchmark",
        "status": "local_synthetic_system_benchmark",
        "root_path": "data/multi_market_energy_reserve/rts_gmlc_system",
        "time_scope": "Pinned upstream commit 3ece0d3725c844056132393ee252b3083dd4eab4",
        "fields": ["network", "generators", "storage", "day-ahead and real-time time series"],
        "boundary": "Not operator history; clearing prices and shortages produced locally are simulation outputs.",
    },
    "D04_EPF_DE": {
        "name": "EPFtoolbox Germany day-ahead benchmark and author forecasts",
        "status": "local_open_paper_benchmark",
        "root_path": "data/multi_market_energy_reserve/epf_de_benchmark",
        "time_scope": "DE.csv: 2012-01-09 to 2017-12-31; author forecasts: canonical two-year test window",
        "fields": ["day-ahead price", "Amprion load forecast", "PV+wind forecast", "LEAR/DNN author forecasts"],
        "boundary": "Exact benchmark for F01; method-transfer benchmark for later architectures, not their original paper result reproduction.",
    },
}


def relation(dataset_id, relationship, note):
    return {"dataset_id": dataset_id, "relationship": relationship, "note": note}


ALIGNMENT = {
    "X01": ("original_market_data_not_released", [relation("D01_DK1_MULTI_MARKET", "same_market_family_not_exact_window", "Danish energy and ancillary-service fields; lacks the paper's device telemetry and 2022-2025 full window.")]),
    "X02": ("original_spanish_data_not_collected", [relation("D01_DK1_MULTI_MARKET", "current_project_main_benchmark", "Transfers the forecast-optimization architecture to a fully aligned DK1 market snapshot.")]),
    "X03": ("original_irish_data_not_collected", [relation("D01_DK1_MULTI_MARKET", "analogous_balancing_market_benchmark", "Supports day-ahead, imbalance, and activation uncertainty; not the Irish paper sample.")]),
    "X04": ("original_german_orderbook_data_not_publicly_reconstructed", [relation("D01_DK1_MULTI_MARKET", "multi_market_method_transfer", "Capacity and balancing fields support strategy transfer, but continuous German order books are absent.")]),
    "X05": ("paper_scenarios_not_packaged_as_public_dataset", [relation("D01_DK1_MULTI_MARKET", "observed_market_driver", "Observed energy and reserve series for robust scheduling."), relation("D03_RTS_GMLC", "mechanism_and_deliverability_benchmark", "Transparent system constraints for stress tests.")]),
    "F01": ("exact_public_benchmark_local", [relation("D04_EPF_DE", "exact_paper_dataset_and_author_forecasts", "DE input and canonical LEAR/DNN forecasts are pinned and checksum-verified.")]),
    "F02": ("original_multi_country_panel_not_fully_local", [relation("D04_EPF_DE", "compatible_epex_subset", "German EPEX subset supports a reduced market-integration baseline, not the full paper panel.")]),
    "F03": ("original_intraday_data_not_collected", [relation("D01_DK1_MULTI_MARKET", "analogous_intraday_balancing_inputs", "Use for hierarchical probabilistic transfer; market product differs.")]),
    "F04": ("review_no_single_original_dataset", [relation("D04_EPF_DE", "evaluation_protocol_benchmark", "Used to apply reliability and sharpness metrics on a public EPF benchmark.")]),
    "F05": ("original_two_market_data_not_fully_local", [relation("D04_EPF_DE", "probabilistic_method_benchmark", "Public EPF data for Bayesian uncertainty baselines.")]),
    "F06": ("exact_public_dataset_local", [relation("D02_FINLAND_AFRR", "exact_paper_dataset", "Zenodo dataset linked by the paper is stored locally.")]),
    "F07": ("original_trading_dataset_not_released_as_local_snapshot", [relation("D04_EPF_DE", "economic_value_evaluation_benchmark", "Tests whether statistical forecast ranking matches battery value on a fixed public series.")]),
    "F08": ("method_paper_no_electricity_dataset", [relation("D04_EPF_DE", "project_tree_baseline", "LightGBM is evaluated locally as a strong tree baseline; this is not a reproduction of the NeurIPS datasets.")]),
    "F09": ("paper_uses_public_epf_markets_partial_local", [relation("D04_EPF_DE", "paper_benchmark_subset", "The German EPFtoolbox market is a directly compatible subset of the multi-market NBEATSx study.")]),
    "F10": ("method_paper_benchmarks_not_mirrored", [relation("D04_EPF_DE", "project_exogenous_transformer_benchmark", "Price is endogenous; load and wind/solar forecasts are exogenous inputs.")]),
    "F11": ("method_paper_benchmarks_not_mirrored", [relation("D04_EPF_DE", "project_cross_correlation_benchmark", "Evaluates cross-correlation with known exogenous forecasts under the project protocol.")]),
    "F12": ("paper_benchmarks_not_mirrored", [relation("D04_EPF_DE", "project_interpretability_benchmark", "Tests prototype explanations on electricity price and exogenous covariates; not the paper's original result table.")]),
    "F13": ("foundation_model_pretraining_data_not_reproducible", [relation("D04_EPF_DE", "project_covariate_adaptation_benchmark", "Requires contamination audit before any zero-shot claim.")]),
    "F14": ("guangdong_original_data_not_publicly_available", [relation("D01_DK1_MULTI_MARKET", "method_transfer_only", "Spatial hypergraph inputs are not present; only selected temporal ideas can transfer.")]),
    "M01": ("paper_market_case_not_packaged_locally", [relation("D03_RTS_GMLC", "mechanism_reproduction_benchmark", "Supports transparent energy-reserve clearing and SOC-dependent bid experiments.")]),
    "M02": ("paper_uses_partly_restricted_nord_pool_data", [relation("D02_FINLAND_AFRR", "same_country_partial_market_fields", "Open Finnish aFRR fields are local; the paper's complete Nord Pool input is not."), relation("D01_DK1_MULTI_MARKET", "current_project_multi_market_transfer", "Provides a complete public energy-capacity-activation transfer setting.")]),
    "M03": ("original_case_data_not_fully_local", [relation("D01_DK1_MULTI_MARKET", "market_design_transfer_benchmark", "Tests asymmetric reserve products under an explicit DK1 rule mapping.")]),
    "M04": ("original_case_data_not_fully_local", [relation("D01_DK1_MULTI_MARKET", "stacked_revenue_transfer_benchmark", "Provides observed energy, capacity, and activation drivers.")]),
    "M05": ("original_case_data_not_packaged_locally", [relation("D03_RTS_GMLC", "scheduling_and_bidding_mechanism_benchmark", "Used for non-anticipativity and reserve-availability constraints.")]),
    "M06": ("paper_scenario_inputs_not_fully_local", [relation("D01_DK1_MULTI_MARKET", "day_ahead_afrr_transfer_benchmark", "Observed joint day-ahead and aFRR series."), relation("D04_EPF_DE", "energy_only_historical_reference", "Germany day-ahead prices only; no German aFRR counterpart.")]),
    "D01": ("device_experiment_data_not_released", [relation("D03_RTS_GMLC", "power_energy_budget_benchmark", "System-level simultaneous-service feasibility."), relation("D01_DK1_MULTI_MARKET", "activation_driver", "Observed activation drives simulated response.")]),
    "D02": ("device_telemetry_not_released", [relation("D01_DK1_MULTI_MARKET", "activation_driver", "Market activation input; converter behavior remains a model parameter." )]),
    "D03": ("experimental_project_data_not_fully_local", [relation("D03_RTS_GMLC", "planning_control_benchmark", "System planning layer."), relation("D01_DK1_MULTI_MARKET", "real_time_market_driver", "Observed activation for control stress tests.")]),
    "L01": ("paper_application_data_not_packaged_locally", [relation("D04_EPF_DE", "decision_focused_method_benchmark", "Public price forecasting benchmark for a storage task loss.")]),
    "L02": ("generic_method_paper_no_single_energy_dataset", [relation("D04_EPF_DE", "spo_method_benchmark", "Local predict-then-optimize benchmark only.")]),
    "L03": ("original_pjm_six_year_dataset_not_yet_local", [relation("D04_EPF_DE", "storage_arbitrage_transfer_benchmark", "Same day-ahead structure but not the paper's PJM sample.")]),
    "L04": ("original_strategic_market_data_not_packaged_locally", [relation("D04_EPF_DE", "strategic_storage_method_benchmark", "Requires a separately specified market-feedback model.")]),
    "L05": ("generic_method_paper_no_energy_dataset", [relation("D04_EPF_DE", "interpretable_spo_tree_benchmark", "Applies the method to EPF-derived decisions, not the ICML paper tasks.")]),
    "L06": ("original_new_york_market_sample_not_local", [relation("D04_EPF_DE", "predict_then_bid_transfer_benchmark", "Method-transfer dataset only; no claim of NYISO reproduction.")]),
}


def main():
    catalog = json.loads(PAPER_CATALOG.read_text(encoding="utf-8"))
    paper_ids = {item["id"] for item in catalog["items"]}
    if paper_ids != set(ALIGNMENT):
        raise ValueError(f"Paper alignment mismatch: missing={sorted(paper_ids-set(ALIGNMENT))}, extra={sorted(set(ALIGNMENT)-paper_ids)}")
    for dataset in DATASETS.values():
        if not (ROOT / dataset["root_path"]).exists():
            raise FileNotFoundError(dataset["root_path"])

    inverse = defaultdict(list)
    paper_rows = []
    for item in catalog["items"]:
        status, links = ALIGNMENT[item["id"]]
        for link in links:
            if link["dataset_id"] not in DATASETS:
                raise ValueError(f"Unknown dataset {link['dataset_id']} for {item['id']}")
            inverse[link["dataset_id"]].append(item["id"])
        item["original_data_status"] = status
        item["data_links"] = links
        paper_rows.append(
            {
                "paper_id": item["id"],
                "title": item["title"],
                "original_data_status": status,
                "data_links": links,
            }
        )

    catalog["data_alignment"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "Every paper has an explicit original-data status and at least one local dataset relation; relation labels distinguish exact datasets from transfer or mechanism benchmarks.",
        "papers_with_local_relation": len(paper_rows),
        "papers_with_exact_public_dataset_local": sum("exact_public" in row["original_data_status"] for row in paper_rows),
        "relations": sum(len(row["data_links"]) for row in paper_rows),
    }
    PAPER_CATALOG.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    data_catalog = {
        "research_cutoff": catalog["research_cutoff"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "Datasets remain source-centric and are not duplicated per paper; paper_ids provide the reverse index.",
        "counts": {"datasets": len(DATASETS), "papers": len(paper_rows), "relations": sum(len(row["data_links"]) for row in paper_rows)},
        "datasets": [dict({"id": key}, **value, paper_ids=sorted(inverse[key])) for key, value in DATASETS.items()],
    }
    (DATA_DIR / "catalog.json").write_text(json.dumps(data_catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (DATA_DIR / "paper_data_alignment.json").write_text(
        json.dumps({"research_cutoff": catalog["research_cutoff"], "papers": paper_rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"papers": len(paper_rows), "datasets": len(DATASETS), "relations": sum(len(row["data_links"]) for row in paper_rows)}))


if __name__ == "__main__":
    main()
