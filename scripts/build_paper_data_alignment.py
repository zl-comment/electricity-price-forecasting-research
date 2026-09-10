"""Build the auditable paper-to-data and data-to-paper catalogs."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_CATALOG = ROOT / "paper/multi_market_energy_reserve/catalog.json"
DATA_DIR = ROOT / "data/multi_market_energy_reserve"

DATASETS = {
    "DS1_DK1_MULTI_MARKET": {
        "name": "Energinet DK1 energy-aFRR-balancing snapshot",
        "status": "local_observed_market_data",
        "root_path": "data/multi_market_energy_reserve/energinet_dk1",
        "time_scope": "2025-10-01 inclusive to 2026-09-01 exclusive; settlement series ends earlier",
        "fields": ["day-ahead price", "aFRR capacity", "imbalance and activation", "wind/solar forecasts", "production and consumption"],
        "boundary": "No storage bids, awards, SOC, or device telemetry; storage operation is simulated.",
    },
    "DS2_FINLAND_AFRR": {
        "name": "Finland aFRR energy market and weather data",
        "status": "local_open_paper_dataset",
        "root_path": "data/multi_market_energy_reserve/finland_afrr_zenodo_17494556",
        "time_scope": "2024-06-20 22:00 UTC to 2025-03-16 22:00 UTC in extended_data_v2.csv",
        "fields": ["aFRR up/down price", "spot price", "capacity price", "weather", "consumption"],
        "boundary": "Exact public dataset for F06; only partial same-market coverage for other Finnish studies.",
    },
    "DS3_RTS_GMLC": {
        "name": "RTS-GMLC pinned power-system benchmark",
        "status": "local_synthetic_system_benchmark",
        "root_path": "data/multi_market_energy_reserve/rts_gmlc_system",
        "time_scope": "Pinned upstream commit 3ece0d3725c844056132393ee252b3083dd4eab4",
        "fields": ["network", "generators", "storage", "day-ahead and real-time time series"],
        "boundary": "Not operator history; clearing prices and shortages produced locally are simulation outputs.",
    },
    "DS4_EPF_DE": {
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
    "X01": ("original_market_data_not_released", [relation("DS1_DK1_MULTI_MARKET", "same_market_family_not_exact_window", "Danish energy and ancillary-service fields; lacks the paper's device telemetry and 2022-2025 full window.")]),
    "X02": ("original_spanish_data_not_collected", [relation("DS1_DK1_MULTI_MARKET", "current_project_main_benchmark", "Transfers the forecast-optimization architecture to a fully aligned DK1 market snapshot.")]),
    "X03": ("original_irish_data_not_collected", [relation("DS1_DK1_MULTI_MARKET", "analogous_balancing_market_benchmark", "Supports day-ahead, imbalance, and activation uncertainty; not the Irish paper sample.")]),
    "X04": ("original_german_orderbook_data_not_publicly_reconstructed", [relation("DS1_DK1_MULTI_MARKET", "multi_market_method_transfer", "Capacity and balancing fields support strategy transfer, but continuous German order books are absent.")]),
    "X05": ("paper_scenarios_not_packaged_as_public_dataset", [relation("DS1_DK1_MULTI_MARKET", "observed_market_driver", "Observed energy and reserve series for robust scheduling."), relation("DS3_RTS_GMLC", "mechanism_and_deliverability_benchmark", "Transparent system constraints for stress tests.")]),
    "X06": ("original_belgian_market_sample_not_local", [relation("DS1_DK1_MULTI_MARKET", "probabilistic_delivery_method_transfer", "DK1 energy, capacity, and activation labels support a same-problem transfer under different rules.")]),
    "X07": ("original_market_and_training_sample_unverified", [relation("DS1_DK1_MULTI_MARKET", "joint_scenario_method_transfer", "DK1 provides the three target families needed to test the verified Copula scenario structure.")]),
    "X08": ("original_wind_and_market_sample_not_publicly_reconstructed", [relation("DS1_DK1_MULTI_MARKET", "forecast_stochastic_method_transfer", "Observed DK1 market labels replace the paper's unspecified market sample; wind-plant data are not mirrored.")]),
    "X09": ("original_spanish_sample_not_packaged_locally", [relation("DS1_DK1_MULTI_MARKET", "sequential_stochastic_method_transfer", "Transfers the open sequential scenario formulation to the frozen DK1 information set.")]),
    "X10": ("original_spanish_inputs_confidential", [relation("DS1_DK1_MULTI_MARKET", "wind_battery_scheduling_transfer", "Observed DK1 prices and activation support a reduced storage-only transfer, not the paper's wind case.")]),
    "X11": ("original_italian_pv_and_frequency_sample_not_local", [relation("DS1_DK1_MULTI_MARKET", "probabilistic_planning_transfer", "DK1 activation can drive a product-adjusted planning and control test; PV telemetry is absent.")]),
    "X12": ("paper_code_public_but_case_inputs_not_mirrored", [relation("DS1_DK1_MULTI_MARKET", "independent_scenario_baseline_transfer", "DK1 day-ahead and aFRR prices can replace the German case after removing heat-load assumptions.")]),
    "X13": ("original_belgian_market_sample_not_local", [relation("DS1_DK1_MULTI_MARKET", "delivery_guarantee_transfer", "DK1 activation volumes support re-estimating delivery-risk constraints under the local protocol.")]),
    "X14": ("original_german_case_not_packaged_locally", [relation("DS1_DK1_MULTI_MARKET", "three_stage_bidding_transfer", "DK1 exposes analogous day-ahead, aFRR capacity, and activation labels; the German sample is absent.")]),
    "X15": ("original_nordic_fcr_samples_not_local", [relation("DS1_DK1_MULTI_MARKET", "nordic_multi_market_transfer", "The local DK1 snapshot replaces older Nordic FCR cases with the post-change aFRR product.")]),
    "X16": ("original_122_day_case_not_publicly_identified", [relation("DS1_DK1_MULTI_MARKET", "forecast_uncertainty_scheduling_transfer", "DK1 market labels support a reduced energy-reserve transfer; load and PV inputs are absent.")]),
    "X17": ("original_german_market_sample_not_local", [relation("DS1_DK1_MULTI_MARKET", "joint_price_dynamics_transfer", "Tests whether sequential price dependence transfers after replacing FCR with DK1 aFRR.")]),
    "X18": ("original_spanish_market_sample_not_local", [relation("DS1_DK1_MULTI_MARKET", "rolling_horizon_transfer", "Observed DK1 energy, capacity, and activation replace the Spanish market stack.")]),
    "X19": ("original_belgian_reserve_sample_not_local", [relation("DS1_DK1_MULTI_MARKET", "decision_focused_reserve_transfer", "DK1 capacity price, activation price, and activation volume provide the required reserve targets without reproducing the paper sample.")]),
    "X20": ("original_german_and_swiss_samples_not_local", [relation("DS1_DK1_MULTI_MARKET", "decision_value_evaluation_transfer", "Uses DK1 to test the paper's forecast-ranking versus decision-ranking diagnostic under one protocol.")]),
    "X21": ("original_swiss_building_and_device_data_not_local", [relation("DS1_DK1_MULTI_MARKET", "stacked_control_transfer", "DK1 aFRR labels can drive the control chain, while building and device telemetry remain unavailable.")]),
    "X22": ("original_danish_wind_and_fcr_sample_not_local", [relation("DS1_DK1_MULTI_MARKET", "same_country_scenario_transfer", "Same-country market data support method transfer after replacing FCR-N and the investment layer with DK1 aFRR.")]),
    "F01": ("exact_public_benchmark_local", [relation("DS4_EPF_DE", "exact_paper_dataset_and_author_forecasts", "DE input and canonical LEAR/DNN forecasts are pinned and checksum-verified.")]),
    "F02": ("original_multi_country_panel_not_fully_local", [relation("DS4_EPF_DE", "compatible_epex_subset", "German EPEX subset supports a reduced market-integration baseline, not the full paper panel.")]),
    "F03": ("original_intraday_data_not_collected", [relation("DS1_DK1_MULTI_MARKET", "analogous_intraday_balancing_inputs", "Use for hierarchical probabilistic transfer; market product differs.")]),
    "F04": ("review_no_single_original_dataset", [relation("DS4_EPF_DE", "evaluation_protocol_benchmark", "Used to apply reliability and sharpness metrics on a public EPF benchmark.")]),
    "F05": ("original_two_market_data_not_fully_local", [relation("DS4_EPF_DE", "probabilistic_method_benchmark", "Public EPF data for Bayesian uncertainty baselines.")]),
    "F06": ("exact_public_dataset_local", [relation("DS2_FINLAND_AFRR", "exact_paper_dataset", "Zenodo dataset linked by the paper is stored locally.")]),
    "F07": ("original_trading_dataset_not_released_as_local_snapshot", [relation("DS4_EPF_DE", "economic_value_evaluation_benchmark", "Tests whether statistical forecast ranking matches battery value on a fixed public series.")]),
    "F08": ("method_paper_no_electricity_dataset", [relation("DS4_EPF_DE", "project_tree_baseline", "LightGBM is evaluated locally as a strong tree baseline; this is not a reproduction of the NeurIPS datasets.")]),
    "F09": ("paper_uses_public_epf_markets_partial_local", [relation("DS4_EPF_DE", "paper_benchmark_subset", "The German EPFtoolbox market is a directly compatible subset of the multi-market NBEATSx study.")]),
    "F10": ("method_paper_benchmarks_not_mirrored", [relation("DS4_EPF_DE", "project_exogenous_transformer_benchmark", "Price is endogenous; load and wind/solar forecasts are exogenous inputs.")]),
    "F11": ("method_paper_benchmarks_not_mirrored", [relation("DS4_EPF_DE", "project_cross_correlation_benchmark", "Evaluates cross-correlation with known exogenous forecasts under the project protocol.")]),
    "F12": ("paper_benchmarks_not_mirrored", [relation("DS4_EPF_DE", "project_interpretability_benchmark", "Tests prototype explanations on electricity price and exogenous covariates; not the paper's original result table.")]),
    "F13": ("foundation_model_pretraining_data_not_reproducible", [relation("DS4_EPF_DE", "project_covariate_adaptation_benchmark", "Requires contamination audit before any zero-shot claim.")]),
    "F14": ("guangdong_original_data_not_publicly_available", [relation("DS1_DK1_MULTI_MARKET", "method_transfer_only", "Spatial hypergraph inputs are not present; only selected temporal ideas can transfer.")]),
    "F15": ("original_integrated_energy_park_data_not_publicly_available", [relation("DS1_DK1_MULTI_MARKET", "multi_task_method_transfer_only", "Shared-layer multi-task structure transfers; the paper couples heat and gas loads, not day-ahead versus aFRR market functions.")]),
    "F16": ("original_nord_pool_panel_and_model_inputs_not_local", [relation("DS1_DK1_MULTI_MARKET", "same_market_family_partial_zones", "The local day-ahead file carries DK1 and five neighbouring zones; the paper's full Nord Pool panel is not local, and whether DK1 is in its sample is unverified.")]),
    "F17": ("original_dk1_intraday_inputs_not_packaged_locally", [relation("DS1_DK1_MULTI_MARKET", "same_market_partial_fields", "Same DK1 bidding zone with day-ahead prices; the paper's intraday price series is not in the local snapshot, and its study window predates it.")]),
    "M01": ("paper_market_case_not_packaged_locally", [relation("DS3_RTS_GMLC", "mechanism_reproduction_benchmark", "Supports transparent energy-reserve clearing and SOC-dependent bid experiments.")]),
    "M02": ("paper_uses_partly_restricted_nord_pool_data", [relation("DS2_FINLAND_AFRR", "same_country_partial_market_fields", "Open Finnish aFRR fields are local; the paper's complete Nord Pool input is not."), relation("DS1_DK1_MULTI_MARKET", "current_project_multi_market_transfer", "Provides a complete public energy-capacity-activation transfer setting.")]),
    "M03": ("original_case_data_not_fully_local", [relation("DS1_DK1_MULTI_MARKET", "market_design_transfer_benchmark", "Tests asymmetric reserve products under an explicit DK1 rule mapping.")]),
    "M04": ("original_case_data_not_fully_local", [relation("DS1_DK1_MULTI_MARKET", "stacked_revenue_transfer_benchmark", "Provides observed energy, capacity, and activation drivers.")]),
    "M05": ("original_case_data_not_packaged_locally", [relation("DS3_RTS_GMLC", "scheduling_and_bidding_mechanism_benchmark", "Used for non-anticipativity and reserve-availability constraints.")]),
    "M06": ("paper_scenario_inputs_not_fully_local", [relation("DS1_DK1_MULTI_MARKET", "day_ahead_afrr_transfer_benchmark", "Observed joint day-ahead and aFRR series."), relation("DS4_EPF_DE", "energy_only_historical_reference", "Germany day-ahead prices only; no German aFRR counterpart.")]),
    "M07": ("original_german_market_and_forecast_inputs_not_released", [relation("DS1_DK1_MULTI_MARKET", "sequential_market_baseline_transfer", "Observed day-ahead, aFRR capacity and imbalance series carry the gate-closure sequence; the paper's German order-book inputs are absent.")]),
    "M08": ("original_danish_case_inputs_not_packaged_locally", [relation("DS1_DK1_MULTI_MARKET", "day_ahead_afrr_transfer_benchmark", "Observed Danish day-ahead and aFRR series allow a same-market transfer; the paper's wind-plant and storage case inputs are not local, and its abstract does not name the bidding zone.")]),
    "D01": ("device_experiment_data_not_released", [relation("DS3_RTS_GMLC", "power_energy_budget_benchmark", "System-level simultaneous-service feasibility."), relation("DS1_DK1_MULTI_MARKET", "activation_driver", "Observed activation drives simulated response.")]),
    "D02": ("device_telemetry_not_released", [relation("DS1_DK1_MULTI_MARKET", "activation_driver", "Market activation input; converter behavior remains a model parameter." )]),
    "D03": ("experimental_project_data_not_fully_local", [relation("DS3_RTS_GMLC", "planning_control_benchmark", "System planning layer."), relation("DS1_DK1_MULTI_MARKET", "real_time_market_driver", "Observed activation for control stress tests.")]),
    "L01": ("paper_application_data_not_packaged_locally", [relation("DS4_EPF_DE", "decision_focused_method_benchmark", "Public price forecasting benchmark for a storage task loss.")]),
    "L02": ("generic_method_paper_no_single_energy_dataset", [relation("DS4_EPF_DE", "spo_method_benchmark", "Local predict-then-optimize benchmark only.")]),
    "L03": ("original_pjm_six_year_dataset_not_yet_local", [relation("DS4_EPF_DE", "storage_arbitrage_transfer_benchmark", "Same day-ahead structure but not the paper's PJM sample.")]),
    "L04": ("original_strategic_market_data_not_packaged_locally", [relation("DS4_EPF_DE", "strategic_storage_method_benchmark", "Requires a separately specified market-feedback model.")]),
    "L05": ("generic_method_paper_no_energy_dataset", [relation("DS4_EPF_DE", "interpretable_spo_tree_benchmark", "Applies the method to EPF-derived decisions, not the ICML paper tasks.")]),
    "L06": ("original_new_york_market_sample_not_local", [relation("DS4_EPF_DE", "predict_then_bid_transfer_benchmark", "Method-transfer dataset only; no claim of NYISO reproduction.")]),
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
