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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-09-06"
PAPER_DIR = ROOT / "paper/multi_market_energy_reserve"
DATA_DIR = ROOT / "data/multi_market_energy_reserve"
USER_AGENT = "energy-reserve-multi-market-research/1.0"

PAPERS = [
    {
        "id": "X01", "title": "Optimizing Multi-Market Participation of Battery and Electrolyser Systems Based on Field Performance",
        "date": "2026-08-17", "doi": "10.48550/arXiv.2608.16238", "venue": "arXiv",
        "publication_status": "preprint; no formal venue verified by the research cutoff", "quality_tier": "preprint", "primary_problem": "intersection_frontier",
        "reported_venue": "IEEE PES General Meeting 2026 (author-reported in arXiv comments)",
        "verification_note": "No official proceedings record was independently verified by the research cutoff, so the library keeps the item in the preprint tier.",
        "version": "arXiv v1", "file": "01_intersection_frontier/03_preprints/X01_2026_Field_Performance_Multi_Market_BESS.pdf",
        "url": "https://arxiv.org/pdf/2608.16238v1",
    },
    {
        "id": "X02", "title": "An Integrated Forecasting and Optimization Framework for Battery Energy Storage Participation in Spanish Electricity Markets",
        "date": "2026-07-29", "doi": "10.32604/ee.2026.084810", "venue": "Energy Engineering",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "version": "publisher PDF, CC BY 4.0", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X02_2026_Integrated_Forecasting_Spanish_Markets.pdf",
        "url": "https://www.techscience.com/energy/online/detail/27739/pdf",
    },
    {
        "id": "X03", "title": "Conformal Prediction for Electricity Price Forecasting in the Day-Ahead and Real-Time Balancing Market",
        "date": "2025-09", "doi": "10.1016/j.egyai.2025.100571", "venue": "Energy and AI",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "version": "arXiv v1 author manuscript; final article DOI recorded", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X03_2025_Conformal_DA_Balancing_Forecasting.pdf",
        "url": "https://arxiv.org/pdf/2502.04935v1",
    },
    {
        "id": "X04", "title": "Joint Bidding on Intraday and Frequency Containment Reserve Markets",
        "date": "2025-10-03", "doi": "10.48550/arXiv.2510.03209", "venue": "arXiv/SSRN",
        "publication_status": "preprint", "quality_tier": "preprint", "primary_problem": "intersection_frontier",
        "version": "arXiv v1", "file": "01_intersection_frontier/03_preprints/X04_2025_Joint_Intraday_FCR_Bidding.pdf",
        "url": "https://arxiv.org/pdf/2510.03209v1",
    },
    {
        "id": "X05", "title": "Storage Participation in Electricity Markets: Time Discretization through Robust Optimization",
        "date": "2026-05-11 revision", "doi": "10.48550/arXiv.2510.10856", "venue": "Optimization Online/arXiv",
        "publication_status": "preprint under review", "quality_tier": "preprint", "primary_problem": "intersection_frontier",
        "version": "arXiv v2", "file": "01_intersection_frontier/03_preprints/X05_2026_Storage_Energy_Ancillary_Services.pdf",
        "url": "https://arxiv.org/pdf/2510.10856v2",
    },
    {
        "id": "F01", "title": "Forecasting Day-Ahead Electricity Prices: A Review of State-of-the-Art Algorithms, Best Practices and an Open-Access Benchmark",
        "date": "2021", "doi": "10.1016/j.apenergy.2021.116983", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv author manuscript corresponding to final article", "file": "02_partial_forecasting_uncertainty/01_top_journals/F01_2021_Applied_Energy_EPF_Benchmark.pdf",
        "url": "https://arxiv.org/pdf/2008.08004",
    },
    {
        "id": "F02", "title": "Forecasting Day-Ahead Electricity Prices in Europe: The Importance of Considering Market Integration",
        "date": "2018", "doi": "10.1016/j.apenergy.2017.11.098", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv author manuscript corresponding to final article", "file": "02_partial_forecasting_uncertainty/01_top_journals/F02_2018_Applied_Energy_Market_Integration_Forecasting.pdf",
        "url": "https://arxiv.org/pdf/1708.07061",
    },
    {
        "id": "F03", "title": "Bayesian Hierarchical Probabilistic Forecasting of Intraday Electricity Prices",
        "date": "2025", "doi": "10.1016/j.apenergy.2024.124975", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "version": "institutional open-access publisher PDF", "file": "02_partial_forecasting_uncertainty/01_top_journals/F03_2025_Applied_Energy_Bayesian_Intraday_Forecasting.pdf",
        "url": "https://opus.bibliothek.uni-augsburg.de/opus4/files/117511/1-s2.0-S0306261924023596-main.pdf",
    },
    {
        "id": "F05", "title": "Bayesian Deep Learning Based Method for Probabilistic Forecast of Day-Ahead Electricity Prices",
        "date": "2019", "doi": "10.1016/j.apenergy.2019.05.068", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "version": "institutional accepted manuscript", "file": "02_partial_forecasting_uncertainty/01_top_journals/F05_2019_Applied_Energy_Bayesian_DL_Price_Forecasting.pdf",
        "url": "https://re.public.polimi.it/bitstream/11311/1120105/2/11311-1120105_Matteucci.pdf",
    },
    {
        "id": "F07", "title": "Probabilistic Forecasting for Day-ahead Electricity Prices, Battery Trading Strategies and the Economic Evaluation of Predictive Accuracy",
        "date": "2026-04-21", "doi": "10.48550/arXiv.2604.19580", "venue": "arXiv",
        "publication_status": "preprint", "quality_tier": "preprint", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v1", "file": "02_partial_forecasting_uncertainty/03_preprints/F07_2026_Probabilistic_Forecasting_Battery_Economic_Value.pdf",
        "url": "https://arxiv.org/pdf/2604.19580v1",
    },
    {
        "id": "M01", "title": "Multi-Interval Energy-Reserve Co-Optimization with SoC-Dependent Bids from Battery Storage",
        "date": "2025", "doi": "10.1109/TPWRS.2024.3509913", "venue": "IEEE Transactions on Power Systems",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "energy_reserve_decision",
        "version": "arXiv v2 author manuscript corresponding to final article", "file": "03_partial_energy_reserve_decision/01_top_journals/M01_2025_TPWRS_Energy_Reserve_Cooptimization_SoC_Bids.pdf",
        "url": "https://arxiv.org/pdf/2401.15525v2",
    },
    {
        "id": "M03", "title": "BESS and the Ancillary Services Markets: A Symbiosis Yet? Impact of Market Design on Performance",
        "date": "2024", "doi": "10.1016/j.apenergy.2024.124153", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "energy_reserve_decision",
        "version": "institutional open-access publisher PDF", "file": "03_partial_energy_reserve_decision/01_top_journals/M03_2024_Applied_Energy_BESS_Ancillary_Market_Design.pdf",
        "url": "https://re.public.polimi.it/bitstream/11311/1271264/1/1-s2.0-S0306261924015368-main.pdf",
    },
    {
        "id": "M04", "title": "Stacked Revenues for Energy Storage Participating in Energy and Reserve Markets with an Optimal Frequency Regulation Modeling",
        "date": "2023", "doi": "10.1016/j.apenergy.2023.121721", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "energy_reserve_decision",
        "version": "HAL open-access author manuscript", "file": "03_partial_energy_reserve_decision/01_top_journals/M04_2023_Applied_Energy_Stacked_Revenues.pdf",
        "url": "https://hal.science/hal-04182119/document",
    },
    {
        "id": "M06", "title": "Economic Evaluation of Battery Storage Systems Bidding on Day-Ahead and Automatic Frequency Restoration Reserves Markets",
        "date": "2021", "doi": "10.1016/j.apenergy.2021.117267", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "energy_reserve_decision",
        "version": "institutional open-access publisher PDF", "file": "03_partial_energy_reserve_decision/01_top_journals/M06_2021_Applied_Energy_DA_aFRR_Economic_Evaluation.pdf",
        "url": "https://www.ee.ruhr-uni-bochum.de/ee/mam/2021_apen_nitschetal_economic_evaluation_of_battery_storage_systems_bidding_da_and_afrr_markets.pdf",
    },
    {
        "id": "M07", "title": "Data-Driven Sequential Market Optimization for Front-of-the-Meter Battery Energy Storage Systems",
        "date": "2026-07-27", "doi": "10.48550/arXiv.2607.24075", "venue": "arXiv",
        "publication_status": "preprint; no formal venue verified by the research cutoff", "quality_tier": "preprint", "primary_problem": "energy_reserve_decision",
        "version": "arXiv v1", "file": "03_partial_energy_reserve_decision/03_preprints/M07_2026_Sequential_Market_Optimization_FTM_BESS.pdf",
        "url": "https://arxiv.org/pdf/2607.24075v1",
    },
    {
        "id": "D01", "title": "Control of Battery Storage Systems for the Simultaneous Provision of Multiple Services",
        "date": "2019", "doi": "10.1109/TSG.2018.2810781", "venue": "IEEE Transactions on Smart Grid",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "deliverability_control",
        "version": "arXiv author manuscript corresponding to final article", "file": "04_partial_deliverability_control/01_top_journals/D01_2019_TSG_Simultaneous_Multiple_Services.pdf",
        "url": "https://arxiv.org/pdf/1803.00978",
    },
    {
        "id": "D02", "title": "Real-Time Control of Battery Energy Storage Systems to Provide Ancillary Services Considering Voltage-Dependent Capability of DC-AC Converters",
        "date": "2021", "doi": "10.1109/TSG.2021.3077696", "venue": "IEEE Transactions on Smart Grid",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "deliverability_control",
        "version": "EPFL institutional open-access manuscript", "file": "04_partial_deliverability_control/01_top_journals/D02_2021_TSG_Realtime_BESS_Ancillary_Control.pdf",
        "url": "https://infoscience.epfl.ch/record/285469/files/Real-timeControlofBatteryEnergyStorageSystemstoProvideAncillaryServicesConsideringVoltage-DependentCapabilityofDC-ACConverters.pdf",
    },
    {
        "id": "D03", "title": "Multi-Service Battery Energy Storage System Optimization and Control",
        "date": "2022", "doi": "10.1016/j.apenergy.2022.118614", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "deliverability_control",
        "version": "OSTI open-access manuscript", "file": "04_partial_deliverability_control/01_top_journals/D03_2022_Applied_Energy_Multi_Service_Optimization_Control.pdf",
        "url": "https://www.osti.gov/servlets/purl/1855830",
    },
    {
        "id": "L01", "title": "Task-Based End-to-End Model Learning in Stochastic Optimization",
        "date": "2017", "doi": "10.5555/3294996.3295052", "venue": "NeurIPS 2017",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "decision_focused_learning",
        "version": "official NeurIPS proceedings PDF", "file": "05_partial_decision_focused_learning/01_top_journal_or_conference/L01_2017_NeurIPS_Task_Based_End_to_End.pdf",
        "url": "https://proceedings.neurips.cc/paper/2017/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf",
    },
    {
        "id": "L02", "title": "Smart Predict, then Optimize",
        "date": "2022", "doi": "10.1287/mnsc.2020.3922", "venue": "Management Science",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "decision_focused_learning",
        "version": "arXiv author manuscript corresponding to final article", "file": "05_partial_decision_focused_learning/01_top_journal_or_conference/L02_2022_Management_Science_Smart_Predict_Then_Optimize.pdf",
        "url": "https://arxiv.org/pdf/1710.08005",
    },
    {
        "id": "L03", "title": "Electricity Price Prediction for Energy Storage System Arbitrage: A Decision-Focused Approach",
        "date": "2022", "doi": "10.1109/TSG.2022.3166791", "venue": "IEEE Transactions on Smart Grid",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "decision_focused_learning",
        "version": "arXiv author manuscript corresponding to final article", "file": "05_partial_decision_focused_learning/01_top_journal_or_conference/L03_2022_TSG_Decision_Focused_Storage_Arbitrage.pdf",
        "url": "https://arxiv.org/pdf/2305.00362",
    },
    {
        "id": "L04", "title": "Perturbed Decision-Focused Learning for Modeling Strategic Energy Storage",
        "date": "2025", "doi": "10.1109/TSG.2025.3548009", "venue": "IEEE Transactions on Smart Grid",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "decision_focused_learning",
        "version": "arXiv author manuscript corresponding to final article", "file": "05_partial_decision_focused_learning/01_top_journal_or_conference/L04_2025_TSG_Perturbed_DFL_Strategic_Storage.pdf",
        "url": "https://arxiv.org/pdf/2406.17085",
    },
    {
        "id": "L05", "title": "Decision Trees for Decision-Making under the Predict-then-Optimize Framework",
        "date": "2020", "doi": "10.5555/3524938.3525204", "venue": "ICML 2020",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "decision_focused_learning",
        "version": "official PMLR proceedings PDF", "file": "05_partial_decision_focused_learning/01_top_journal_or_conference/L05_2020_ICML_Decision_Trees_Predict_Optimize.pdf",
        "url": "https://proceedings.mlr.press/v119/elmachtoub20a/elmachtoub20a.pdf",
    },
    {
        "id": "L06", "title": "A Decision-Focused Predict-then-Bid Framework for Strategic Energy Storage",
        "date": "2025-05-06 revision", "doi": "10.48550/arXiv.2505.01551", "venue": "arXiv",
        "publication_status": "preprint", "quality_tier": "preprint", "primary_problem": "decision_focused_learning",
        "version": "arXiv v2", "file": "05_partial_decision_focused_learning/03_preprints/L06_2025_Decision_Focused_Predict_Then_Bid.pdf",
        "url": "https://arxiv.org/pdf/2505.01551v2",
    },
    {
        "id": "F08", "title": "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
        "date": "2017", "doi": None, "venue": "NeurIPS 2017",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "official NeurIPS proceedings PDF migrated from the prior EPF library", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F08_2017_NeurIPS_LightGBM.pdf",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf",
        "migrated_from": "zl-comment/epf-frontier-study@8c6b0ee720c60ed85bed6f138201228951ca4c76",
    },
    {
        "id": "F09", "title": "Neural Basis Expansion Analysis with Exogenous Variables: Forecasting Electricity Prices with NBEATSx",
        "date": "2023", "doi": "10.1016/j.ijforecast.2022.03.001", "venue": "International Journal of Forecasting",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v6 author manuscript corresponding to final article; migrated from prior EPF library", "file": "02_partial_forecasting_uncertainty/01_top_journals/F09_2023_IJF_NBEATSx_EPF.pdf",
        "url": "https://arxiv.org/pdf/2104.05522v6",
        "migrated_from": "zl-comment/epf-frontier-study@8c6b0ee720c60ed85bed6f138201228951ca4c76",
    },
    {
        "id": "F10", "title": "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables",
        "date": "2024", "doi": "10.52202/079017-0015", "venue": "NeurIPS 2024",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v4 author manuscript corresponding to final paper; migrated from prior EPF library", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F10_2024_NeurIPS_TimeXer.pdf",
        "url": "https://arxiv.org/pdf/2402.19072v4",
        "migrated_from": "zl-comment/epf-frontier-study@8c6b0ee720c60ed85bed6f138201228951ca4c76",
    },
    {
        "id": "F11", "title": "CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables",
        "date": "2025", "doi": "10.1145/3711896.3736899", "venue": "KDD 2025",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v1 author manuscript corresponding to final paper; migrated from prior EPF library", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F11_2025_KDD_CrossLinear.pdf",
        "url": "https://arxiv.org/pdf/2505.23116v1",
        "migrated_from": "zl-comment/epf-frontier-study@8c6b0ee720c60ed85bed6f138201228951ca4c76",
    },
    {
        "id": "F12", "title": "ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting",
        "date": "2026", "doi": None, "venue": "ICLR 2026",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v4 author manuscript corresponding to ICLR paper; migrated from prior EPF library", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F12_2026_ICLR_ProtoTS.pdf",
        "url": "https://arxiv.org/pdf/2509.23159v4",
        "migrated_from": "zl-comment/epf-frontier-study@8c6b0ee720c60ed85bed6f138201228951ca4c76",
        "official_record": "https://openreview.net/forum?id=IbcdVwzLrp",
    },
    {
        "id": "F13", "title": "UniCA: Unified Covariate Adaptation for Time Series Foundation Model",
        "date": "2026", "doi": None, "venue": "ICLR 2026",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v2 author manuscript corresponding to ICLR paper; migrated from prior EPF library", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F13_2026_ICLR_UniCA.pdf",
        "url": "https://arxiv.org/pdf/2506.22039v2",
        "migrated_from": "zl-comment/epf-frontier-study@8c6b0ee720c60ed85bed6f138201228951ca4c76",
        "official_record": "https://openreview.net/forum?id=I8q4MZb4OP",
    },
]

SOURCE_CARDS = [
    {
        "id": "F04",
        "title": "Recent Advances in Electricity Price Forecasting: A Review of Probabilistic Forecasting",
        "date": "2018", "doi": "10.1016/j.rser.2017.05.234", "venue": "Renewable and Sustainable Energy Reviews",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "availability": "source_card", "file": "02_partial_forecasting_uncertainty/01_top_journals/F04_SOURCE_2018_RSER_Probabilistic_EPF_Review.md",
        "source_url": "https://doi.org/10.1016/j.rser.2017.05.234",
        "reason_no_pdf": "No stable public PDF was verified without bypassing TLS or access controls.",
    },
    {
        "id": "F06",
        "title": "Forecasting Finnish aFRR Energy Reserve Market Prices Using Deep Learning and Tree-Based Models",
        "date": "2026", "doi": "10.1016/j.egyai.2026.100724", "venue": "Energy and AI",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "forecasting_uncertainty",
        "availability": "source_card", "file": "02_partial_forecasting_uncertainty/02_peer_reviewed_specialized/F06_SOURCE_2026_Energy_AI_Finnish_aFRR_Forecasting.md",
        "source_url": "https://doi.org/10.1016/j.egyai.2026.100724",
        "reason_no_pdf": "The publisher full text was not available to the automated collector; the linked dataset is public.",
    },
    {
        "id": "F15",
        "title": "Joint Forecasting of Source-Load-Price for Integrated Energy System Based on Multi-Task Learning and Hybrid Attention Mechanism",
        "date": "2024", "doi": "10.1016/j.apenergy.2024.122821", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "availability": "source_card", "file": "02_partial_forecasting_uncertainty/01_top_journals/F15_SOURCE_2024_Applied_Energy_Source_Load_Price_MTL.md",
        "source_url": "https://doi.org/10.1016/j.apenergy.2024.122821",
        "reason_no_pdf": "Publisher full text is behind an access wall and cannot be re-fetched by the unauthenticated collector, so the library keeps a source card instead of an unreproducible PDF.",
    },
    {
        "id": "M02",
        "title": "Multi-Market Revenue Optimization for Integrated Wind and Hybrid Energy Storage Systems",
        "date": "2026", "doi": "10.1016/j.apenergy.2025.126773", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "energy_reserve_decision",
        "availability": "source_card", "file": "03_partial_energy_reserve_decision/01_top_journals/M02_SOURCE_2026_Applied_Energy_Multi_Market_Wind_HESS.md",
        "source_url": "https://doi.org/10.1016/j.apenergy.2025.126773",
        "reason_no_pdf": "The publisher full text was not available to the automated collector.",
    },
    {
        "id": "M05",
        "title": "Integrated Scheduling and Bidding of Power and Reserve of Energy Resource Aggregators with Storage Plants",
        "date": "2022", "doi": "10.1016/j.apenergy.2022.119285", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "energy_reserve_decision",
        "availability": "source_card", "file": "03_partial_energy_reserve_decision/01_top_journals/M05_SOURCE_2022_Applied_Energy_Integrated_Scheduling_Bidding.md",
        "source_url": "https://doi.org/10.1016/j.apenergy.2022.119285",
        "reason_no_pdf": "The publisher full text was not available to the automated collector.",
    },
    {
        "id": "F14",
        "title": "Day-Ahead Electricity Price Forecasting Method Integrating Multi-Scale Hypergraph Features and Dual-Layer Transformer",
        "date": "2026", "doi": "10.1016/j.apenergy.2026.127396", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "availability": "source_card", "file": "02_partial_forecasting_uncertainty/01_top_journals/F14_SOURCE_2026_Applied_Energy_DTHG_Guangdong.md",
        "source_url": "https://doi.org/10.1016/j.apenergy.2026.127396",
        "reason_no_pdf": "The prior repository marks the PDF as a user-supplied attachment whose redistribution license must be checked; only a verified source card is migrated.",
        "migrated_from": "zl-comment/epf-frontier-study@8c6b0ee720c60ed85bed6f138201228951ca4c76",
    },
]

TAXONOMY = {
    "intersection_frontier": "处于多市场场景，并交叉覆盖预测/不确定性、联合决策、可交付性或闭环评价中的至少两个方面。",
    "forecasting_uncertainty": "主要解决价格、激活或联合分布预测，不完整处理能量—备用协同决策。",
    "energy_reserve_decision": "主要解决联合竞价、调度或出清，不以联合预测为核心。",
    "deliverability_control": "主要解决功率、能量、SOC、变流器或实时跟踪可交付性。",
    "decision_focused_learning": "主要连接预测与下游优化目标，未必针对能量—备用市场。",
}

CROSS_TAGS = {
    "X01": ["multi_market", "joint_decision", "deliverability_control", "field_characterization"],
    "X02": ["multi_market", "forecasting_uncertainty", "joint_decision", "economic_evaluation"],
    "X03": ["multi_market", "forecasting_uncertainty", "decision_value_evaluation"],
    "X04": ["multi_market", "joint_decision", "uncertainty_aware_bidding"],
    "X05": ["multi_market", "joint_decision", "deliverability_control", "robust_optimization"],
    "F01": ["forecasting_uncertainty", "benchmark_protocol"],
    "F02": ["forecasting_uncertainty", "cross_market_features"],
    "F03": ["forecasting_uncertainty", "probabilistic_forecasting"],
    "F04": ["forecasting_uncertainty", "probabilistic_evaluation"],
    "F05": ["forecasting_uncertainty", "bayesian_deep_learning"],
    "F06": ["forecasting_uncertainty", "reserve_price", "up_down_asymmetry"],
    "F07": ["forecasting_uncertainty", "single_energy_market", "decision_value_evaluation"],
    "F08": ["forecasting_uncertainty", "tree_baseline"],
    "F09": ["forecasting_uncertainty", "electricity_price", "exogenous_variables", "interpretable_decomposition"],
    "F10": ["forecasting_uncertainty", "exogenous_variables", "transformer"],
    "F11": ["forecasting_uncertainty", "exogenous_variables", "cross_correlation"],
    "F12": ["forecasting_uncertainty", "interpretable_forecasting", "prototypes"],
    "F13": ["forecasting_uncertainty", "foundation_model", "covariate_adaptation"],
    "F14": ["forecasting_uncertainty", "electricity_price", "spatial_hypergraph", "china_transfer"],
    "F15": ["forecasting_uncertainty", "multi_task_learning", "joint_source_load_price", "integrated_energy_system"],
    "M01": ["joint_decision", "market_clearing", "soc_dependent_bids"],
    "M02": ["joint_decision", "activation_aware", "degradation", "hybrid_storage"],
    "M03": ["joint_decision", "market_design", "up_down_asymmetry"],
    "M04": ["joint_decision", "frequency_activation", "degradation"],
    "M05": ["joint_decision", "bid_acceptance", "reserve_availability"],
    "M06": ["joint_decision", "scenario_economics", "day_ahead_afrr"],
    "M07": ["joint_decision", "sequential_market_baseline", "gate_closure_timing", "rolling_forecast"],
    "D01": ["deliverability_control", "simultaneous_services", "power_energy_budget"],
    "D02": ["deliverability_control", "converter_capability", "real_time_control"],
    "D03": ["deliverability_control", "planning_control_loop", "multi_service"],
    "L01": ["decision_focused_learning", "end_to_end_stochastic_optimization"],
    "L02": ["decision_focused_learning", "spo_plus", "optimization_loss"],
    "L03": ["decision_focused_learning", "single_energy_market", "storage_arbitrage"],
    "L04": ["decision_focused_learning", "strategic_storage", "market_feedback"],
    "L05": ["decision_focused_learning", "interpretable_trees"],
    "L06": ["decision_focused_learning", "single_energy_market", "predict_then_bid"],
}

QUALITY_TIERS = {
    "top_journal": "经正式发表且属于本课题锚定的领域顶级或高影响力期刊。",
    "top_conference": "经正式录用并进入可核验的国际顶会正式论文集。",
    "peer_reviewed_specialized": "正式同行评审的专业期刊或会议论文，但不作为本库的 Top 锚点。",
    "preprint": "仅可核实为预印本、工作论文或在审稿件，不计作正式同行评审成果。",
}

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
EPF_REPO = "zl-comment/epf-frontier-study"
EPF_COMMIT = "8c6b0ee720c60ed85bed6f138201228951ca4c76"
EPF_FILES = [
    {
        "source_url": "https://zenodo.org/records/4624805/files/DE.csv?download=1",
        "upstream_path": "Zenodo 4624805/DE.csv",
        "filename": "DE.csv",
        "sha256": "e421cf4eb160cecfec6c7c8222e7168b3c751d0dd6771ba21401784f3f2a4565",
        "kind": "benchmark_input",
    },
    {
        "source_url": "https://raw.githubusercontent.com/jeslago/epftoolbox/47d6e0629f65ebd19d3c12cb5689dbad0c2ea078/forecasts/Forecasts_DE_DNN_LEAR_ensembles.csv",
        "upstream_path": "jeslago/epftoolbox@47d6e062/forecasts/Forecasts_DE_DNN_LEAR_ensembles.csv",
        "filename": "Forecasts_DE_DNN_LEAR_ensembles.csv",
        "sha256": "1577f8ae0851fb536e654b69608c94840a9813aa5776d04691dd5ec1a2a3ea13",
        "kind": "paper_author_forecasts",
    },
]
MIGRATED_PAPER_IDS = {"F08", "F09", "F10", "F11", "F12", "F13"}


def retrieve(url, timeout=120):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.url, response.headers.get("Content-Type", "")


def sha256(body):
    return hashlib.sha256(body).hexdigest()


def write_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def write_paper_catalog(pdf_items):
    catalog_path = PAPER_DIR / "catalog.json"
    previous = {}
    previous_alignment_meta = None
    if catalog_path.exists():
        old_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        previous = {item["id"]: item for item in old_catalog.get("items", [])}
        previous_alignment_meta = old_catalog.get("data_alignment")
    items = []
    for item in pdf_items:
        entry = dict(item)
        entry["availability"] = "full_text_pdf"
        entry["local_path"] = entry.pop("path")
        entry["cross_tags"] = CROSS_TAGS[entry["id"]]
        for key in ("original_data_status", "data_links"):
            if key in previous.get(entry["id"], {}):
                entry[key] = previous[entry["id"]][key]
        items.append(entry)
    for item in SOURCE_CARDS:
        entry = dict(item)
        entry["local_path"] = f"paper/multi_market_energy_reserve/{entry.pop('file')}"
        entry["cross_tags"] = CROSS_TAGS[entry["id"]]
        for key in ("original_data_status", "data_links"):
            if key in previous.get(entry["id"], {}):
                entry[key] = previous[entry["id"]][key]
        items.append(entry)
    items.sort(key=lambda item: item["id"])
    catalog = {
        "research_cutoff": CUTOFF,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "organization_rule": "One canonical location per paper: first by problem coverage, then by publication-quality tier. Cross-problem relevance is recorded as metadata, not duplicate PDFs.",
        "taxonomy": TAXONOMY,
        "quality_tiers": QUALITY_TIERS,
        "counts": {
            "items": len(items),
            "full_text_pdfs": sum(item["availability"] == "full_text_pdf" for item in items),
            "source_cards": sum(item["availability"] == "source_card" for item in items),
            "by_problem": dict(sorted(Counter(item["primary_problem"] for item in items).items())),
            "by_quality_tier": dict(sorted(Counter(item["quality_tier"] for item in items).items())),
        },
        "items": items,
    }
    if previous_alignment_meta:
        catalog["data_alignment"] = previous_alignment_meta
    catalog_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def catalog_from_manifest():
    manifest_path = PAPER_DIR / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_paper_catalog(manifest["files"])


def download_papers(selected=None, merge=False):
    results = []
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    selected = PAPERS if selected is None else selected
    for item in selected:
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
    if merge:
        manifest_path = PAPER_DIR / "download_manifest.json"
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        selected_ids = {item["id"] for item in selected}
        results = [item for item in existing if item["id"] not in selected_ids] + results
        order = {item["id"]: index for index, item in enumerate(PAPERS)}
        results.sort(key=lambda item: order[item["id"]])
    manifest = {
        "research_cutoff": CUTOFF,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Problem-first, quality-tiered literature for energy-reserve multi-market forecasting and storage decision research.",
        "files": results,
    }
    (PAPER_DIR / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_paper_catalog(results)


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


def collect_epf_de():
    entries = []
    directory = DATA_DIR / "epf_de_benchmark"
    for source in EPF_FILES:
        source_url = source["source_url"]
        body, resolved, content_type = retrieve(source_url)
        digest = sha256(body)
        if digest != source["sha256"]:
            raise ValueError(f"EPF benchmark SHA-256 mismatch: {source['upstream_path']}")
        target = directory / source["filename"]
        write_bytes(target, body)
        entries.append(
            {
                "dataset": "EPF-DE benchmark",
                "kind": source["kind"],
                "migrated_from_repository": EPF_REPO,
                "migrated_from_commit": EPF_COMMIT,
                "upstream_path": source["upstream_path"],
                "path": str(target.relative_to(ROOT)),
                "source_url": source_url,
                "resolved_url": resolved,
                "content_type": content_type,
                "bytes": len(body),
                "sha256": digest,
                "status": "downloaded",
            }
        )
    print(f"OK EPF-DE benchmark: {len(entries)} pinned files", flush=True)
    return entries


def write_data_manifest(entries):
    manifest = {
        "research_cutoff": CUTOFF,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Observed DK1 energy/aFRR/balancing data, a Finland reserve-price forecasting dataset, a pinned system benchmark, and a pinned EPF-DE forecasting benchmark.",
        "files": entries,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def collect_epf_de_incremental():
    entries = collect_epf_de()
    manifest_path = DATA_DIR / "download_manifest.json"
    current = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    current = [item for item in current if item.get("dataset") != "EPF-DE benchmark"]
    write_data_manifest(current + entries)


def collect_data():
    entries = []
    entries.extend(collect_energinet())
    entries.extend(collect_finland_afrr())
    entries.extend(collect_rts())
    entries.extend(collect_epf_de())
    write_data_manifest(entries)
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
    parser.add_argument("group", choices=("papers", "migrated-papers", "catalog", "epf-data", "data", "all"))
    args = parser.parse_args()
    if args.group in ("papers", "all"):
        download_papers()
    elif args.group == "migrated-papers":
        download_papers([item for item in PAPERS if item["id"] in MIGRATED_PAPER_IDS], merge=True)
    elif args.group == "catalog":
        catalog_from_manifest()
    if args.group == "epf-data":
        collect_epf_de_incremental()
    if args.group in ("data", "all"):
        collect_data()


if __name__ == "__main__":
    main()
