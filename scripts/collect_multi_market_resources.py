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
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-09-16"
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
        "id": "X06", "title": "Data-Driven Scheduling of Energy Storage in Day-Ahead Energy and Reserve Markets With Probabilistic Guarantees on Real-Time Delivery",
        "date": "2021", "doi": "10.1109/TPWRS.2020.3046710", "venue": "IEEE Transactions on Power Systems",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "intersection_frontier",
        "version": "University of Mons institutional manuscript corresponding to the final article", "file": "01_intersection_frontier/01_top_journal_or_conference/X06_2021_TPWRS_Data_Driven_DA_Reserve.pdf",
        "url": "https://orbi.umons.ac.be/bitstream/20.500.12907/27738/1/09305967.pdf",
    },
    {
        "id": "X09", "title": "Two-Stage Stochastic Optimization Frameworks to Aid in Decision-Making Under Uncertainty for Variable Resource Generators Participating in a Sequential Energy Market",
        "date": "2021", "doi": "10.1016/j.apenergy.2021.116882", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "intersection_frontier",
        "version": "arXiv author manuscript corresponding to the final article", "file": "01_intersection_frontier/01_top_journal_or_conference/X09_2021_Applied_Energy_Sequential_Stochastic.pdf",
        "url": "https://arxiv.org/pdf/2012.13459",
    },
    {
        "id": "X10", "title": "Optimisation Models for the Day-Ahead Energy and Reserve Self-Scheduling of a Hybrid Wind-Battery Virtual Power Plant",
        "date": "2023", "doi": "10.1016/j.est.2022.106296", "venue": "Journal of Energy Storage",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "version": "arXiv author manuscript corresponding to the final article", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X10_2023_JES_Wind_Battery_Self_Scheduling.pdf",
        "url": "https://arxiv.org/pdf/2206.13784",
    },
    {
        "id": "X11", "title": "Day-Ahead and Intra-Day Planning of Integrated BESS-PV Systems Providing Frequency Regulation",
        "date": "2020", "doi": "10.1109/TSTE.2019.2941369", "venue": "IEEE Transactions on Sustainable Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "version": "arXiv author manuscript corresponding to the final article", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X11_2020_TSTE_BESS_PV_Frequency_Regulation.pdf",
        "url": "https://arxiv.org/pdf/2104.07352",
    },
    {
        "id": "X12", "title": "An Open Source Stochastic Unit Commitment Tool Using the PyPSA-Framework",
        "date": "2024", "doi": "10.1016/j.ifacol.2024.07.485", "venue": "IFAC-PapersOnLine",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "version": "arXiv author manuscript corresponding to the final paper", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X12_2024_IFAC_PyPSA_Stochastic_UC.pdf",
        "url": "https://arxiv.org/pdf/2405.06490",
        "official_code": "https://github.com/PPGS-Tools/PyPSA-stochUC",
    },
    {
        "id": "X13", "title": "Revenue Stacking of BESSs in Wholesale and aFRR Markets with Delivery Guarantees",
        "date": "2024", "doi": "10.1016/j.epsr.2024.110633", "venue": "Electric Power Systems Research",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "version": "Zenodo open-access submitted manuscript corresponding to the final article", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X13_2024_EPSR_BESS_aFRR_Delivery_Guarantees.pdf",
        "url": "https://zenodo.org/api/records/15373462/files/manuscript_r1.pdf/content",
    },
    {
        "id": "X15", "title": "Optimal BESS Scheduling for Multi-Market Participation in the Nordics",
        "date": "2025", "doi": "10.1109/ISGTEurope64741.2025.11305471", "venue": "IEEE PES ISGT Europe 2025",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "version": "arXiv v1 author manuscript corresponding to the final paper", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X15_2025_ISGT_Nordic_Multi_Market_BESS.pdf",
        "url": "https://arxiv.org/pdf/2506.02837v1",
    },
    {
        "id": "X17", "title": "Coordinated Trading Strategies for Battery Storage in Reserve and Spot Markets",
        "date": "2024", "doi": "10.48550/arXiv.2406.08390", "venue": "arXiv",
        "publication_status": "preprint; no formal venue verified by the research cutoff", "quality_tier": "preprint", "primary_problem": "intersection_frontier",
        "version": "arXiv manuscript", "file": "01_intersection_frontier/03_preprints/X17_2024_Coordinated_Reserve_Spot_Trading.pdf",
        "url": "https://arxiv.org/pdf/2406.08390",
    },
    {
        "id": "X19", "title": "On the Participation of Energy Storage Systems in Reserve Markets Using Decision Focused Learning",
        "date": "2025", "doi": "10.1016/j.segan.2025.101677", "venue": "Sustainable Energy, Grids and Networks",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "version": "University of Mons institutional open-access final manuscript", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X19_2025_SEGAN_Reserve_Decision_Focused.pdf",
        "url": "https://orbi.umons.ac.be/bitstream/20.500.12907/52795/1/1-s2.0-S2352467725000591-main.pdf",
    },
    {
        "id": "X20", "title": "When Forecast Accuracy Fails: Rank Correlation and Decision Quality in Multi-Market Battery Storage Optimization",
        "date": "2026", "doi": "10.48550/arXiv.2604.12082", "venue": "arXiv",
        "publication_status": "preprint; no formal venue verified by the research cutoff", "quality_tier": "preprint", "primary_problem": "intersection_frontier",
        "version": "arXiv manuscript", "file": "01_intersection_frontier/03_preprints/X20_2026_Forecast_Accuracy_Multi_Market_BESS.pdf",
        "url": "https://arxiv.org/pdf/2604.12082",
    },
    {
        "id": "X21", "title": "Control and Scheduling of Behind-the-Meter Battery Energy Storage Systems for Stacked Grid and Building Services",
        "date": "2026", "doi": "10.1016/j.segy.2026.100249", "venue": "Smart Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "version": "arXiv author manuscript corresponding to the final article", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X21_2026_Smart_Energy_BTM_BESS_Stacked_Services.pdf",
        "url": "https://arxiv.org/pdf/2605.07762",
    },
    {
        "id": "X22", "title": "Co-optimized Trading of Hybrid Wind Power Plant with Retired EV Batteries in Energy and Reserve Markets under Uncertainties",
        "date": "2020", "doi": "10.1016/j.ijepes.2019.105631", "venue": "International Journal of Electrical Power & Energy Systems",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "version": "OSTI open-access accepted manuscript corresponding to the final article", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X22_2020_IJEPES_Wind_Retired_Battery.pdf",
        "url": "https://www.osti.gov/servlets/purl/1579636",
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
        "id": "F06", "title": "Forecasting Finnish aFRR Energy Reserve Market Prices Using Deep Learning and Tree-Based Models",
        "date": "2026", "doi": "10.1016/j.egyai.2026.100724", "venue": "Energy and AI",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "forecasting_uncertainty",
        "version": "Åbo Akademi institutional repository final published version, CC BY",
        "file": "02_partial_forecasting_uncertainty/02_peer_reviewed_specialized/F06_2026_Energy_AI_Finnish_aFRR_Forecasting.pdf",
        "url": "https://research.abo.fi/ws/files/74781835/1-s2.0-S2666546826000509-main.pdf",
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
        "url": "https://infoscience.epfl.ch/server/api/core/bitstreams/aed3ecf7-97e4-46f3-8a4d-6348b6fb5aa1/content",
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
    {
        "id": "F17", "title": "Intraday Electricity Price Forecasting via LSTM and Trading Strategy for the Power Market: A Case Study of the West Denmark DK1 Grid Region",
        "date": "2024", "doi": "10.3390/en17122909", "venue": "Energies",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "forecasting_uncertainty",
        "version": "MDPI final published version (open access)", "file": "02_partial_forecasting_uncertainty/02_peer_reviewed_specialized/F17_2024_Energies_DK1_Intraday_LSTM_Trading.pdf",
        "url": "https://mdpi-res.com/d_attachment/energies/energies-17-02909/article_deploy/energies-17-02909.pdf",
    },
    {
        "id": "F19", "title": "Marginal Tail-Adaptive Normalizing Flows",
        "date": "2022", "doi": None, "venue": "ICML 2022",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "PMLR final published version", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F19_2022_ICML_Marginal_Tail_Adaptive_Flows.pdf",
        "url": "https://proceedings.mlr.press/v162/laszkiewicz22a/laszkiewicz22a.pdf",
        "official_record": "https://proceedings.mlr.press/v162/laszkiewicz22a.html", "official_code": "https://github.com/MikeLasz/marginalTailAdaptiveFlow",
    },
    {
        "id": "F20", "title": "High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes",
        "date": "2019", "doi": None, "venue": "NeurIPS 2019",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "NeurIPS final published version", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F20_2019_NeurIPS_Low_Rank_Gaussian_Copula.pdf",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2019/file/0b105cf1504c4e241fcc6d519ea962fb-Paper.pdf",
        "official_record": "https://proceedings.neurips.cc/paper/2019/hash/0b105cf1504c4e241fcc6d519ea962fb-Abstract.html", "official_code": "https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release",
    },
    {
        "id": "F21", "title": "TACTiS: Transformer-Attentional Copulas for Time Series",
        "date": "2022", "doi": None, "venue": "ICML 2022",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "PMLR final published version", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F21_2022_ICML_TACTiS.pdf",
        "url": "https://proceedings.mlr.press/v162/drouin22a/drouin22a.pdf",
        "official_record": "https://proceedings.mlr.press/v162/drouin22a.html", "official_code": "https://github.com/ServiceNow/TACTiS",
    },
    {
        "id": "F22", "title": "TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series",
        "date": "2024", "doi": None, "venue": "ICLR 2024",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "ICLR final published version", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F22_2024_ICLR_TACTiS_2.pdf",
        "url": "https://proceedings.iclr.cc/paper_files/paper/2024/file/63796148c99205adb0fcac069cc714d4-Paper-Conference.pdf",
        "official_record": "https://proceedings.iclr.cc/paper_files/paper/2024/hash/63796148c99205adb0fcac069cc714d4-Abstract-Conference.html", "official_code": "https://github.com/ServiceNow/TACTiS",
    },
    {
        "id": "F23", "title": "Multivariate Probabilistic Time Series Forecasting with Correlated Errors",
        "date": "2024", "doi": "10.52202/079017-1720", "venue": "NeurIPS 2024",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "NeurIPS final published version", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F23_2024_NeurIPS_Correlated_Errors.pdf",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2024/file/619b8e3ead58dce90bc615f2a7d5d102-Paper-Conference.pdf",
        "official_record": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/619b8e3ead58dce90bc615f2a7d5d102-Abstract-Conference.html", "official_code": "https://github.com/rottenivy/mv_pts_correlatederr",
    },
    {
        "id": "F24", "title": "Multivariate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows",
        "date": "2021", "doi": None, "venue": "ICLR 2021",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v3 author manuscript corresponding to ICLR paper", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F24_2021_ICLR_TempFlow.pdf",
        "url": "https://arxiv.org/pdf/2002.06103v3",
        "official_record": "https://openreview.net/forum?id=WiGQBFuVRv", "official_code": "https://github.com/zalandoresearch/pytorch-ts/tree/master/pts/model/tempflow",
    },
    {
        "id": "F26", "title": "Unified Training of Universal Time Series Forecasting Transformers",
        "date": "2024", "doi": None, "venue": "ICML 2024",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v2 author manuscript corresponding to ICML paper", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F26_2024_ICML_Moirai.pdf",
        "url": "https://arxiv.org/pdf/2402.02592v2",
        "official_record": "https://proceedings.mlr.press/v235/woo24a.html", "official_code": "https://github.com/SalesforceAIResearch/uni2ts",
    },
    {
        "id": "F27", "title": "Chronos: Learning the Language of Time Series",
        "date": "2024", "doi": None, "venue": "Transactions on Machine Learning Research",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v3 author manuscript corresponding to TMLR paper", "file": "02_partial_forecasting_uncertainty/02_peer_reviewed_specialized/F27_2024_TMLR_Chronos.pdf",
        "url": "https://arxiv.org/pdf/2403.07815v3",
        "official_record": "https://openreview.net/forum?id=gerNCVqqtR", "official_code": "https://github.com/amazon-science/chronos-forecasting",
    },
    {
        "id": "F28", "title": "Conformal Risk Control",
        "date": "2024", "doi": None, "venue": "ICLR 2024",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "ICLR final published version", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F28_2024_ICLR_Conformal_Risk_Control.pdf",
        "url": "https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf",
        "official_record": "https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html", "official_code": "https://github.com/aangelopoulos/conformal-risk",
    },
    {
        "id": "F29", "title": "Non-Exchangeable Conformal Risk Control",
        "date": "2024", "doi": None, "venue": "ICLR 2024",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "ICLR final published version", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F29_2024_ICLR_Non_Exchangeable_CRC.pdf",
        "url": "https://proceedings.iclr.cc/paper_files/paper/2024/file/de04896f011beff76c91e094f72727f4-Paper-Conference.pdf",
        "official_record": "https://proceedings.iclr.cc/paper_files/paper/2024/hash/de04896f011beff76c91e094f72727f4-Abstract-Conference.html", "official_code": "https://github.com/deep-spin/non-exchangeable-crc",
    },
    {
        "id": "F30", "title": "Copula Conformal Prediction for Multi-step Time Series Forecasting",
        "date": "2024", "doi": None, "venue": "ICLR 2024",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "ICLR final published version", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F30_2024_ICLR_CopulaCPTS.pdf",
        "url": "https://proceedings.iclr.cc/paper_files/paper/2024/file/8707924df5e207fa496f729f49069446-Paper-Conference.pdf",
        "official_record": "https://proceedings.iclr.cc/paper_files/paper/2024/hash/8707924df5e207fa496f729f49069446-Abstract-Conference.html", "official_code": "https://github.com/Rose-STL-Lab/CopulaCPTS",
    },
    {
        "id": "F31", "title": "Conformal PID Control for Time Series Prediction",
        "date": "2023", "doi": "10.52202/075280-1000", "venue": "NeurIPS 2023",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "NeurIPS final published version", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F31_2023_NeurIPS_Conformal_PID.pdf",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2023/file/47f2fad8c1111d07f83c91be7870f8db-Paper-Conference.pdf",
        "official_record": "https://proceedings.neurips.cc/paper_files/paper/2023/hash/47f2fad8c1111d07f83c91be7870f8db-Abstract-Conference.html", "official_code": "https://github.com/aangelopoulos/conformal-time-series",
    },
    {
        "id": "F32", "title": "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting",
        "date": "2021", "doi": None, "venue": "ICML 2021",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "PMLR final published version", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F32_2021_ICML_TimeGrad.pdf",
        "url": "https://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf",
        "official_record": "https://proceedings.mlr.press/v139/rasul21a.html", "official_code": "https://github.com/zalandoresearch/pytorch-ts/tree/master/pts/model/time_grad",
    },
    {
        "id": "F33", "title": "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation",
        "date": "2021", "doi": None, "venue": "NeurIPS 2021",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "forecasting_uncertainty",
        "version": "NeurIPS final published version", "file": "02_partial_forecasting_uncertainty/01_top_conferences/F33_2021_NeurIPS_CSDI.pdf",
        "url": "https://proceedings.neurips.cc/paper/2021/file/cfe8504bda37b575c70ee1a8276f3486-Paper.pdf",
        "official_record": "https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html", "official_code": "https://github.com/ermongroup/CSDI",
    },
    {
        "id": "F34", "title": "Probabilistic Day-Ahead Forecasting of System-Level Renewable Energy and Electricity Demand",
        "date": "2026-02-28", "doi": "10.1038/s41467-026-69015-w", "venue": "Nature Communications",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "version": "publisher PDF, CC BY 4.0", "file": "02_partial_forecasting_uncertainty/01_top_journals/F34_2026_Nature_Communications_Joint_System_Forecasting.pdf",
        "url": "https://www.nature.com/articles/s41467-026-69015-w.pdf",
        "official_data": "https://doi.org/10.5281/zenodo.16729434", "official_code": "https://doi.org/10.5281/zenodo.18156677",
    },
    {
        "id": "F35", "title": "PriceFM: Foundation Model for Probabilistic Electricity Price Forecasting",
        "date": "2026-05-08 revision", "doi": "10.48550/arXiv.2508.04875", "venue": "arXiv",
        "publication_status": "preprint; no formal venue verified by the research cutoff", "quality_tier": "preprint", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v4", "file": "02_partial_forecasting_uncertainty/03_preprints/F35_2026_PriceFM.pdf",
        "url": "https://arxiv.org/pdf/2508.04875v4", "official_code": "https://github.com/runyao-yu/PriceFM",
    },
    {
        "id": "F36", "title": "Foundation Models for Electricity Price Forecasting and Battery Arbitrage: Can They Replace Market-Specific Forecasting Models?",
        "date": "2026-08-31", "doi": "10.48550/arXiv.2609.00089", "venue": "arXiv",
        "publication_status": "preprint; no formal venue verified by the research cutoff", "quality_tier": "preprint", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v1", "file": "02_partial_forecasting_uncertainty/03_preprints/F36_2026_Foundation_Models_Battery_Value.pdf",
        "url": "https://arxiv.org/pdf/2609.00089v1",
    },
    {
        "id": "F37", "title": "On-Line Conformalized Neural Networks Ensembles for Probabilistic Forecasting of Day-Ahead Electricity Prices",
        "date": "2025-11-15", "doi": "10.1016/j.apenergy.2025.126412", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v2 author manuscript corresponding to the final article", "file": "02_partial_forecasting_uncertainty/01_top_journals/F37_2025_Applied_Energy_Online_Conformal_EPF.pdf",
        "url": "https://arxiv.org/pdf/2404.02722v2",
        "official_code": "https://github.com/bruale/PefCodeBench",
    },
    {
        "id": "F38", "title": "Adaptive Conformal Inference Under Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic",
        "date": "2026-09-07", "doi": "10.48550/arXiv.2609.07251", "venue": "arXiv",
        "publication_status": "preprint; no formal venue verified by the research cutoff", "quality_tier": "preprint", "primary_problem": "forecasting_uncertainty",
        "version": "arXiv v1", "file": "02_partial_forecasting_uncertainty/03_preprints/F38_2026_Delayed_Feedback_ACI.pdf",
        "url": "https://arxiv.org/pdf/2609.07251v1",
    },
    {
        "id": "M08", "title": "Optimal Participation of a Wind and Hybrid Battery Storage System in the Day-Ahead and Automatic Frequency Restoration Reserve Markets",
        "date": "2024", "doi": "10.1016/j.est.2024.112309", "venue": "Journal of Energy Storage",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "energy_reserve_decision",
        "version": "Aarhus University institutional open-access publisher PDF, CC BY 4.0", "file": "03_partial_energy_reserve_decision/02_peer_reviewed_specialized/M08_2024_JES_Wind_HESS_DA_aFRR.pdf",
        "url": "https://pure.au.dk/ws/files/445224752/1-s2.0-S2352152X24018954-main.pdf",
    },
    {
        "id": "M09", "title": "The Role of Electricity Market Design for Energy Storage in Cost-Efficient Decarbonization",
        "date": "2023-06-21", "doi": "10.1016/j.joule.2023.05.014", "venue": "Joule",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "energy_reserve_decision",
        "version": "NSF public-access copy corresponding to the final article", "file": "03_partial_energy_reserve_decision/01_top_journals/M09_2023_Joule_Storage_Market_Design.pdf",
        "url": "https://par.nsf.gov/servlets/purl/10477880", "official_code": "https://github.com/Huskyseen/Storage_Market",
    },
    {
        "id": "D04", "title": "Multi-Year Field Measurements of Home Storage Systems and Their Use in Capacity Estimation",
        "date": "2024-09-16", "doi": "10.1038/s41560-024-01620-9", "venue": "Nature Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "deliverability_control",
        "version": "publisher PDF, open access", "file": "04_partial_deliverability_control/01_top_journals/D04_2024_Nature_Energy_Home_Storage_Field_Measurements.pdf",
        "url": "https://www.nature.com/articles/s41560-024-01620-9.pdf", "official_data": "https://doi.org/10.5281/zenodo.12091223",
    },
    {
        "id": "L07", "title": "Online Energy Storage Arbitrage under Imperfect Predictions: A Conformal Risk-Aware Approach",
        "date": "2026-06-22", "doi": "10.1145/3744255.3798116", "venue": "ACM e-Energy 2026",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "peer_reviewed_specialized", "primary_problem": "decision_focused_learning",
        "version": "arXiv v2 author manuscript corresponding to the final paper", "file": "05_partial_decision_focused_learning/02_peer_reviewed_specialized/L07_2026_eEnergy_Online_Storage_Conformal_Risk.pdf",
        "url": "https://arxiv.org/pdf/2511.01032v2", "official_record": "https://doi.org/10.1145/3744255.3798116",
    },
    {
        "id": "L08", "title": "Conformal Risk Training: End-to-End Optimization of Conformal Risk Control",
        "date": "2025", "doi": "10.52202/085713-2355", "venue": "NeurIPS 2025",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "top_conference", "primary_problem": "decision_focused_learning",
        "version": "NeurIPS final published version", "file": "05_partial_decision_focused_learning/01_top_journal_or_conference/L08_2025_NeurIPS_Conformal_Risk_Training.pdf",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2025/file/6559542f75b4452ebaaf82094c7defb7-Paper-Conference.pdf",
        "official_record": "https://proceedings.neurips.cc/paper_files/paper/2025/hash/6559542f75b4452ebaaf82094c7defb7-Abstract-Conference.html",
        "official_code": "https://github.com/chrisyeh96/conformal-risk-training",
    },
    {
        "id": "L09", "title": "Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions",
        "date": "2024-05-02 revision", "doi": "10.48550/arXiv.2310.05921", "venue": "arXiv",
        "publication_status": "preprint; no formal venue verified by the research cutoff", "quality_tier": "preprint", "primary_problem": "decision_focused_learning",
        "version": "arXiv v3", "file": "05_partial_decision_focused_learning/03_preprints/L09_2024_Conformal_Decision_Theory.pdf",
        "url": "https://arxiv.org/pdf/2310.05921v3",
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
    {
        "id": "F16",
        "title": "Forecasting day-ahead electricity prices with spatial dependence",
        "date": "2024", "doi": "10.1016/j.ijforecast.2023.11.006", "venue": "International Journal of Forecasting",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "availability": "source_card", "file": "02_partial_forecasting_uncertainty/01_top_journals/F16_SOURCE_2024_IJF_Nord_Pool_Spatial_Dependence.md",
        "source_url": "https://doi.org/10.1016/j.ijforecast.2023.11.006",
        "reason_no_pdf": "The publisher licence is text-and-data-mining only and no open-access or arXiv version was found, so the library keeps a source card.",
    },
    {
        "id": "X07",
        "title": "Forecast-Driven Stochastic Scheduling of a Virtual Power Plant in Energy and Reserve Markets",
        "date": "2022", "doi": "10.1109/JSYST.2021.3114445", "venue": "IEEE Systems Journal",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "availability": "source_card", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X07_SOURCE_2022_IEEE_Systems_VPP_Stochastic.md",
        "source_url": "https://doi.org/10.1109/JSYST.2021.3114445",
        "reason_no_pdf": "The full text was read from a copy the user obtained through an institutional IEEE subscription; it is not open access and cannot be re-fetched by the unauthenticated collector, so the library keeps a source card instead of an unreproducible PDF.",
    },
    {
        "id": "X08",
        "title": "Evaluation of a Data Driven Stochastic Approach to Optimize the Participation of a Wind and Storage Power Plant in Day-Ahead and Reserve Markets",
        "date": "2018", "doi": "10.1016/j.energy.2018.04.185", "venue": "Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "availability": "source_card", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X08_SOURCE_2018_Energy_Wind_Storage_Stochastic.md",
        "source_url": "https://doi.org/10.1016/j.energy.2018.04.185",
        "reason_no_pdf": "The publisher abstract was verified, but no stable public full text was found.",
    },
    {
        "id": "X14",
        "title": "Wind-Battery Pool Optimal Bidding in German Energy and Secondary Control Reserve Markets",
        "date": "2024", "doi": "10.1109/EEM60825.2024.10608835", "venue": "20th International Conference on the European Energy Market",
        "publication_status": "peer-reviewed conference paper", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "availability": "source_card", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X14_SOURCE_2024_EEM_Wind_Battery_aFRR.md",
        "source_url": "https://doi.org/10.1109/EEM60825.2024.10608835",
        "reason_no_pdf": "The full text was read from a copy the user obtained through an institutional IEEE subscription; it is not open access and cannot be re-fetched by the unauthenticated collector, so the library keeps a source card instead of an unreproducible PDF.",
    },
    {
        "id": "X16",
        "title": "Optimal Scheduling of Energy Storage under Forecast Uncertainties",
        "date": "2017", "doi": "10.1049/iet-gtd.2017.0037", "venue": "IET Generation, Transmission & Distribution",
        "publication_status": "peer-reviewed journal article", "quality_tier": "peer_reviewed_specialized", "primary_problem": "intersection_frontier",
        "availability": "source_card", "file": "01_intersection_frontier/02_peer_reviewed_specialized/X16_SOURCE_2017_IET_Storage_Forecast_Uncertainty.md",
        "source_url": "https://doi.org/10.1049/iet-gtd.2017.0037",
        "reason_no_pdf": "The Wiley PDF endpoint returned an access-control page to the unauthenticated collector.",
    },
    {
        "id": "X18",
        "title": "Revenue Stacking, Dispatch Optimisation, and Economic Viability of PV-BESS Plants: A Spanish Case Study",
        "date": "2026", "doi": "10.2139/ssrn.6960523", "venue": "SSRN",
        "publication_status": "posted content; no formal venue verified by the research cutoff", "quality_tier": "preprint", "primary_problem": "intersection_frontier",
        "availability": "source_card", "file": "01_intersection_frontier/03_preprints/X18_SOURCE_2026_SSRN_Spanish_PV_BESS.md",
        "source_url": "https://doi.org/10.2139/ssrn.6960523",
        "reason_no_pdf": "The SSRN page and PDF endpoint were blocked by access controls, so the collector stores a verified source card.",
    },
    {
        "id": "F18",
        "title": "Forecasting the Intermittent Demand for Slow-Moving Inventories: A Modelling Approach",
        "date": "2012", "doi": "10.1016/j.ijforecast.2011.03.009", "venue": "International Journal of Forecasting",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "availability": "source_card", "file": "02_partial_forecasting_uncertainty/01_top_journals/F18_SOURCE_2012_IJF_Intermittent_Demand.md",
        "source_url": "https://doi.org/10.1016/j.ijforecast.2011.03.009",
        "reason_no_pdf": "No stable public full text was verified for automated collection; the publisher record and DOI remain reproducible.",
    },
    {
        "id": "F25",
        "title": "Probabilistic Load Forecasting Considering Temporal Correlation: Online Models for the Prediction of Households' Electrical Load",
        "date": "2021", "doi": "10.1016/j.apenergy.2021.117594", "venue": "Applied Energy",
        "publication_status": "peer-reviewed journal article", "quality_tier": "top_journal", "primary_problem": "forecasting_uncertainty",
        "availability": "source_card", "file": "02_partial_forecasting_uncertainty/01_top_journals/F25_SOURCE_2021_Applied_Energy_Online_Quantile_Copula.md",
        "source_url": "https://doi.org/10.1016/j.apenergy.2021.117594",
        "reason_no_pdf": "No stable public full text was verified for automated collection; the publisher record and DOI remain reproducible.",
    },
    {
        "id": "M10",
        "title": "Designing the Future Electricity Spot Market with High Renewables via Reliable Simulations",
        "date": "2025", "doi": "10.1038/s44287-025-00163-9", "venue": "Nature Reviews Electrical Engineering",
        "publication_status": "peer-reviewed review article", "quality_tier": "top_journal", "primary_problem": "energy_reserve_decision",
        "availability": "source_card", "file": "03_partial_energy_reserve_decision/01_top_journals/M10_SOURCE_2025_NREE_Future_Spot_Market_Design.md",
        "source_url": "https://www.nature.com/articles/s44287-025-00163-9",
        "reason_no_pdf": "The official record and abstract were verified, but no stable openly licensed PDF was verified for automated collection.",
    },
    {
        "id": "M11",
        "title": "Artificial Intelligence-Based Methods for Renewable Power System Operation",
        "date": "2024", "doi": "10.1038/s44287-024-00018-9", "venue": "Nature Reviews Electrical Engineering",
        "publication_status": "peer-reviewed review article", "quality_tier": "top_journal", "primary_problem": "energy_reserve_decision",
        "availability": "source_card", "file": "03_partial_energy_reserve_decision/01_top_journals/M11_SOURCE_2024_NREE_AI_Renewable_System_Operation.md",
        "source_url": "https://www.nature.com/articles/s44287-024-00018-9",
        "reason_no_pdf": "The official record and abstract were verified, but no stable openly licensed PDF was verified for automated collection.",
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
    "X06": ["multi_market", "probabilistic_activation", "joint_decision", "probabilistic_delivery_constraint"],
    "X07": ["multi_market", "joint_probabilistic_forecasting", "copula", "stochastic_optimization"],
    "X08": ["multi_market", "activation_forecasting", "scenario_generation", "stochastic_optimization"],
    "X09": ["multi_market", "joint_price_scenarios", "sequential_decision", "stochastic_optimization"],
    "X10": ["multi_market", "point_forecasting", "wind_scenarios", "stochastic_optimization"],
    "X11": ["multi_market", "probabilistic_forecasting", "intraday_control", "frequency_regulation"],
    "X12": ["multi_market", "independent_scenarios", "stochastic_unit_commitment", "official_code"],
    "X13": ["multi_market", "activation_uncertainty", "joint_chance_constraints", "joint_chance_delivery_constraint"],
    "X14": ["multi_market", "historical_scenarios", "cvar", "three_stage_gate_sequence"],
    "X15": ["multi_market", "point_forecasting", "stochastic_optimization", "nordic_market"],
    "X16": ["multi_market", "forecast_uncertainty", "real_time_mpc", "storage_scheduling"],
    "X17": ["multi_market", "joint_price_dynamics", "sddp", "preprint"],
    "X18": ["multi_market", "point_forecasting", "rolling_horizon", "preprint"],
    "X19": ["reserve_market", "multi_output_forecasting", "decision_focused_learning", "real_time_correction"],
    "X20": ["multi_market", "point_forecasting", "decision_value_evaluation", "preprint"],
    "X21": ["multi_service", "probabilistic_scenarios", "real_time_control", "behind_the_meter"],
    "X22": ["multi_market", "scenario_generation", "joint_decision", "danish_market"],
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
    "F16": ["forecasting_uncertainty", "electricity_price", "spatial_dependence", "nord_pool_zones"],
    "F17": ["forecasting_uncertainty", "electricity_price", "dk1_market", "tree_baseline", "intraday_trading"],
    "F18": ["forecasting_uncertainty", "intermittent_demand", "zero_mass", "hurdle_margin"],
    "F19": ["forecasting_uncertainty", "normalizing_flow", "heavy_tail", "official_code"],
    "F20": ["forecasting_uncertainty", "gaussian_copula", "low_rank", "multivariate", "official_code"],
    "F21": ["forecasting_uncertainty", "attentional_copula", "multivariate", "official_code"],
    "F22": ["forecasting_uncertainty", "attentional_copula", "multivariate", "official_code"],
    "F23": ["forecasting_uncertainty", "correlated_errors", "low_rank", "temporal_dependence", "official_code"],
    "F24": ["forecasting_uncertainty", "normalizing_flow", "multivariate", "official_code"],
    "F25": ["forecasting_uncertainty", "quantile_forecast", "gaussian_copula", "online_update"],
    "F26": ["forecasting_uncertainty", "foundation_model", "any_variate", "official_code"],
    "F27": ["forecasting_uncertainty", "foundation_model", "probabilistic_forecasting", "official_code"],
    "F28": ["forecasting_uncertainty", "conformal_risk_control", "bounded_loss", "official_code"],
    "F29": ["forecasting_uncertainty", "conformal_risk_control", "distribution_shift", "official_code"],
    "F30": ["forecasting_uncertainty", "conformal_prediction", "copula", "multi_step", "official_code"],
    "F31": ["forecasting_uncertainty", "conformal_prediction", "online_calibration", "official_code"],
    "F32": ["forecasting_uncertainty", "diffusion", "multivariate", "official_code"],
    "F33": ["forecasting_uncertainty", "diffusion", "imputation", "official_code"],
    "F34": ["forecasting_uncertainty", "joint_probabilistic_forecasting", "system_operation", "reserve_allocation", "open_data"],
    "F35": ["forecasting_uncertainty", "foundation_model", "probabilistic_forecasting", "spatial_graph", "preprint", "official_code"],
    "F36": ["forecasting_uncertainty", "foundation_model", "decision_value_evaluation", "storage_arbitrage", "preprint"],
    "F37": ["forecasting_uncertainty", "electricity_price", "conformal_prediction", "online_recalibration", "official_code"],
    "F38": ["forecasting_uncertainty", "adaptive_conformal_inference", "delayed_feedback", "preprint"],
    "M01": ["joint_decision", "market_clearing", "soc_dependent_bids"],
    "M02": ["joint_decision", "activation_aware", "degradation", "hybrid_storage"],
    "M03": ["joint_decision", "market_design", "up_down_asymmetry"],
    "M04": ["joint_decision", "frequency_activation", "degradation"],
    "M05": ["joint_decision", "bid_acceptance", "reserve_availability"],
    "M06": ["joint_decision", "scenario_economics", "day_ahead_afrr"],
    "M07": ["joint_decision", "sequential_market_baseline", "gate_closure_timing", "rolling_forecast"],
    "M08": ["joint_decision", "day_ahead_afrr", "hybrid_storage", "robust_optimization", "danish_market"],
    "M09": ["market_design", "storage_operation", "decarbonization", "system_cost", "emissions", "official_code"],
    "M10": ["market_design", "high_renewables", "market_simulation", "review"],
    "M11": ["forecasting_uncertainty", "dispatch", "deliverability_control", "electricity_market", "review"],
    "D01": ["deliverability_control", "simultaneous_services", "power_energy_budget"],
    "D02": ["deliverability_control", "converter_capability", "real_time_control"],
    "D03": ["deliverability_control", "planning_control_loop", "multi_service"],
    "D04": ["deliverability_control", "field_measurements", "capacity_fade", "home_storage", "open_data"],
    "L01": ["decision_focused_learning", "end_to_end_stochastic_optimization"],
    "L02": ["decision_focused_learning", "spo_plus", "optimization_loss"],
    "L03": ["decision_focused_learning", "single_energy_market", "storage_arbitrage"],
    "L04": ["decision_focused_learning", "strategic_storage", "market_feedback"],
    "L05": ["decision_focused_learning", "interpretable_trees"],
    "L06": ["decision_focused_learning", "single_energy_market", "predict_then_bid"],
    "L07": ["decision_focused_learning", "single_energy_market", "online_risk_calibration", "conformal_decision_theory", "storage_arbitrage"],
    "L08": ["decision_focused_learning", "conformal_risk_control", "tail_risk", "storage_operation", "official_code"],
    "L09": ["decision_focused_learning", "conformal_decision_theory", "safe_backup_policy", "preprint"],
}

QUALITY_TIERS = {
    "top_journal": "经正式发表且属于本课题锚定的领域顶级或高影响力期刊。",
    "top_conference": "经正式录用并进入可核验的国际顶会正式论文集。",
    "peer_reviewed_specialized": "正式同行评审的专业期刊或会议论文，但不作为本库的 Top 锚点。",
    "preprint": "仅可核实为预印本、工作论文或在审稿件，不计作正式同行评审成果。",
}

ENERGINET_DATASETS = [
    ("DayAheadPrices", "day_ahead_prices", ["DK1", "DK2", "DE", "NO2", "SE3", "SE4"]),
    ("AfrrReservesNordic", "afrr_capacity_market", ["DK1"]),
    ("ImbalancePrice", "imbalance_and_activation", ["DK1"]),
    ("Forecasts_Hour", "wind_solar_forecasts", ["DK1"]),
    ("ProductionConsumptionSettlement", "production_consumption_settlement", ["DK1"]),
]
ENERGINET_START = "2025-10-01"
ENERGINET_END = "2026-09-01"
OPEN_METEO_ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_DOCUMENTATION = "https://open-meteo.com/en/docs/single-runs-api"
OPEN_METEO_MODEL = "ecmwf_ifs"
OPEN_METEO_START = "2025-10-01"
OPEN_METEO_END = "2026-08-28"
OPEN_METEO_VARIABLES = [
    "temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"
]
OPEN_METEO_GRID = [
    (latitude, longitude)
    for latitude in (55.50, 56.25, 57.00)
    for longitude in (8.00, 9.00, 10.00)
]
OPEN_METEO_GATES = {
    "gate_0730": {"run_day_offset": -2, "run_hour_utc": 18, "civil_time": "07:30"},
    "gate_1200": {"run_day_offset": -1, "run_hour_utc": 0, "civil_time": "12:00"},
}
OPEN_METEO_MAX_LATENCY_HOURS = 6
OPEN_METEO_CACHE = Path("/tmp/dk1_open_meteo_gate_weather")
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


def source_card_manifest_items():
    results = []
    for item in SOURCE_CARDS:
        target = PAPER_DIR / item["file"]
        body = target.read_bytes()
        results.append(
            {
                "id": item["id"],
                "title": item["title"],
                "date": item["date"],
                "doi": item["doi"],
                "path": str(target.relative_to(ROOT)),
                "source_url": item["source_url"],
                "content_type": "text/markdown; charset=utf-8",
                "bytes": len(body),
                "sha256": sha256(body),
                "status": "recorded",
            }
        )
    return results


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
        "source_cards": source_card_manifest_items(),
    }
    (PAPER_DIR / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_paper_catalog(results)


def energinet_url(dataset, price_areas):
    query = urllib.parse.urlencode(
        {
            "start": ENERGINET_START,
            "end": ENERGINET_END,
            "filter": json.dumps({"PriceArea": price_areas}, separators=(",", ":")),
        }
    )
    return f"https://api.energidataservice.dk/dataset/{dataset}?{query}"


def collect_energinet():
    entries = []
    directory = DATA_DIR / "energinet_dk1"
    directory.mkdir(parents=True, exist_ok=True)
    for dataset, stem, price_areas in ENERGINET_DATASETS:
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

        source_url = energinet_url(dataset, price_areas)
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
                "price_areas": price_areas,
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
        print(f"OK {dataset}: {len(records)} records for {','.join(price_areas)}", flush=True)

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


def _open_meteo_run(delivery_day, gate):
    settings = OPEN_METEO_GATES[gate]
    day = datetime.fromisoformat(delivery_day)
    run_day = day + timedelta(days=settings["run_day_offset"])
    return run_day.replace(hour=settings["run_hour_utc"], tzinfo=timezone.utc)


def _open_meteo_url(run):
    query = urllib.parse.urlencode(
        {
            "latitude": ",".join(str(point[0]) for point in OPEN_METEO_GRID),
            "longitude": ",".join(str(point[1]) for point in OPEN_METEO_GRID),
            "hourly": ",".join(OPEN_METEO_VARIABLES),
            "models": OPEN_METEO_MODEL,
            "run": run.strftime("%Y-%m-%dT%H:%M"),
            "timezone": "UTC",
            "forecast_hours": 60,
            "wind_speed_unit": "ms",
        }
    )
    return f"{OPEN_METEO_ENDPOINT}?{query}"


def _gate_time(delivery_day, gate):
    settings = OPEN_METEO_GATES[gate]
    hour, minute = (int(value) for value in settings["civil_time"].split(":"))
    civil_day = datetime.fromisoformat(delivery_day) - timedelta(days=1)
    local = civil_day.replace(hour=hour, minute=minute, tzinfo=ZoneInfo("Europe/Copenhagen"))
    return local.astimezone(timezone.utc)


def _aggregate_weather(payload, delivery_day, gate, run):
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != len(OPEN_METEO_GRID):
        raise ValueError(f"{delivery_day} {gate}: unexpected weather-grid size")
    local_zone = ZoneInfo("Europe/Copenhagen")
    values = {}
    for location in locations:
        hourly = location["hourly"]
        for index, stamp_text in enumerate(hourly["time"]):
            stamp = datetime.fromisoformat(stamp_text).replace(tzinfo=timezone.utc)
            if stamp.astimezone(local_zone).date().isoformat() != delivery_day:
                continue
            bucket = values.setdefault(stamp, {name: [] for name in OPEN_METEO_VARIABLES})
            for name in OPEN_METEO_VARIABLES:
                value = hourly[name][index]
                bucket[name].append(None if value is None else float(value))
    if len(values) not in (23, 24, 25):
        raise ValueError(f"{delivery_day} {gate}: expected a complete civil day")
    available = run + timedelta(hours=OPEN_METEO_MAX_LATENCY_HOURS)
    if available > _gate_time(delivery_day, gate):
        raise ValueError(f"{delivery_day} {gate}: weather run is not gate-visible")
    return _weather_rows(values, delivery_day, gate, run, available, local_zone)


def _weather_rows(values, delivery_day, gate, run, available, local_zone):
    rows = []
    for stamp, bucket in sorted(values.items()):
        if any(len(items) != len(OPEN_METEO_GRID) for items in bucket.values()):
            raise ValueError(f"{delivery_day} {gate} {stamp}: incomplete weather grid")
        averages = {name: None if any(value is None for value in items) else mean(items)
                    for name, items in bucket.items()}
        wind_values = bucket["wind_speed_100m"]
        wind_std = None if any(value is None for value in wind_values) else pstdev(wind_values)
        rows.append(
            {
                "delivery_day": delivery_day,
                "gate": gate,
                "run_time_utc": run.isoformat().replace("+00:00", "Z"),
                "available_at_utc": available.isoformat().replace("+00:00", "Z"),
                "timestamp_utc": stamp.isoformat().replace("+00:00", "Z"),
                "timestamp_local": stamp.astimezone(local_zone).isoformat(),
                "source_grid_points": len(OPEN_METEO_GRID),
                "temperature_2m_c_mean": averages["temperature_2m"],
                "wind_speed_100m_ms_mean": averages["wind_speed_100m"],
                "wind_speed_100m_ms_std": wind_std,
                "shortwave_radiation_wm2_mean": averages["shortwave_radiation"],
                "cloud_cover_pct_mean": averages["cloud_cover"],
            }
        )
    return rows


def _download_open_meteo(task):
    delivery_day, gate = task
    run = _open_meteo_run(delivery_day, gate)
    cache = OPEN_METEO_CACHE / f"{delivery_day}_{gate}.json"
    if cache.exists():
        body = cache.read_bytes()
    else:
        body = _retrieve_open_meteo(_open_meteo_url(run))
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_bytes(body)
        temporary.replace(cache)
    return _aggregate_weather(json.loads(body), delivery_day, gate, run)


def _retrieve_open_meteo(url):
    for attempt in range(60):
        try:
            body, _, _ = retrieve(url, timeout=30)
            return body
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 59:
                raise
            message = error.read().decode("utf-8")
            if "Hourly API request limit exceeded" in message:
                now = datetime.now(timezone.utc)
                delay = 3605 - now.minute * 60 - now.second
                print(f"Open-Meteo hourly limit; resuming in {delay} seconds", flush=True)
                while delay > 0:
                    pause = min(delay, 30)
                    time.sleep(pause)
                    delay -= pause
                continue
            if "Too many concurrent requests" in message:
                print("Open-Meteo concurrent-request lock; retrying in 60 seconds", flush=True)
                time.sleep(30)
                time.sleep(30)
                continue
            retry_after = int(error.headers.get("Retry-After", "10"))
            time.sleep(min(max(retry_after, 1), 30))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 59:
                raise
            time.sleep(5)
    raise AssertionError("unreachable Open-Meteo retry loop")


def _open_meteo_metadata():
    return {
        "dataset": "ECMWF IFS gate-available weather for DK1",
        "provider": "Open-Meteo Single Runs API",
        "upstream_model": "ECMWF IFS HRES",
        "documentation": OPEN_METEO_DOCUMENTATION,
        "api_endpoint": OPEN_METEO_ENDPOINT,
        "model_parameter": OPEN_METEO_MODEL,
        "start_delivery_day": OPEN_METEO_START,
        "end_delivery_day": OPEN_METEO_END,
        "civil_timezone": "Europe/Copenhagen",
        "maximum_assumed_publication_latency_hours": OPEN_METEO_MAX_LATENCY_HOURS,
        "grid": [{"latitude": point[0], "longitude": point[1]} for point in OPEN_METEO_GRID],
        "variables": OPEN_METEO_VARIABLES,
        "output_units": {
            "temperature_2m_c_mean": "degree_Celsius",
            "wind_speed_100m_ms_mean": "m_per_s",
            "wind_speed_100m_ms_std": "m_per_s",
            "shortwave_radiation_wm2_mean": "W_per_m2",
            "cloud_cover_pct_mean": "percent",
        },
        "gate_run_rules": OPEN_METEO_GATES,
        "transformation": "Equal-weight mean over nine requested DK1 grid points; wind-speed population standard deviation is also retained.",
    }


def collect_open_meteo_weather():
    days = []
    current = datetime.fromisoformat(OPEN_METEO_START)
    final = datetime.fromisoformat(OPEN_METEO_END)
    while current <= final:
        days.append(current.date().isoformat())
        current += timedelta(days=1)
    tasks = [(day, gate) for day in days for gate in OPEN_METEO_GATES]
    rows = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for completed, batch in enumerate(pool.map(_download_open_meteo, tasks), start=1):
            rows.extend(batch)
            if completed % 50 == 0 or completed == len(tasks):
                print(f"ECMWF gate-weather requests: {completed}/{len(tasks)}", flush=True)
    directory = DATA_DIR / "open_meteo_dk1"
    directory.mkdir(parents=True, exist_ok=True)
    metadata_body = (json.dumps(_open_meteo_metadata(), indent=2) + "\n").encode("utf-8")
    metadata_path = directory / "ecmwf_gate_weather_metadata.json"
    write_bytes(metadata_path, metadata_body)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_body = output.getvalue().encode("utf-8")
    csv_path = directory / "ecmwf_gate_weather.csv"
    write_bytes(csv_path, csv_body)
    print(f"OK ECMWF gate weather: {len(rows)} hourly rows", flush=True)
    return _open_meteo_entries(metadata_path, metadata_body, csv_path, csv_body, len(tasks), len(rows))


def _open_meteo_entries(metadata_path, metadata_body, csv_path, csv_body, requests, records):
    common = {"dataset": "ECMWF IFS gate-available weather for DK1"}
    return [
        dict(common, kind="collection_metadata", path=str(metadata_path.relative_to(ROOT)),
             source_url=OPEN_METEO_DOCUMENTATION, bytes=len(metadata_body),
             sha256=sha256(metadata_body), status="generated"),
        dict(common, kind="gate-aligned_api_extract", path=str(csv_path.relative_to(ROOT)),
             source_url=OPEN_METEO_ENDPOINT, api_requests=requests, records=records,
             transformation="Exact archived runs selected before each gate; equal-grid hourly aggregation.",
             bytes=len(csv_body), sha256=sha256(csv_body), status="generated",
             license="Open-Meteo non-commercial API terms; ECMWF open-data attribution applies"),
    ]


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
        "scope": "Observed DK1 energy/aFRR/balancing data, gate-aligned ECMWF weather forecasts for DK1, a Finland reserve-price forecasting dataset, a pinned system benchmark, and a pinned EPF-DE forecasting benchmark.",
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


def collect_open_meteo_incremental():
    entries = collect_open_meteo_weather()
    manifest_path = DATA_DIR / "download_manifest.json"
    current = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    dataset = "ECMWF IFS gate-available weather for DK1"
    current = [item for item in current if item.get("dataset") != dataset]
    write_data_manifest(current + entries)


def collect_data():
    entries = []
    entries.extend(collect_energinet())
    entries.extend(collect_open_meteo_weather())
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
    parser.add_argument(
        "group",
        choices=("papers", "migrated-papers", "catalog", "epf-data", "weather-data", "data", "all"),
    )
    args = parser.parse_args()
    if args.group in ("papers", "all"):
        download_papers()
    elif args.group == "migrated-papers":
        download_papers([item for item in PAPERS if item["id"] in MIGRATED_PAPER_IDS], merge=True)
    elif args.group == "catalog":
        catalog_from_manifest()
    if args.group == "epf-data":
        collect_epf_de_incremental()
    if args.group == "weather-data":
        collect_open_meteo_incremental()
    if args.group in ("data", "all"):
        collect_data()


if __name__ == "__main__":
    main()
