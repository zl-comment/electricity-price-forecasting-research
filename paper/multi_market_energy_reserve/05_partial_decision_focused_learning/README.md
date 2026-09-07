# 05 决策导向学习

该组解决“预测损失是否应直接对准下游决策质量”。它提供方法桥梁，但只有部分论文直接针对储能，且多数不含备用市场。

| ID | 论文 | 质量 | 本项目角色 |
|---|---|---|---|
| L01 | [NeurIPS Task-Based End-to-End Learning](01_top_journal_or_conference/L01_2017_NeurIPS_Task_Based_End_to_End.pdf) | Top 国际会议 | 任务损失端到端学习基础 |
| L02 | [Management Science Smart Predict, then Optimize](01_top_journal_or_conference/L02_2022_Management_Science_Smart_Predict_Then_Optimize.pdf) | Top 期刊 | SPO/SPO+ 理论与基线 |
| L03 | [TSG Decision-Focused Storage Arbitrage](01_top_journal_or_conference/L03_2022_TSG_Decision_Focused_Storage_Arbitrage.pdf) | Top 期刊 | 储能套利的决策导向价格预测 |
| L04 | [TSG Perturbed DFL for Strategic Storage](01_top_journal_or_conference/L04_2025_TSG_Perturbed_DFL_Strategic_Storage.pdf) | Top 期刊 | 战略储能与可微扰动优化 |
| L05 | [ICML Predict-then-Optimize Trees](01_top_journal_or_conference/L05_2020_ICML_Decision_Trees_Predict_Optimize.pdf) | Top 国际会议 | 可解释的决策导向树模型 |
| L06 | [Decision-Focused Predict-then-Bid](03_preprints/L06_2025_Decision_Focused_Predict_Then_Bid.pdf) | 预印本 | 预测—报价—出清闭环的最新近邻方法 |

本项目应比较统计损失训练、两阶段 predict-then-optimize 和决策导向训练，并保持完全相同的储能可行域与市场结算协议。
