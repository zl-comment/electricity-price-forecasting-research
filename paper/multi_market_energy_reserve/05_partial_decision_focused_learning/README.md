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
| L07 | [Online Storage Conformal Risk](02_peer_reviewed_specialized/L07_2026_eEnergy_Online_Storage_Conformal_Risk.pdf) | 专业同行评审 | NYISO 实时套利中以时序差分误差代理不可观测利润损失并在线校准风险 |
| L08 | [NeurIPS Conformal Risk Training](01_top_journal_or_conference/L08_2025_NeurIPS_Conformal_Risk_Training.pdf) | Top 国际会议 | 将共形风险控制纳入端到端训练，并以储能运行展示金融尾部风险目标 |
| L09 | [Conformal Decision Theory](03_preprints/L09_2024_Conformal_Decision_Theory.pdf) | 预印本 | 一般决策风险校准与安全备份切换；没有储能实证 |

P3 若启动，应比较统计损失训练、两阶段 predict-then-optimize 和决策导向训练，并保持完全相同的储能可行域与市场结算协议。P1 冻结预测器，只借用 L07/L09 的在线控制结构，不做端到端风险训练。L07 已占据“在线共形风险用于储能”的宽泛主张，但只研究实时能量套利和代理时序差分风险；L08 的正式保证依赖交换性校准设置，L09 仍是预印本且没有储能实验。两闸门能量—备用决策中对实际 aFRR 未交付、至少两日结算延迟和长期交付率的直接控制仍需单独验证。
