# 03 电能量—备用协同决策

该组解决“给定价格/场景后怎样联合报价、调度和出清”，是本项目优化层的 Top 强基线，但通常不以联合预测为主要贡献。

| ID | 论文 | 质量 | 本项目角色 |
|---|---|---|---|
| M01 | [TPWRS SOC-Dependent Energy-Reserve Bids](01_top_journals/M01_2025_TPWRS_Energy_Reserve_Cooptimization_SoC_Bids.pdf) | Top 期刊 | SOC 相关报价、能量—备用联合出清 |
| M02 | [Applied Energy Wind-HESS Multi-Market](01_top_journals/M02_SOURCE_2026_Applied_Energy_Multi_Market_Wind_HESS.md) | Top 期刊 / 来源卡 | 激活感知、退化与多市场收益堆叠 |
| M03 | [BESS and Ancillary Market Design](01_top_journals/M03_2024_Applied_Energy_BESS_Ancillary_Market_Design.pdf) | Top 期刊 | 市场规则与上下调非对称性 |
| M04 | [Stacked Energy and Reserve Revenues](01_top_journals/M04_2023_Applied_Energy_Stacked_Revenues.pdf) | Top 期刊 | 联合收益、调频能量与寿命经济性 |
| M05 | [Integrated Scheduling and Bidding](01_top_journals/M05_SOURCE_2022_Applied_Energy_Integrated_Scheduling_Bidding.md) | Top 期刊 / 来源卡 | 非前视约束与备用可用性保证 |
| M06 | [DA and aFRR Economic Evaluation](01_top_journals/M06_2021_Applied_Energy_DA_aFRR_Economic_Evaluation.pdf) | Top 期刊 | 日前+aFRR 联合经济评价与场景基线 |
| M07 | [Sequential Market Optimization FTM BESS](03_preprints/M07_2026_Sequential_Market_Optimization_FTM_BESS.pdf) | 预印本 | 门限时序 + 滚动预测下的顺序市场决策基线规范 |
| M08 | [丹麦风电—混合储能日前加 aFRR](02_peer_reviewed_specialized/M08_2024_JES_Wind_HESS_DA_aFRR.pdf) | 专业同行评审 | 丹麦风电、锂电与液流电池参与日前和 aFRR 的鲁棒 MILP；假设风电与价格可完美预测 |
| M09 | [Joule Storage Market Design](01_top_journals/M09_2023_Joule_Storage_Market_Design.pdf) | Top 期刊 | 比较日前与实时市场设计如何改变储能的系统成本和减排作用 |
| M10 | [Future Spot Market Design Review](01_top_journals/M10_SOURCE_2025_NREE_Future_Spot_Market_Design.md) | Top 期刊 / 来源卡 | 高比例新能源市场机制、仿真与评价综述；本文只据官方摘要 |
| M11 | [AI for Renewable Power-System Operation Review](01_top_journals/M11_SOURCE_2024_NREE_AI_Renewable_System_Operation.md) | Top 期刊 / 来源卡 | 串联预测、经济调度、实时控制与市场的系统综述；本文只据官方摘要 |

本组与交叉区的 X14 共同说明：“联合优化能增加收益”与“按闸门分阶段随机决策”本身都已经不新。S3-B 已显示概率场景相对点预测决策有价值，但复杂依赖结构没有稳定优势，固定风险系数也没有稳定兑现名义尾部覆盖。P1 的候选贡献因此集中于按已实现未交付事件在线控制风险、后改制 DK1 上的实际结算，以及同信息集、同可行域的协议比较；是否成立由 S3-C 决定，见[论文规划](../../../research-foundation-2026/PAPER_PLAN.md)第 6 节。M09--M11 用于解释市场设计和系统运行含义，不替代参与者层的决策基线。
