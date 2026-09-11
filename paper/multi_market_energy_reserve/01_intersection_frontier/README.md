# 01 交叉前沿

纳入标准：处于多市场场景，并交叉覆盖预测/不确定性、联合决策、可交付性或闭环评价中的至少两个方面。该组最接近论文题目，但每篇仍只覆盖完整证据链的一部分；截至 2026-09-10，没有一篇同时联合预测 DK1 日前价、aFRR 容量价与激活量。

| ID | 论文 | 质量 | 已覆盖 | 仍缺少 |
|---|---|---|---|---|
| X01 | [Field Performance Multi-Market BESS](03_preprints/X01_2026_Field_Performance_Multi_Market_BESS.pdf) | 预印本 | 真实设备参数 + 丹麦多市场优化 | 非联合预测；正式会议论文集尚未独立核验 |
| X02 | [Integrated Forecasting and Optimization](02_peer_reviewed_specialized/X02_2026_Integrated_Forecasting_Spanish_Markets.pdf) | 专业同行评审 | 多产品预测接入 BESS 优化 | 未显式学习联合概率依赖与激活风险 |
| X03 | [Conformal DA and Balancing Forecasting](02_peer_reviewed_specialized/X03_2025_Conformal_DA_Balancing_Forecasting.pdf) | 专业同行评审 | 跨结算预测区间 + 交易评价 | 不是能量—备用容量联合决策 |
| X04 | [Joint Intraday and FCR Bidding](03_preprints/X04_2025_Joint_Intraday_FCR_Bidding.pdf) | 预印本 | 日内—FCR 联合竞价与动态切换 | 预测层与严格闭环履约不足 |
| X05 | [Storage Energy and Ancillary Services](03_preprints/X05_2026_Storage_Energy_Ancillary_Services.pdf) | 预印本 | 鲁棒承诺、离散时间与可交付性 | 缺实证联合预测比较 |
| X06 | [Data-Driven DA and Reserve Scheduling](01_top_journal_or_conference/X06_2021_TPWRS_Data_Driven_DA_Reserve.pdf) | Top 期刊 | 激活经验分布、机会约束与交付保证 | 未联合预测三类市场目标 |
| X07 | [Forecast-Driven VPP Stochastic Scheduling](02_peer_reviewed_specialized/X07_SOURCE_2022_IEEE_Systems_VPP_Stochastic.md) | 专业同行评审 / 来源卡 | Copula 多变量轨迹与随机调度 | 市场、数据期与全文细节未核实 |
| X08 | [Data-Driven Wind-Storage Stochastic Approach](02_peer_reviewed_specialized/X08_SOURCE_2018_Energy_Wind_Storage_Stochastic.md) | 专业同行评审 / 来源卡 | 调节需求预测、场景生成与随机优化 | 市场、数据期与跨变量依赖未核实 |
| X09 | [Sequential Stochastic Frameworks](01_top_journal_or_conference/X09_2021_Applied_Energy_Sequential_Stochastic.pdf) | Top 期刊 | 七类市场价格联合场景与顺序随机决策 | 风、价格、调节需求三块相互独立 |
| X10 | [Wind-Battery Self-Scheduling](02_peer_reviewed_specialized/X10_2023_JES_Wind_Battery_Self_Scheduling.pdf) | 专业同行评审 | 点价格预测、风电场景与两阶段排程 | 原始数据保密；非联合概率预测 |
| X11 | [BESS-PV Frequency Regulation Planning](02_peer_reviewed_specialized/X11_2020_TSTE_BESS_PV_Frequency_Regulation.pdf) | 专业同行评审 | PV/调频能量概率预测与日内更新 | 误差跨时段独立；产品不是 aFRR |
| X12 | [PyPSA Stochastic Unit Commitment](02_peer_reviewed_specialized/X12_2024_IFAC_PyPSA_Stochastic_UC.pdf) | 专业同行评审 | DA/aFRR 独立场景、随机机组组合与开放代码 | 三个目标按变量独立处理 |
| X13 | [BESS aFRR Delivery Guarantees](02_peer_reviewed_specialized/X13_2024_EPSR_BESS_aFRR_Delivery_Guarantees.pdf) | 专业同行评审 | 激活风险表征与联合机会约束 | 价格预测和完整数据期未核实 |
| X14 | [German Wind-Battery aFRR Bidding](02_peer_reviewed_specialized/X14_SOURCE_2024_EEM_Wind_Battery_aFRR.md) | 专业同行评审 / 来源卡 | 三阶段随机竞价与 CVaR | 场景生成方式未核实 |
| X15 | [Nordic Multi-Market BESS](02_peer_reviewed_specialized/X15_2025_ISGT_Nordic_Multi_Market_BESS.pdf) | 专业同行评审 | GAM 点预测与随机优化 | 使用 2019–2021 FCR 数据，非改制后 aFRR |
| X16 | [Storage under Forecast Uncertainties](02_peer_reviewed_specialized/X16_SOURCE_2017_IET_Storage_Forecast_Uncertainty.md) | 专业同行评审 / 来源卡 | 日前能量/调频规划与实时 MPC | 市场、数据年与联合场景未核实 |
| X17 | [Coordinated Reserve and Spot Trading](03_preprints/X17_2024_Coordinated_Reserve_Spot_Trading.pdf) | 预印本 | DA/ID 多维 Markov 依赖与 SDDP | FCR 与 spot 块独立；载体未核验 |
| X18 | [Spanish PV-BESS Revenue Stacking](03_preprints/X18_SOURCE_2026_SSRN_Spanish_PV_BESS.md) | 预印本 / 来源卡 | 多市场滚动时域调度 | 仅点预测；未正式同行评审 |
| X19 | [Reserve Decision-Focused Learning](02_peer_reviewed_specialized/X19_2025_SEGAN_Reserve_Decision_Focused.pdf) | 专业同行评审 | 多输出点预测、DFL 与实时纠偏 | 随机切分且插值；非联合概率预测 |
| X20 | [Forecast Accuracy and Multi-Market Decisions](03_preprints/X20_2026_Forecast_Accuracy_Multi_Market_BESS.pdf) | 预印本 | 预测排序与决策收益的闭环评价 | 仅点预测；载体未核验 |
| X21 | [BTM BESS Stacked Services](02_peer_reviewed_specialized/X21_2026_Smart_Energy_BTM_BESS_Stacked_Services.pdf) | 专业同行评审 | aFRR 场景、容量分配与实时控制 | 无批发日前能量市场 |
| X22 | [Danish Wind and Retired Batteries](02_peer_reviewed_specialized/X22_2020_IJEPES_Wind_Retired_Battery.pdf) | 专业同行评审 | 丹麦 spot/FCR-N 场景与两阶段随机决策 | 跨变量依赖未核实；产品不是 aFRR |

本组的用途是定义“最接近工作”；方法和强基线必须继续从后面四个分问题的 Top 文献中组合。
