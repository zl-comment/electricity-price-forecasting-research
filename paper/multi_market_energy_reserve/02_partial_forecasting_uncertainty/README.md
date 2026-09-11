# 02 预测与不确定性

该组解决“预测什么、如何表达不确定性、怎样评价预测”，但不完整解决储能能量—备用协同决策。

| ID | 论文 | 质量 | 本项目角色 |
|---|---|---|---|
| F01 | [Applied Energy EPF Benchmark](01_top_journals/F01_2021_Applied_Energy_EPF_Benchmark.pdf) | Top 期刊 | 统一数据切分、强统计/深度基线和显著性检验 |
| F02 | [Market Integration Forecasting](01_top_journals/F02_2018_Applied_Energy_Market_Integration_Forecasting.pdf) | Top 期刊 | 跨市场信息可提高日前预测，是联合特征基线 |
| F03 | [Bayesian Intraday Forecasting](01_top_journals/F03_2025_Applied_Energy_Bayesian_Intraday_Forecasting.pdf) | Top 期刊 | 分层概率预测与市场效率分析 |
| F04 | [Probabilistic EPF Review](01_top_journals/F04_SOURCE_2018_RSER_Probabilistic_EPF_Review.md) | Top 期刊 / 来源卡 | 可靠性—锐度评价协议 |
| F05 | [Bayesian DL Price Forecasting](01_top_journals/F05_2019_Applied_Energy_Bayesian_DL_Price_Forecasting.pdf) | Top 期刊 | 贝叶斯深度概率基线 |
| F06 | [Finnish aFRR Forecasting](02_peer_reviewed_specialized/F06_2026_Energy_AI_Finnish_aFRR_Forecasting.pdf) | 专业同行评审 | 最新备用价格树模型基线与开放数据出处 |
| F07 | [Forecast Accuracy and Battery Value](03_preprints/F07_2026_Probabilistic_Forecasting_Battery_Economic_Value.pdf) | 预印本 | 检验统计排名与经济排名错位 |
| F08 | [LightGBM](01_top_conferences/F08_2017_NeurIPS_LightGBM.pdf) | Top 国际会议 | 备用价格与多目标预测的强树模型基线 |
| F09 | [NBEATSx for EPF](01_top_journals/F09_2023_IJF_NBEATSx_EPF.pdf) | Top 期刊 | 电价专用、含外生变量且可解释的深度基线 |
| F10 | [TimeXer](01_top_conferences/F10_2024_NeurIPS_TimeXer.pdf) | Top 国际会议 | 显式融合内生目标和外生市场/风光变量 |
| F11 | [CrossLinear](01_top_conferences/F11_2025_KDD_CrossLinear.pdf) | Top 国际会议 | 跨变量相关嵌入的轻量强基线 |
| F12 | [ProtoTS](01_top_conferences/F12_2026_ICLR_ProtoTS.pdf) | Top 国际会议 | 可解释原型与极端时段模式诊断 |
| F13 | [UniCA](01_top_conferences/F13_2026_ICLR_UniCA.pdf) | Top 国际会议 | 基础模型异构协变量适配；必须做污染审计 |
| F14 | [广东 DTHG-Transformer](01_top_journals/F14_SOURCE_2026_Applied_Energy_DTHG_Guangdong.md) | Top 期刊 / 来源卡 | 中国省级市场迁移和空间超图参考 |
| F15 | [源-荷-价多任务联合预测](01_top_journals/F15_SOURCE_2024_Applied_Energy_Source_Load_Price_MTL.md) | Top 期刊 / 来源卡 | RQ1 的多任务方法参考；载体耦合，非跨市场功能耦合 |
| F16 | [Nord Pool 价区空间依赖预测](01_top_journals/F16_SOURCE_2024_IJF_Nord_Pool_Spatial_Dependence.md) | Top 期刊 / 来源卡 | 跨价区空间依赖结构的外部参照；是否含 DK1、是否用 LEAR 基线未核实 |
| F17 | [DK1 Intraday LSTM Trading](02_peer_reviewed_specialized/F17_2024_Energies_DK1_Intraday_LSTM_Trading.pdf) | 专业同行评审 | DK1 电价预测已有 LightGBM、XGBoost、随机森林等树模型基线；只做电能量，不含备用 |
| F18 | [Intermittent Demand](01_top_journals/F18_SOURCE_2012_IJF_Intermittent_Demand.md) | Top 期刊 / 来源卡 | 激活量零质量与正值规模分开的边际候选 |
| F19 | [Marginal Tail-Adaptive Flows](01_top_conferences/F19_2022_ICML_Marginal_Tail_Adaptive_Flows.pdf) | Top 国际会议 | 混合轻尾/重尾边际候选 |
| F20 | [Low-Rank Gaussian Copula Processes](01_top_conferences/F20_2019_NeurIPS_Low_Rank_Gaussian_Copula.pdf) | Top 国际会议 | 低秩跨变量联合分布候选 |
| F21 | [TACTiS](01_top_conferences/F21_2022_ICML_TACTiS.pdf) | Top 国际会议 | 注意力 Copula 结构起点 |
| F22 | [TACTiS-2](01_top_conferences/F22_2024_ICLR_TACTiS_2.pdf) | Top 国际会议 | 两阶段边际—注意力 Copula 候选 |
| F23 | [Correlated Errors](01_top_conferences/F23_2024_NeurIPS_Correlated_Errors.pdf) | Top 国际会议 | 低秩同期协方差与跨期残差候选 |
| F24 | [TempFlow](01_top_conferences/F24_2021_ICLR_TempFlow.pdf) | Top 国际会议 | 条件归一化流联合场景候选 |
| F25 | [Online Quantile Copula](01_top_journals/F25_SOURCE_2021_Applied_Energy_Online_Quantile_Copula.md) | Top 期刊 / 来源卡 | 在线分位数与时间 Copula 更新候选 |
| F26 | [Moirai](01_top_conferences/F26_2024_ICML_Moirai.pdf) | Top 国际会议 | 冻结 any-variate 基础表征候选 |
| F27 | [Chronos](02_peer_reviewed_specialized/F27_2024_TMLR_Chronos.pdf) | 专业同行评审 | 冻结概率基础模型表征候选 |
| F28 | [Conformal Risk Control](01_top_conferences/F28_2024_ICLR_Conformal_Risk_Control.pdf) | Top 国际会议 | 有界单调未交付损失校准候选 |
| F29 | [Non-Exchangeable CRC](01_top_conferences/F29_2024_ICLR_Non_Exchangeable_CRC.pdf) | Top 国际会议 | 漂移下加权风险校准候选 |
| F30 | [CopulaCPTS](01_top_conferences/F30_2024_ICLR_CopulaCPTS.pdf) | Top 国际会议 | 多步联合覆盖候选 |
| F31 | [Conformal PID](01_top_conferences/F31_2023_NeurIPS_Conformal_PID.pdf) | Top 国际会议 | 在线时序校准候选 |
| F32 | [TimeGrad](01_top_conferences/F32_2021_ICML_TimeGrad.pdf) | Top 国际会议 | 自回归扩散场景的小样本风险参照 |
| F33 | [CSDI](01_top_conferences/F33_2021_NeurIPS_CSDI.pdf) | Top 国际会议 | 条件扩散结构的小样本风险参照；原任务为插补 |

这些论文支持建立独立预测、跨市场特征、概率分布、校准、树模型和外生变量深度模型基线；它们不能单独证明储能备用承诺可交付。F01 与本地 D4 是精确公开基准关系，F08--F13 只是在 D4 上做同协议迁移评测，F18--F33 只作为结构候选与风险参照，F14 的广东原始数据尚未公开取得。
