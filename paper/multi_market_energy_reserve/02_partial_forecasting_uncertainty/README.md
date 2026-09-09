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

这些论文支持建立独立预测、跨市场特征、概率分布、校准、树模型和外生变量深度模型基线；它们不能单独证明储能备用承诺可交付。F01 与本地 D4 是精确公开基准关系，F08--F13 只是在 D4 上做同协议迁移评测，F14 的广东原始数据尚未公开取得。
