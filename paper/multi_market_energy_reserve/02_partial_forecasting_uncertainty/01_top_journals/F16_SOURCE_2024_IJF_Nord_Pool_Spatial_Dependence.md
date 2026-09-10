# F16 来源卡：Nord Pool 多价区日前电价的空间依赖预测

- 题目：*Forecasting day-ahead electricity prices with spatial dependence*
- 作者：Yifan Yang, Ju'e Guo, Yi Li, Jiandong Zhou
- 期刊：*International Journal of Forecasting* 40(3) (2024), 1255–1270
- DOI：[10.1016/j.ijforecast.2023.11.006](https://doi.org/10.1016/j.ijforecast.2023.11.006)
- 分类：预测与不确定性 / Top 期刊

论文把 Nord Pool 各价区的日前电价转为图数据，用 R-vine copula 预先刻画价区之间的空间依赖结构，作为时空图神经网络（STGNN）的邻接矩阵，对多个价区同时预测。作者报告该依赖结构与电力系统的物理特征相符，STGNN 在整体精度和日内逐小时精度上显著优于既有模型。它是跨价区依赖结构建模的外部参照，但只涉及日前电能量价格，不含备用市场、激活或储能决策。

FB2 交接文档曾称该文“在 DK1 日前电价任务中直接使用 LEAR 基线”。摘要只写明研究对象为 Nord Pool，未点名 DK1，也未列出 LEAR 等基准模型；该说法**未核实**，读到全文前不得引用。

截至 2026-09-10，DOI、卷期与页码已由 Crossref 核验，摘要取自 RePEc/IDEAS；出版社许可仅为文本挖掘许可，Semantic Scholar 标为非开放获取，未见 arXiv 版本，故只保存来源卡。
