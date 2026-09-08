# F15 来源卡：源-荷-价多任务联合预测

- 题目：*Joint forecasting of source-load-price for integrated energy system based on multi-task learning and hybrid attention mechanism*
- 作者：Li Ke, Mu Yuchen, Yang Fan, Wang Haiyang, Yan Yi, Zhang Chenghui
- 期刊：*Applied Energy* 360 (2024), 122821
- DOI：[10.1016/j.apenergy.2024.122821](https://doi.org/10.1016/j.apenergy.2024.122821)
- 配套数据：未公开
- 分类：预测与不确定性 / Top 期刊

本文用共享层 + 多列 CNN + 序贯卷积注意力模块（SCAM）+ LSTM 的多任务框架，同时预测综合能源系统的可再生出力、多能负荷和能源价格。作者报告长期预测平均 MAPE 约 4.10%，冬季短期约 3.14%，并称在计算速度和结果一致性上优于所比方法。

它是 RQ1「联合预测价值」目前能找到的最接近的多任务证据，但**只支持方法层面，不支持设定层面**：这里的"多"是同一系统内的电/热/气载体与园区能源价格，不是日前电能量市场与 aFRR 容量市场这两个**市场功能**；共享层带来的增益来自载体间的物理耦合，而非跨市场的结算与时序耦合。引用时不得与 F02 的"多市场"混用——F02 的多市场指多价区。

因此本文不构成 RQ1 的同设定占位：能量—备用双市场的联合预测仍是空白，但也意味着本文的 MAPE 数字不能作为 P1 的对比基线。

截至 2026-09-06，正式发表信息已核验；出版社全文位于访问墙之后，不能由本仓库的无认证采集器再获取，故只保留来源卡。以上数字均为论文作者报告，不是本仓库复现结果。
