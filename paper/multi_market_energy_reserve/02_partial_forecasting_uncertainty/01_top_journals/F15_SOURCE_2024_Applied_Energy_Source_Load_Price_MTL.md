# F15 来源卡：源-荷-价多任务联合预测

- 题目：*Joint forecasting of source-load-price for integrated energy system based on multi-task learning and hybrid attention mechanism*
- 作者：Li Ke, Mu Yuchen, Yang Fan, Wang Haiyang, Yan Yi, Zhang Chenghui
- 期刊：*Applied Energy* 360 (2024), 122821
- DOI：[10.1016/j.apenergy.2024.122821](https://doi.org/10.1016/j.apenergy.2024.122821)
- 配套数据：未公开
- 分类：预测与不确定性 / Top 期刊

本文用共享层 + 多列 CNN + 序贯卷积注意力模块（SCAM）+ LSTM 的多任务框架，同时预测综合能源系统的可再生出力、多能负荷和能源价格。作者报告长期预测平均 MAPE 约 4.10%，冬季短期约 3.14%，并称在计算速度和结果一致性上优于所比方法。

F15 是跨载体多任务结构参考，不是能量—备用市场的同设定证据：这里的“多”是同一系统内的电/热/气载体与园区能源价格，不是日前电能量与 aFRR 容量这两个市场功能；共享层增益来自载体间物理耦合，而非跨市场结算与时序耦合。引用时不得与 F02 的“多市场”混用，F02 的多市场指多价区。

F15 没有覆盖 P1 的顺序市场设定。截至截止日，本库未见完全相同设定的证据，但这不是对整个领域“空白”的穷尽证明；其 MAPE 也不进入 P1 的同协议对比。

截至 2026-09-06，正式发表信息已核验；出版社全文位于访问墙之后，不能由本仓库的无认证采集器再获取，故只保留来源卡。以上数字均为论文作者报告，不是本仓库复现结果。
