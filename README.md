# 电力市场预测研究仓库

当前研究方向（2026-09-07）：以“午间新能源消纳—晚峰保供”为行业背景，研究储能参与电能量—备用市场时的多市场联合预测与协同决策。

## 当前入口

- [研究目标与问题边界](research-foundation-2026/MULTI_MARKET_FORECASTING_RESEARCH_TARGET_2026-09-07.md)
- [新论文与数据讨论](research-foundation-2026/MULTI_MARKET_PAPERS_AND_DATA_DISCUSSION_2026-09-07.md)
- [阶段执行计划](research-foundation-2026/Q3_NEXT_STEP_EXECUTION_PLAN_2026-09-06.md)
- [论文库](paper/multi_market_energy_reserve/README.md)与[论文结果表](paper/multi_market_energy_reserve/PAPER_RESULTS.md)
- [数据库](data/multi_market_energy_reserve/README.md)

当前真实序列主数据是 Energinet DK1；芬兰 aFRR 用作备用预测外部基准；RTS-GMLC 只用于透明的联合出清和物理机制验证；EPF-DE 用于论文精确基准和预测骨干的同协议迁移评测。四套数据角色不同，不拼接成一个真实系统。34 项文献均已记录原始数据状态和本地数据关系，见[论文—数据对应表](data/multi_market_energy_reserve/paper_data_alignment.json)。

## 资源复现

```bash
python3 scripts/collect_multi_market_resources.py all
python3 scripts/build_paper_data_alignment.py
python3 scripts/audit_multi_market_resources.py
```

论文作者报告的结果与本仓库复现结果严格分开。采集成功、校验和通过或 PDF 可读取，不代表已经复现模型指标。
