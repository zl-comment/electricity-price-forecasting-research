# 电力市场预测研究仓库

当前研究方向（2026-09-08）：在 aFRR 容量闸门早于日前能量闸门的市场时序下，研究一座价格接受者储能的电能量—备用联合概率预测与联合承诺决策。市场选择依据是闸门结构而非数据可得性，理由见研究目标第 2 节。

## 当前入口

- [研究目标与问题边界](research-foundation-2026/MULTI_MARKET_FORECASTING_RESEARCH_TARGET_2026-09-07.md)
- [数据字典](research-foundation-2026/MULTI_MARKET_DATA_DICTIONARY.md)
- [阶段执行计划](research-foundation-2026/Q3_NEXT_STEP_EXECUTION_PLAN_2026-09-06.md)
- [论文库](paper/multi_market_energy_reserve/README.md)与[论文结果表](paper/multi_market_energy_reserve/PAPER_RESULTS.md)
- [数据库](data/multi_market_energy_reserve/README.md)
- [行业背景与已作废讨论](background/)：只作动机引用，不含研究主张

当前真实序列主数据是 Energinet DK1；芬兰 aFRR 用作备用预测外部基准；RTS-GMLC 只用于透明的联合出清和物理机制验证；EPF-DE 用于论文精确基准和预测骨干的同协议迁移评测。四套数据角色不同，不拼接成一个真实系统。36 项文献均已记录原始数据状态和本地数据关系，见[论文—数据对应表](data/multi_market_energy_reserve/paper_data_alignment.json)。

## 资源复现

```bash
python3 scripts/collect_multi_market_resources.py all
python3 scripts/build_paper_data_alignment.py
python3 scripts/audit_multi_market_resources.py
```

论文作者报告的结果与本仓库复现结果严格分开。采集成功、校验和通过或 PDF 可读取，不代表已经复现模型指标。
