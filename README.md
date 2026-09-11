# 电力市场预测研究仓库

当前研究方向（2026-09-11）：在 aFRR 容量闸门早于日前能量闸门的 DK1 市场，研究一座价格接受者储能 07:30 承诺的备用能否在交割日兑现——以闸门条件联合概率场景驱动三阶段决策，按已实现未交付事件在线校准风险系数，并在同一协议、按实际结算下与重建的已发表方法比较。三阶段决策结构来自已有工作（X14），不作为贡献。市场选择依据是闸门结构而非数据可得性，理由见研究框架第 2 节。

当前执行任务：[S3-A 预测侧完整实验](TASK_S3_FORECAST_SIDE.md)；它合并后执行 [S3-B 已发表方法同协议对比](TASK_S3_COMPARISON.md)。两份任务书在对应 PR 合并后删除。

## 当前入口

研究叙述只有三份，各回答一个问题，互不重述：

- [研究框架](research-foundation-2026/RESEARCH_FRAMEWORK.md)：研究什么。命题、市场选择依据、RQ1–RQ5、三类对照臂、证据边界
- [执行计划](research-foundation-2026/EXECUTION_PLAN.md)：怎么做。S0–S5 阶段、验收条件、当前进度
- [论文规划](research-foundation-2026/PAPER_PLAN.md)：写成什么。P1–P4 分工、P1 证据状态、题目、停止条件

规则与参考：

- [AGENTS.md](AGENTS.md) 代理行为契约、[CONVENTIONS.md](CONVENTIONS.md) 术语与编号的唯一定义处
- [数据字典](data/multi_market_energy_reserve/DATA_DICTIONARY.md)：DK1 的市场时序、结构、缺口、可见性
- [论文库](paper/multi_market_energy_reserve/README.md)与[论文结果表](paper/multi_market_energy_reserve/PAPER_RESULTS.md)
- [数据库](data/multi_market_energy_reserve/README.md)
- [行业背景与已作废讨论](background/)：只作动机引用，不含研究主张

当前真实序列主数据是 Energinet DK1；芬兰 aFRR 用作备用预测外部基准；RTS-GMLC 只用于透明的联合出清和物理机制验证；EPF-DE 用于论文精确基准和预测骨干的同协议迁移评测。四套数据角色不同，不拼接成一个真实系统。39 项文献均已记录原始数据状态和本地数据关系，见[论文—数据对应表](data/multi_market_energy_reserve/paper_data_alignment.json)。

## 资源复现

```bash
python3 scripts/collect_multi_market_resources.py all
python3 scripts/build_paper_data_alignment.py
python3 scripts/audit_multi_market_resources.py
```

论文作者报告的结果与本仓库复现结果严格分开。采集成功、校验和通过或 PDF 可读取，不代表已经复现模型指标。
