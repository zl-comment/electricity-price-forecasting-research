# 电力市场预测研究仓库

当前研究方向（2026-09-16）：在 aFRR 容量闸门早于日前能量闸门的 DK1 市场，研究一座价格接受者储能如何利用**延迟结算后才能观测的实际未交付损失**，在线调节备用承诺风险。S3-B 的 211 天同协议回测支持概率场景相对点预测具有决策价值，但复杂联合依赖与 12:00 容量价条件化没有稳定、独特的增益；预测侧因此冻结为决策接口和消融。P1 的待检验主贡献收缩为顺序备用—日前—实时运行中的直接交付风险控制与同协议证据，不主张首次将一般在线风险控制用于储能，也不把三阶段结构或新预测网络作为贡献。

当前执行阶段是 **S3-C 延迟反馈交付风险控制的预注册与实现**。已合并结果包括 [`s3_forecast_side/`](p1_paper/results/s3_forecast_side/)（S3-A）、[`s3_cross_market/`](p1_paper/results/s3_cross_market/) 与 [`s3_joint_surfaces/`](p1_paper/results/s3_joint_surfaces/)（预测侧补充研究），以及 [`s3_comparison/`](p1_paper/results/s3_comparison/)（S3-B，同协议决策比较）。在正式证明延迟反馈与嵌套决策族的条件前，只使用“经验校准/控制”，不使用“长期保证”。

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

当前真实序列主数据是 Energinet DK1；芬兰 aFRR 用作备用预测外部基准；RTS-GMLC 只用于透明的联合出清和物理机制验证；EPF-DE 用于论文精确基准和预测骨干的同协议迁移评测。四套数据角色不同，不拼接成一个真实系统。文献条目、全文状态、证据等级和本地数据关系分别见[论文库](paper/multi_market_energy_reserve/README.md)、[论文结果表](paper/multi_market_energy_reserve/PAPER_RESULTS.md)与[论文—数据对应表](data/multi_market_energy_reserve/paper_data_alignment.json)。

## 环境

依赖版本是结果的一部分（见 [AGENTS.md](AGENTS.md)）。实验环境是 Python 3.9 虚拟环境，按 `requirements-r1.txt` 的固定版本重建，不与任何机器路径绑定：

```bash
conda create -y -p <venv 路径> python=3.9
<venv 路径>/bin/python -m pip install --no-deps -r requirements-r1.txt
```

`data/multi_market_energy_reserve/energinet_dk1/day_ahead_prices_raw.json` 被 `.gitignore` 排除，缺失时按 `download_manifest.json` 里的 `resolved_url` 重新下载，并核对其中的 sha256。

## 资源复现

```bash
python3 scripts/collect_multi_market_resources.py all
python3 scripts/build_paper_data_alignment.py
python3 scripts/audit_multi_market_resources.py
```

论文作者报告的结果与本仓库复现结果严格分开。采集成功、校验和通过或 PDF 可读取，不代表已经复现模型指标。
