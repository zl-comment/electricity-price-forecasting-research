# 电力市场预测研究仓库

当前分支研究一座价格接受者储能在 DK1 市场中如何利用午间新能源富余时段充电，并在晚峰保留可交付的 aFRR 上调能力。市场边界限定为日前能量与上调 aFRR；下调 aFRR、FCR、mFRR、日内市场和系统级联合出清不进入本分支。

| 层级 | 本分支的问题 |
|---|---|
| 应用 | 储能能否形成午间新能源富余时段充电、晚峰提供上调能力的跨时段转移？ |
| 市场 | aFRR 容量在 D-1 07:30 先关闭、日前能量在 D-1 12:00 后关闭，这一顺序如何改变功率与 SOC 分配？ |
| 方法 | 在价格与激活不确定、交付结果延迟可见时，什么决策方式能改善收益、午间充电和晚峰交付的权衡？ |

DK1 数据没有真实弃风弃光量、未供电量、储能报价、成交、SOC 或设备遥测。本分支只能评价模型生成的新能源富余时段对齐充电、SOC 转移、上调备用承诺与实际激活交付，不能声称降低了实网弃电或晚峰失负荷。术语、证据等级和免责规则见 [CONVENTIONS.md](CONVENTIONS.md)。

## 当前入口

- [研究框架](research-foundation-2026/RESEARCH_FRAMEWORK.md)：要解决的问题、数据边界、已有证据与开放问题；
- [已完成实验与开放研究记录](research-foundation-2026/EXECUTION_PLAN.md)：只记录已经完成的实验及尚可探索的研究维度，不规定执行顺序；
- [论文规划](research-foundation-2026/PAPER_PLAN.md)：论文叙事、相关工作边界和允许的主张；
- [DK1 数据字典](data/multi_market_energy_reserve/DATA_DICTIONARY.md)：市场时序、字段、缺失、单位和信息可见性；
- [已有实验结果](p1_paper/results/)：S0--S3-B 的审计、预测和决策结果；
- [论文与数据目录](paper/multi_market_energy_reserve/README.md)：本地论文、来源、校验和及论文—数据关系；
- [行业背景](background/ELECTRICITY_MARKET_POLICY_AND_INDUSTRY_PROBLEMS_2026-09-06.md)：只作问题动机，不作为实证结论。

## 环境

依赖版本属于结果的一部分。实验环境按 `requirements-r1.txt` 的固定版本重建，不得为运行方便静默升级。

```bash
conda create -y -p <venv 路径> python=3.9
<venv 路径>/bin/python -m pip install --no-deps -r requirements-r1.txt
```

## 资源复现

```bash
python3 scripts/collect_multi_market_resources.py all
python3 scripts/build_paper_data_alignment.py
python3 scripts/audit_multi_market_resources.py
```

`data/multi_market_energy_reserve/energinet_dk1/day_ahead_prices_raw.json` 被 `.gitignore` 排除，缺失时按 `download_manifest.json` 中的 `resolved_url` 重新下载并核对 SHA-256。论文作者报告结果与本仓库实验结果分开记录。
