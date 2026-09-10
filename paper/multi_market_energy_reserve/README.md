# 电能量—备用多市场论文库

研究截止日：2026-09-10；重建日期：2026-09-10。

本库采用两级分类：**先按论文解决的问题覆盖度与主问题归类，再按发表质量分层**。每篇论文只有一个主位置；它与其他问题的交叉关系写入问题说明和结果表，不复制 PDF。

## 总览

| 主问题 | 文献数 | Top | 专业同行评审 | 预印本 | 入口 |
|---|---:|---:|---:|---:|---|
| 交叉前沿：多市场 + 至少两个方法问题 | 22 | 2 | 14 | 6 | [01_intersection_frontier](01_intersection_frontier/README.md) |
| 预测与不确定性 | 33 | 29 | 3 | 1 | [02_partial_forecasting_uncertainty](02_partial_forecasting_uncertainty/README.md) |
| 电能量—备用协同决策 | 8 | 6 | 1 | 1 | [03_partial_energy_reserve_decision](03_partial_energy_reserve_decision/README.md) |
| 可交付性与实时控制 | 3 | 3 | 0 | 0 | [04_partial_deliverability_control](04_partial_deliverability_control/README.md) |
| 决策导向学习 | 6 | 5 | 0 | 1 | [05_partial_decision_focused_learning](05_partial_decision_focused_learning/README.md) |
| **合计** | **72** | **45** | **18** | **9** | 58 份全文 PDF + 14 张来源卡 |

这里的“Top=45”由 25 篇锚定 Top 期刊和 20 篇正式国际顶会论文组成。交叉前沿的 X06 与 X09 分别处理激活交付概率保证和部分联合价格场景，但都没有同时联合预测本文三个目标；F18--F33 是结构候选，不作为同问题对比对象。

## 质量分层规则

目录层级与载体判定只采用 [CONVENTIONS.md 第 5 节](../../CONVENTIONS.md#5-文献质量分层)；本目录不另行定义。

## 与论文主线的关系

论文核心问题是：在 aFRR 容量闸门早于日前能量闸门的固定时序下，如何联合刻画日前电价、备用容量价格及激活不确定性，并把联合预测送入具有连续 SOC、备用持续能量和可交付约束的储能联合承诺决策，在同一闭环中评价利润、尾部风险和保供/履约失败。

当前证据结构很清楚：Top 文献分别把预测、协同优化、控制和决策导向学习做得较深，但尚缺一篇把四者严谨接通的工作。项目应以这些 Top 分问题论文作强基线，以交叉区论文作最接近工作。

逐篇作者报告结果和不可外推边界见[论文结果表](PAPER_RESULTS.md)；机器可读的分类、状态、本地位置及 `data_links` 见[总目录](catalog.json)；逐篇数据对应关系见[论文—数据对应表](../../data/multi_market_energy_reserve/paper_data_alignment.json)；PDF 来源与 SHA-256 见[下载清单](download_manifest.json)；完整性与文本可提取性见[PDF 审计](pdf_audit.json)。

数据对应不等于全部严格复现：目前 F01↔EPF-DE、F06↔芬兰 aFRR 是已经落地的论文精确公开数据；其余论文均明确标成同市场部分字段、当前项目迁移基准、机制基准或原始数据未公开/未取得。这样既保证每篇论文有数据去向，也不会把“可用于测试该方法”偷换成“取得原论文完整数据”。

## 复现与维护

```bash
python3 scripts/collect_multi_market_resources.py papers
python3 scripts/collect_multi_market_resources.py catalog
python3 scripts/collect_multi_market_resources.py epf-data
python3 scripts/build_paper_data_alignment.py
python3 scripts/audit_multi_market_resources.py
```

采集成功只证明文件、版本和校验和可核验；论文数字始终标记为 `paper-reported`，不能写成本仓库已经复现。
