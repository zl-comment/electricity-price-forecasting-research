# 电能量—备用多市场论文库

研究截止日：2026-09-16；重建日期：2026-09-16。

本库采用两级分类：**先按论文解决的问题覆盖度与主问题归类，再按发表质量分层**。每篇论文只有一个主位置；它与其他问题的交叉关系写入问题说明和结果表，不复制 PDF。

## 总览

| 主问题 | 文献数 | Top | 专业同行评审 | 预印本 | 入口 |
|---|---:|---:|---:|---:|---|
| 交叉前沿：多市场 + 至少两个方法问题 | 22 | 2 | 14 | 6 | [01_intersection_frontier](01_intersection_frontier/README.md) |
| 预测与不确定性 | 38 | 31 | 3 | 4 | [02_partial_forecasting_uncertainty](02_partial_forecasting_uncertainty/README.md) |
| 电能量—备用协同决策 | 11 | 9 | 1 | 1 | [03_partial_energy_reserve_decision](03_partial_energy_reserve_decision/README.md) |
| 可交付性与实时控制 | 4 | 4 | 0 | 0 | [04_partial_deliverability_control](04_partial_deliverability_control/README.md) |
| 决策导向学习 | 9 | 6 | 1 | 2 | [05_partial_decision_focused_learning](05_partial_decision_focused_learning/README.md) |
| **合计** | **84** | **52** | **19** | **13** | 69 份全文 PDF + 15 张来源卡 |

这里的“Top=52”由 31 篇锚定 Top 期刊和 21 篇正式国际顶会论文组成。交叉前沿的 X06 与 X09 分别处理激活交付的模型内名义概率约束和部分联合价格场景，但都没有同时联合预测本文三个目标；X14 已按 09:00 容量、12:00 日前、实时三阶段决策但场景取独立历史抽样，X07 的 Copula 场景中容量价与激活价为常数。F18--F33 是结构候选；F34 提供系统级联合概率预测接备用分配的实证，F35--F36 是尚未同行评审的电价基础模型证据，F37--F38 分别补充在线电价区间校准和固定反馈延迟理论。它们均不作为同问题的储能能量—备用决策对比对象。

## 质量分层规则

目录层级与载体判定只采用 [CONVENTIONS.md 第 5 节](../../CONVENTIONS.md#5-文献质量分层)；本目录不另行定义。

## 与论文主线的关系

论文核心问题是：在 aFRR 容量闸门早于日前能量闸门的 DK1 市场中，一座价格接受者储能能否利用午间新能源富余时段充电，并在晚峰保留可交付的上调能力。已有工作分别给出分阶段随机决策（X14）、概率交付约束（X06、X13）、丹麦日前＋aFRR 联合优化（M08）和单一能量市场的在线储能风险控制（L07），但当前检索尚未发现这些工作共同量化后改制 DK1 的午间充电—晚峰 SOC 与上调交付路径。既有实验已表明概率场景能改善收益和聚合未交付，但尚未形成逐时路径证据。相关工作差异、允许主张和待补证据见[论文规划](../../research-foundation-2026/PAPER_PLAN.md)。

交叉区论文定义最接近工作与同协议对比方法；Top 分问题论文提供预测骨干、评分规则与可借用结构，不作为同问题对比方法。

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
