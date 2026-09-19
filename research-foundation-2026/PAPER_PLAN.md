# 论文规划：GRASCO——DK1 两闸门市场中的新能源对齐储能联合决策

## 0. 版本说明

本文件对齐到 `paper/dk1-midday-evening-afrr` 分支上 2026-09-18 完成的正文重写（提交
`978e674`→`bd53c66`），即 [`paper/dk1_midday_evening_afrr.tex`](../paper/dk1_midday_evening_afrr.tex)
当前稿。此前版本的本文件仍停留在没有 GRASCO/GCJS/WLPF 命名、没有 ρ 对齐信用的旧阶段，
与已经写出的正文不再一致，因此在此重写。

**已知的反向缺口（正文尚未包含、但已有数据支持）：** 原规划第 5.1 节"市场时段事实"
（午间/晚峰日前价、上调容量价、风光/负荷比及 89.9%/93.7% 的按日占比、0.923 的相关系数）
目前**没有出现在 `dk1_midday_evening_afrr.tex` 正文的任何一节或任何表格中**（已用
`grep` 核实关键数字和小节标题均不存在）。这些数字来自 `s0_dk1_audit`/`s1_dk1_coupling`
的既有结果，仍然成立，只是尚未被写回当前这版正文。是否需要把它作为 Introduction/Motivation
的证据段落补回去，由作者决定；本文件把它保留为"已核验但未入稿"的事实。

## 1. 题目与方法命名

英文题目（当前正文口径）：

> Gate-Aware Renewable-Aligned Stochastic Battery Co-Optimization in DK1:
> Forecasts, Decisions, and Settlement Evidence

中文叙事题目（仍是论文的问题动机，未必是最终中文标题）：

> 从午间新能源富余充电到晚峰上调备用：DK1 顺序市场中的储能联合决策

方法体系命名为 **GRASCO**（Gate-aware Renewable-Alignment Stochastic Co-Optimization），
由两个本文提出的子模型组成：

| 简称 | 全称 | 角色 |
|---|---|---|
| GCJS | Gate-Conditioned Joint Scenarios | 闸门条件下的市场联合概率场景；实验中冻结使用的是其中经验上更强的合并高斯 copula（PGC） |
| WLPF | Weather--Load Proxy Forecast | 闸门可见天气 → 风光/负荷分量 → VRE-above-load 代理量的预测 |

GRASCO 有四个实验变体 GRASCO-0/25/50/100，后缀是 EUR/MWh 计的对齐信用 ρ；
GRASCO-0 是纯市场对照臂（不含对齐信用），25/50/100 是预注册的敏感性臂。GRASCO 一词
指"两闸门决策架构 + 新能源对齐机制"的整体，不特指某一种 copula 估计器。

题目中的"新能源富余充电"指模型充电与 DK1 的 VRE-above-load 代理量对齐，**不表示已经
观测到真实弃电削减**（这一边界贯穿全文，见第 7 节）。

## 2. 论文要回答的问题

> 在 aFRR 容量承诺早于日前能量交易的 DK1 市场中，储能能否利用午间新能源富余时段充电，
> 并在晚峰保留可交付的上调能力；概率信息和 GRASCO 的新能源对齐信号能否改善午间吸收、
> 晚峰交付、收益与风险之间的权衡？

三个子问题及其**当前回答状态**：

| 子问题 | 论文中的作用 | 当前状态 |
|---|---|---|
| DK1 是否存在午间新能源富余与晚峰上调价值的时段错配 | 经验事实与研究动机 | 数字已算出（第 0 节所述的反向缺口），但**尚未写回当前正文** |
| 联合决策是否形成午间充电—晚峰 SOC 与备用转移 | 核心应用结果 | **部分确认**：午间充电、预测/实际重叠度、17:00 SOC 随 ρ 单调上升；但晚峰备用与交付**不**随 ρ 显著变化，因为备用在 07:30 已冻结且已接近电池功率上限 |
| 概率信息和实际交付反馈如何改变收益—交付权衡 | 方法与风险结果 | 概率场景相对点预测显著增收降未交付（已确认）；固定 CVaR 权重不能校准实际交付（已确认）；基于延迟实损的自适应风险处理仍未检验（未回答） |

## 3. 数据与案例

| 项目 | 论文口径 |
|---|---|
| 市场 | DK1 日前能量与 aFRR 上调容量/激活 |
| 数据期（市场决策） | 2025-10-01 至 2026-08-28 的共同可结算窗口 |
| 数据期（新能源代理） | 2025-10-01 至 2026-08-27，330 个完整交割日、7,920 个有效小时 |
| 市场决策评价期 | 211 个统一结算交割日（自 2026-01-28 起） |
| 天气公共评价期 | 209 个风光输入完整日 |
| WLPF 预测评价期 | 208 天、每闸门 4,990 个有效小时 |
| GRASCO 匹配评价期 | 206 天（两个闸门 24 小时输入均完整） |
| 时间分辨率 | 能量与激活 15 分钟，容量与代理量 1 小时 |
| 储能 | 4 MW / 8 MWh，往返效率 0.85，单向效率 √0.85，初始与终端 SOC 4.4 MWh |
| 场景 | 生成 100 个，供优化器使用 50 个等权场景 |
| 随机种子 | 预测评分用种子 1/2/3；决策导出用种子 1 |
| 风险 | 下尾 CVaR，置信水平 0.95，χ∈{0,0.5}（W\&SPP 额外含 0.1） |
| GRASCO 灵敏度 | 对齐信用 ρ∈{0,25,50,100} EUR/MWh；实现利润不含该信用 |
| 信息时序 | 07:30 备用容量承诺，09:20 前结果，12:00 日前能量决策，D 日交付 |
| 不含 | 真实储能动作、真实弃电、未供电量、系统价格反事实、电量来源追踪 |

数据事实全部引用[数据字典](../data/multi_market_energy_reserve/DATA_DICTIONARY.md)，
避免在正文不同章节重复定义。

## 4. 已有论文与本文边界

| 相关工作 | 已有贡献 | 本文不能重复声称 | 尚未共同覆盖的交叉点 |
|---|---|---|---|
| MPF，Al-Lawati 等，2021（正文引用键 `allawati2021`） | 数据驱动的多阶段随机框架、顺序场景更新 | 顺序场景更新不是新意 | 反转为 DK1 备用先于日前的顺序，去掉风电场/日内市场/下调 |
| DD-CC，Toubeau 等，2021（`toubeau2021`） | 数据驱动的激活分布与概率交付约束 | 概率交付保证不是新意 | 用 DK1 冻结小时分位数近似替代比利时双边市场与联合交付保证 |
| W\&SPP，Mancini 等，2024（`mancini2024`） | 三阶段风储投标、历史场景与 CVaR | 三阶段和 CVaR 不是新意 | 用 DK1 逐小时 aFRR 与统一一小时假设替代德国四小时产品与 1.5 小时备用时长 |
| Rockafellar & Uryasev，2000（`rockafellar2000`） | CVaR 的数学定义 | CVaR 本身不是新意 | 应用于两闸门顺序市场的实际未交付反馈 |

基于当前文献库，尚未发现完整结合以下内容的工作：

1. 后改制 DK1 的 07:30 备用容量—12:00 日前能量顺序；
2. 午间新能源富余时段充电到晚峰上调能力的路径量化——**这一点现在已有量化证据**：
   市场专用方法的午间-代理量对齐能量占午间总代理量的 0.0116%--0.1409%，GRASCO 的正
   ρ 敏感性臂把这一比例提高到 0.1906%（GRASCO-100 的上限），但仍不足 0.2%，
   因此仍然只是"充电时序与代理量对齐"，不构成弃电削减证据；
3. 使用实际激活结算储能交付与未交付；
4. 在交付结果延迟可见时，研究固定风险与自适应风险处理的差异（仍未完成，见第 6 节）。

该判断限定于当前检索范围。论文使用"现有近邻工作尚未共同覆盖"，不使用无法穷尽证明的"首次"。

## 5. 当前正文已写入的结果

以下按主题给出结果摘要与核验来源；**逐行数字以 `dk1_midday_evening_afrr.tex` 对应表格
为准**，本节不重复维护小数位，只维护"哪类结论已经成立"。所有数字已经用
`p1_paper/results/{s3_comparison,s3_forecast_side}/*.csv|*.json` 逐一核对，未发现不一致。

### 5.1 预测结果（forecast side）

- 跨小时合并优于独立日前价/容量价场景（pinball 显著改善，p<0.001）。
- 低秩高斯（GCJS 的核心构件）相对独立分位数场景显著改善 Energy Score（p=0.0016），
  但相对经验依赖场景（p=0.058）和相对 X09 块场景（现称 MPF 重建，p=0.727）均未
  达到显著优势。
- WLPF 闸门可见代理预测：07:30 精确率 79.19%/召回率 61.92%，12:00 精确率
  80.66%/召回率 66.10%；12:00 相对 07:30 全时段 MAE 降低 3.06%，RMSE 降低 1.59%；
  逐小时正例判定在 4.87% 的公共小时上发生翻转。预测在能量幅度上系统性偏保守
  （07:30/12:00 分别只预测出观测代理能量的 58.61%/63.37%）。
- 详见正文 `\subsection{Marginal and joint probabilistic forecasts}` 与
  `\subsection{Gate-available proxy forecast results}`（表 `tab:gate-proxy-skill`）。

### 5.2 决策结果（市场专用臂，211 天同协议）

自制对照臂与已发表方法同协议重建（正文命名对照见下）：

| 正文简称 | 含义 | 出处/证据角色 |
|---|---|---|
| EA-PERS / PT-RES | 持久性能量套利 / 阈值备用 | present heuristic 操作基线 |
| LGBM-DO | LightGBM 点预测 + 确定性优化 | 点预测决策基线 |
| Hist-P-SO | 历史配对场景 + 随机优化 | 简单场景基线 |
| GCJS-SO / PGC-SO | 本文场景 / 合并高斯 copula + 随机优化 | 本文方法 |
| DD-CC | Toubeau 等同协议重建 | 外部比较 |
| MPF | Al-Lawati 等同协议重建（原 X09） | 外部比较 |
| W\&SPP | Mancini 等同协议重建（原 X14） | 外部比较 |
| PI | 完美信息上界 | 不可部署参考 |

主要结论：Hist-P-SO 相对 LGBM-DO 日收益 +330.6 EUR（95% 区间 [199.6, 479.2]），
日级未交付占比 -1.462 个百分点（区间 [-2.627, -0.450]）；GCJS-SO 相对 LGBM-DO
日收益 +430.3 EUR（区间 [277.2, 588.8]），未交付占比变化不显著；PGC-SO 收益
最高但相对其他场景方法的优势未达统计显著（p=0.262、p=0.161）。**支持的结论是
概率场景相对点预测有决策价值，不是某一种 copula 普遍最优。**

固定 CVaR 权重（χ=0.5）没有稳定改善实际下尾利润或交付；名义 5% VaR 的实际
超越率在不同方法间从约 6.2% 到 31.9% 不等，全部配置中的最高值为 44.4%。**模型内
风险设定不是样本外交付的校准保证。**

详见正文 `\subsection{Decision value and external-method comparison}`、
`\subsection{Risk-weight sensitivity}`（表 `tab:decision-main-results`、
`tab:risk-sensitivity`），源数据 `arm_summary.csv`、`paired_differences.csv`。

### 5.3 新能源代理量的规模与构成

330 个完整代理日、7,920 个有效小时中，2,097 小时（26.48%）代理量为正；午间窗口
占比 28.54%，7 月最高（43.55%）。全样本代理能量 1,580,970 MWh，其中午间窗口
427,646 MWh。**这 2,097 个正值小时全部是净出口小时**，对其做核算分解（净出口
2,811,562 MWh − 非风光发电 1,230,592 MWh = 代理量 1,580,970 MWh，误差 1.6×10⁻⁵
MWh）表明：**代理量是扣除非风光发电后的净出口余量，不是已证实的未利用电量**；
若把电转热（P2H，679,051 MWh，占反事实分母 30.05%）也算作反事实抵扣，代理量会
升至 2,260,021 MWh，但这属于会计层面的反事实，不能重复扣减。

详见正文 `\subsection{Data coverage and proxy occurrence}`、
`\subsection{What explains the 2,097 proxy-positive hours?}`（表
`tab:proxy-occurrence`、`tab:proxy-decomposition`）。

### 5.4 午间—晚峰联动与 GRASCO 灵敏度（206 天匹配样本）

市场专用方法的午间充电量在 0.51--6.21 MWh/day 之间，晚峰五小时窗口的物理上限是
4 MW×5 h=20 MWh/day，观测承诺已占用该功率-时长包络的
88.4%--98.1%；晚峰要求激活量只占已承诺备用能量的 8.1%--8.3%。**因此额外的午间能量
无法实质性提高备用容量，其主要物理作用是抬高交割前 SOC、保护交付。**

GRASCO 的 ρ 敏感性（GRASCO-0/25/50/100，同 206 天、同市场场景、同电池约束、
χ=0，仅 ρ 不同）显示：午间充电、预测/实际重叠度、17:00 SOC 随 ρ
**单调上升**（GRASCO-100 午间充电达 5.232 MWh/day，是 GRASCO-0 的 2.7 倍）；但
晚峰备用与交付率**不随 ρ 显著变化**（因为备用已在 07:30 冻结且接近功率上限）。
实现利润呈**非单调**响应：GRASCO-25 显著增收（+13.4 EUR/day，区间 [2.7, 28.2]），
GRASCO-50 不显著（-18.0，区间 [-42.9, 10.0]），GRASCO-100 显著减收
（-71.5，区间 [-115.6, -24.3]）。

**关键诊断数字**：206 天匹配样本的午间代理总量为 318,957 MWh；市场专用方法中
最大的午间-代理重叠是 LGBM-DO 的 449.5 MWh，对齐比例仅 0.1409%；GRASCO-100 把
这一比例提高到 0.1906%（608.0/318,957），**仍不足午间代理量的 0.2%**。也就是说，
GRASCO 确实按预期方向改变了充电时序，但幅度远小于代理量本身。

详见正文 `\subsection{Midday--evening outcomes by method}`、
`\subsection{GRASCO renewable-alignment sensitivity}`、
`\subsection{Diagnosis of the observed problem}`（表 `tab:strategy-linkage`、
`tab:proxy-aware-results`、`tab:problem-diagnosis`），源数据
`midday_evening_summary.csv`、`midday_evening_paired.csv`。

## 6. 论文仍需的证据类型

| 论文问题 | 状态 | 所需证据 |
|---|---|---|
| 午间是否充电、充电对齐程度 | **已回答**（5.4） | — |
| 电量是否保留到晚峰（17:00 SOC） | **已回答**（5.4） | — |
| 晚峰是否可交付（承诺/激活/交付/未交付） | **已回答**（5.2、5.4） | — |
| 概率场景相对点预测的决策价值 | **已回答**（5.2） | — |
| 午间充电贡献多少（严格反事实） | **部分回答**：GRASCO 的 ρ 敏感性是"软激励"而非"禁止/强制午间充电"的硬反事实 | 禁止午间充电或等价反事实的成对差 |
| 联合价值来自哪里（能量/备用/规则分解） | **未回答** | 仅能量、仅备用、固定规则和联合决策的同协议分解 |
| 风险方法是否有效 | **部分回答**：已证明固定 CVaR 不校准实际交付（5.2） | 基于延迟实损的自适应风险处理与固定风险前沿的对比 |
| 经济性是否稳健（退化、效率、E/P 比） | **未回答** | 退化成本、效率、备用持续时间和 E/P 比敏感性 |
| 结论是否可确认（样本外） | **未回答** | 未参与模型选择的未来 DK1 样本 |
| 真实系统级弃电反事实 | **未回答**（正文已给出公式 `eq:counterfactual-curtailment-reduction`，但当前价格接受者仿真不含系统平衡约束，无法计算） | 系统级出清模型 + 有/无电池两种反事实 |
| 电量来源追踪（充电是否来自 VRE） | **未回答**（正文已给出公式 `eq:source-tracing`，当前不评价） | 电量来源追踪或耦合约束 |

## 7. 允许与禁止的论文主张

以下与正文 `\section{Aggregation rules and admissible claims}` 的
`tab:claim-boundaries` 保持一致：

| 状态 | 主张 |
|---|---|
| 已允许 | 在当前同协议仿真中，历史/GCJS/PGC 概率场景相对点预测显著增收并降低未交付 |
| 已允许 | GRASCO 的正 ρ 使午间充电、预测/实际重叠度、17:00 SOC 随 ρ 单调上升 |
| 已允许 | 复杂依赖结构（高秩/表面耦合）和固定风险参数没有形成稳定、全面优势 |
| 已允许 | DK1 VRE-above-load 代理量是可复现的诊断量，且已通过净出口/非风光发电/P2H 的会计分解核实其构成 |
| 已允许 | GRASCO-25 相对 GRASCO-0 显著增收；GRASCO-50 不显著；GRASCO-100 显著减收——这是灵敏度结果，不是"ρ=25 是社会最优价值"的主张 |
| 待证据 | 联合决策增加新能源富余时段充电并提高晚峰可交付能力（当前证据显示晚峰不随 ρ 显著变化） |
| 待证据 | 午间充电对晚峰 SOC、容量和交付具有可量化的因果贡献（当前只有描述性相关，非因果） |
| 待证据 | 基于延迟实损的自适应风险调整优于固定风险前沿 |
| 禁止 | 储能降低了实际弃风弃光或晚峰失负荷 |
| 禁止 | 模拟动作等同真实电站运行 |
| 禁止 | 把代理量对齐等同于"每充的一度电都来自 VRE 或等价减少了弃电" |
| 禁止 | 把净出口或 P2H 等同于观测到的弃电 |
| 禁止 | 首次研究日前＋aFRR、顺序市场、概率场景或在线储能风险控制 |

## 8. 论文结构（与正文一致）

正文当前分为 16 个大节（不含参考文献），依次为：

1. Research question and contribution
2. Scope and terminology
3. Indices, information sets, and dependency graph
4. Data-to-settlement research chain
5. System-level renewable energy utilization
6. DK1 observations and model-generated variables
7. Ex-post proxy metrics observable in DK1（含子节：Exchange, conventional-generation, and Power-to-Heat decomposition）
8. Market forecasting and scenario construction
9. Gate-available weather, load, and proxy forecasts: WLPF
10. Proposed GRASCO decision model
11. Fifteen-minute settlement trace and linkage diagnostics
12. Experimental design and method provenance
13. Results（子节：market point forecasts；marginal/joint probabilistic forecasts；decision value and external-method comparison；risk-weight sensitivity；data coverage and proxy occurrence；what explains the 2,097 proxy-positive hours；gate-available proxy forecast results；midday–evening outcomes by method；GRASCO renewable-alignment sensitivity；diagnosis of the observed problem）
14. Research claim and remaining uncertainty
15. Aggregation rules and admissible claims
16. Relation to established formulations

**与原规划（第 8 节旧版）的差异**：原计划中的"稳健性与限制"独立章节**尚未写出**——
退化、单年度、价格接受者假设等限制目前只在个别段落里提及，没有集中成节。是否需要
补一节，由作者决定。

## 9. 图表内容

**当前正文只有 20 张表格，0 张图**（已用 `grep -c '\begin{figure}'` 核实为 0）。
原规划设想的下列图目前都还没有做：

| 计划中的图 | 当前状态 |
|---|---|
| 24 小时市场画像（日前价、上调容量价、风光/负荷、净负荷） | 未做；对应数字目前也不在正文里（见第 0 节的反向缺口） |
| 市场时间线（07:30/09:20/12:00/D 日） | 未做；正文用文字和依赖关系公式（`eq:research-chain` 等）表达 |
| 储能逐时轨迹（充电/放电/SOC/承诺/激活） | 未做；正文用表格汇总量代替 |
| 午间—晚峰传递（充电、17:00 SOC、承诺、交付、未交付） | 未做；对应数字已在 `tab:strategy-linkage`、`tab:proxy-aware-results` 表格中 |
| 收益—风险前沿 | 未做；`frontier.csv`、`risk_calibration.csv` 已有数据 |
| 方法对比表 | 已做，即 `tab:method-provenance` |

是否需要把表格数据转成图，由作者根据投稿目标（期刊/会议排版偏好）决定；本文件只
记录"数据已具备、尚未可视化"这一状态。
