# 任务 S3-B：已发表方法在 DK1 上的同协议对比（子代理分工版）

目标：在同一 07:30/12:00 信息集、同一可行域与同一结算层上，重建已发表的电能量—备用储能决策方法，与自制对照和本文未校准的 FB2＋B3 一起按实际激活结算 211 个交割日，报告收益—未交付率—尾部损失。
改哪里：新建 `configs/s3_comparison.yaml`、`src/s3_comparison/`、`experiments/s3_comparison.py`、`p1_paper/results/s3_comparison/`；按第 8 节更新锚定表与三份研究文档的指定位置。
不要做：第 9 节全部条目。
验收：第 10 节命令全部通过，`redline_reviewer` 交回表中无「阻塞」「需修正」。

**前提**：[任务 S3-A](TASK_S3_FORECAST_SIDE.md) 已合并，`main` 上存在 `p1_paper/results/s3_forecast_side/` 的 `summary.json`、`point_0730/`、`scenarios_0730/`、`scenarios_1200/`。不存在即停。

基线：`main` 最新提交。环境固定 `/public/ZLCODE/.venvs/r1_lear_de/bin/python`，不升级、不新装依赖。scipy 1.7.3 **没有 `milp`**，只能用 `linprog(method="highs")` 解线性规划；论文中的二进制变量一律按本仓库冻结的毛功率约束处理并记为偏差。

**分支与 PR**：从 `main` 开 `codex/s3-comparison`，按第 8 节分次提交，开 PR 后停下等用户确认。本文件即已确认的改动计划，执行中不因 AGENTS.md 第 6 节再次停下；停点只有开出 PR 之后或第 11 节停止条件。

---

## 0. 已确定的定位（不要重新论证）

P1 主张见 [论文规划](research-foundation-2026/PAPER_PLAN.md) 第 6 节。本任务回答两件事：

1. **同协议外部基准**：已发表方法在后改制 DK1 上、按实际激活与不平衡价结算时各自表现如何。原论文多为单日（X07）或模型内期望收益（X14）评估，本任务给出样本外逐日结算。
2. **判定实验**：只用历史日作场景的三阶段随机优化，相对点预测联合优化（B2）有没有决策价值；打乱跨目标配对后价值还剩多少。它决定「闸门条件联合场景」主张是否值得继续。

本任务**不实现**交付风险在线校准（B4），那是 S3-C。B3 在本任务中固定风险系数。

## 1. 名称与代号

| 代号 | 名称 | 本任务中的定义 |
|---|---|---|
| B0、B1、Oracle | 只做能量、顺序规则、条件完美预知 | 读 `p1_paper/results/s2_dk1_baselines/`，不重跑 |
| B2 | 点预测＋联合优化 | 已有 `B2_F01`、`B2_F08` 只读；新增 `B2_FB0`、`B2_FB0_gate` |
| B3 | 三阶段随机联合优化，固定风险系数 | 第 3.3 节；阶段结构引用 X14，不是本文贡献 |
| B4 | B3＋按已实现未交付事件在线校准风险系数 | 不在本任务 |

## 2. 子代理编排

### 2.1 角色与写入范围

| 子代理（实例） | 权限 | 负责 | 允许写入 |
|---|---|---|---|
| `library_paper_reviewer` | 只读 | 抽取 X14（附录 A 全部式）、X07（第 II-B 节）、X06（第 II–III 节机会约束与分位近似）、X13（第 II 节联合机会约束、式 (3a)–(3e)、可靠性与样本数）、X09（决策模型全文）的**决策部分**规格，字段见第 4.2 节 | 无 |
| `structure_scout` | 只读 | 方向 b：通读 `src/epf_harness/dk1_storage.py`、`src/f01_lear_dk1/settlement.py`、`src/s2c_decision_layer/arms.py`，给出第 3.3 节三阶段规划与现有可行域、恢复层、结算层之间的逐约束映射，指出重复占用风险，并给出两个可解析求解的手算案例 | 无 |
| `literature_searcher` | 只读 | 检索问题：X06、X07、X09、X13、X14 是否有官方代码、数据、勘误或后续同协议重建；有则给链接与许可证 | 无 |
| `dk1_data_profiler`（实例 `scenario_io`） | 可写 | 读取并校验 S3-A 产物；第 3.4 节的共同场景处理 | `src/s3_comparison/scenario_io.py`；`p1_paper/results/s3_comparison/scenario_io_audit.csv` |
| `dk1_data_profiler`（实例 `hand_checks`） | 可写 | 第 7 节第 1–3 项手算案例的独立实现（不 import 主线程的求解模块，只 import `src/epf_harness/`） | `src/s3_comparison/hand_checks.py` |
| `redline_reviewer` | 只读 | 开 PR 前审查全部改动 | 无 |
| **主线程** | 可写 | yaml、`three_stage.py`、`published_rules.py`、实验脚本、结果、锚定表、研究文档、提交与 PR | 除上面两个实例专属文件外本任务涉及的全部文件 |

X07、X14 全文 PDF 的本地位置与禁止提交规则同 [S3-A](TASK_S3_FORECAST_SIDE.md) 第 2.2 节。

### 2.2 执行顺序

| 波次 | 谁 | 做什么 | 前提 |
|---|---|---|---|
| 0 | 主线程 | 核对 S3-A 产物存在；开分支；写 `configs/s3_comparison.yaml` 的 storage、solver、tree、risk、arms 键与 `src/s3_comparison/__init__.py` | 无 |
| 1（并行） | `library_paper_reviewer`、`structure_scout`、`literature_searcher`、`dk1_data_profiler` 的 `scenario_io` 与 `hand_checks` 两个实例 | 按第 2.1 节 | 波次 0 |
| 主线程 A | 主线程 | 把第 4.2 节规格与偏差写入 yaml `published_rules`；实现 `three_stage.py`、`published_rules.py`、实验脚本；跑 `--check`；全量运行两次 | 波次 1 全部交回 |
| 主线程 B | 主线程 | 锚定表与研究文档（第 8 节） | 主线程 A |
| 2 | `redline_reviewer` | 审查全部改动 | 主线程 B |
| 主线程 C | 主线程 | 修正后开 PR，停下 | 审查交回 |

整合规则同 S3-A 第 2.4 节。

## 3. 决策协议

### 3.1 共同可行域与结算（冻结，不改）

储能参数、效率、交付时长 T=1 h、初始与终端 SOC 4.4 MWh、毛功率约束、备用能量预留约束一律取自 `configs/s2c_decision_layer.yaml` 并调用 `src/epf_harness/dk1_storage.py` 的现有函数构造；结算一律调用 `settle_day`，不改其任何逻辑。测试日为 S2 的 211 天。规划中的采购量沿用 S2 口径（面板值），在结果中声明。

### 3.2 两次求解的时序

| 时刻 | 已知 | 求解 | 冻结 |
|---|---|---|---|
| 07:30 (D-1) | 07:30 场景 | 第 3.3 节三阶段规划 | 备用承诺 r_up（24 维） |
| 12:00 (D-1) | r_up；D 日已公布容量价；12:00 场景 | 两阶段规划：能量计划为第一阶段，每个场景的交付与恢复为追索；能量计划须满足 `solve_day` 同样的无激活日末 SOC 等式 | 能量计划 p_ch、p_dis（96 维） |
| 交割日 | 实际激活、激活价、不平衡价 | `settle_day` | — |

12:00 场景：`fb2_gate`、`fb2_no_temporal`、`fb2_plus_weather` 用 S3-A 的精确条件化产物；`empirical_copula` 用类比条件化产物；其余臂无条件化能力，沿用 07:30 场景，这一差别即「容量价条件化」的检验对象。

### 3.3 三阶段随机联合优化（B3）

- 场景树：把 50 个 07:30 场景按容量价路径（按 F(D) 标准差标准化的 24 维）用 `sklearn.cluster.KMeans` 聚成 K 簇（yaml `tree.clusters: 10`，种子写 yaml），簇概率为场景占比。
- 第一阶段：r_up(h)，全部场景共用。
- 第二阶段：每簇一套能量计划 p_ch,k、p_dis,k；同簇场景共用（非预期性）。
- 追索：每个场景 s、每个 15 分钟槽 t 的交付量 delivered ∈ [0, required]（required = r_up(h)·激活份额_s,t）与恢复充电 recovery ≥ 0；约束与 `recovery_constraints` 同构（SOC 上下界、剩余毛功率、备用能量预留、终端 SOC 回到初值）。
- 目标（最大化）：Σ_s π_s [容量价_s·r_up ＋ 日前价_s·(p_dis,k − p_ch,k)·Δt ＋ 激活价_s·delivered − 不平衡价_s·(recovery·Δt ＋ required − delivered)]，可选 (1−χ)·期望 ＋ χ·CVaR_α（Rockafellar–Uryasev 线性化）。
- 求解：`linprog(method="highs")`，时限写 yaml。超时或不可行时当日改用 B1 的冻结规则决策，记入 `fallback_days`，并计入结果。

### 3.4 场景的共同处理（`scenario_io` 实例实现，所有臂相同）

| 项 | 规则 |
|---|---|
| 场景来源 | 读 `scenarios_0730/<arm>.npz` 与 `scenarios_1200/<arm>.npz`，校验交割日与 211 天逐日一致、形状、`information_set` |
| 小时 → 15 分钟 | 日前价、激活价在小时内四槽相同；激活量四槽均分；激活份额 = 槽激活量 / 面板采购量 |
| 不平衡价 | 不平衡价_s,t = 日前价_s,h ＋ 溢价；溢价从 F(D)（定义同 S3-A 第 3.3 节）同一四小时时段、同一激活状态（该小时激活量是否为正）的实际 15 分钟溢价中按种子抽取 |
| 期望型激活 | `activation_representation: expected_value` 的场景（X14 式）在追索中用确定性 required，不生成路径 |
| 场景数 | 50；不在测试期上选择 |

## 4. 对比臂

### 4.1 总表

| 臂 id | 类别 | 07:30 场景 | 07:30 规划 | 12:00 规划 | 风险处理 |
|---|---|---|---|---|---|
| `B2_FB0` | 自制对照 | `point_0730/fb0_lgbm.csv` | `solve_day` 自由备用（同 S2-C） | 不重排 | 无 |
| `B2_FB0_gate` | 自制对照 | 同上 | 同上 | `solve_day` 固定 r_up，日前价用 `fb0_lgbm_cap1200` | 无 |
| `B3_hist_paired` | 判定实验 | `hist_paired` | B3 | 无条件，沿用 07:30 场景 | χ=0 |
| `B3_hist_independent` | 判定实验 | `hist_independent` | B3 | 同上 | χ=0 |
| `B3_fb1` | 消融 | `fb1_qr` | B3 | 同上 | χ=0 |
| `B3_empirical_copula` | 自制对照 | `empirical_copula` | B3 | 类比条件化 | χ=0 |
| `B3_fb2_no_cross_target` | 消融 | `fb2_no_cross_target` | B3 | 无条件 | χ=0 |
| `B3_fb2` | 本文未校准 | `fb2_gate` | B3 | 精确条件化 | χ=0 |
| `B3_fb2_plus_weather` | 上界 | `fb2_plus_weather` | B3 | 精确条件化 | χ=0，标注不可部署 |
| `X14_rebuild` | 已发表重建 | `x14_hist_independent` | B3 结构，激活用期望型 | 按 X14 同簇非预期性 | CVaR：χ ∈ {0, 0.1, 0.5}，主行 χ=0；α 取原文值 |
| `X07_rebuild` | 已发表重建 | `x07_static_copula` | 单次决策：K=1，r_up 与能量计划在 07:30 一起定，12:00 不重排 | 不重排 | CVaR β=0.5；每个场景须全额交付（原文式 (36)–(37)），不可行时按第 3.3 节后备 |
| `X09_rebuild` | 已发表重建 | `x09_block_copula` | 按原文顺序两阶段随机模型改到 07:30 | 按原文 | 按原文 |
| `X06_rebuild` | 已发表重建 | 日前价、容量价用 `fb0_lgbm`；激活用 `climatology` 的逐时经验分位 | 确定性 LP＋逐时机会约束，ε=0.2 按原文在各分位层均分 | 不重排 | 事先设定的 ε |
| `X13_rebuild` | 已发表重建 | 价格用 `fb0_lgbm`（原文为代表性价格曲线，记偏差）；激活比例与持续时间的联合经验分布由 F(D) 的 15 分钟激活事件估计 | 确定性 LP＋联合机会约束化为逐约束机会约束：可靠性 98%，β=99%，样本数按原文式 (3d)，逐约束水平按式 (3b)–(3c) 蒙特卡洛收紧 | 不重排 | 事先设定的可靠性 |

所有 B3 类臂的 χ、K、场景数、时限都在 yaml 预先写定；不得用测试期结果选择。

### 4.2 已发表方法决策部分的抽取字段

`library_paper_reviewer` 按下列字段交回，主线程写入 yaml `published_rules.<id>`（`source`、`paper_value`、`dk1_value`、`deviation`、`undetermined`）：决策时刻与阶段；决策变量；目标函数各项；SOC 与备用持续约束；激活表示方式；非预期性结构；风险度量及其参数原值；场景数；二进制变量及其作用；需删除的资产（风电、光伏、常规机组、需求响应）；市场产品差异（德国 4 小时块、PICASSO、FCR-N、比利时产品）；原文评估方式。

某字段「未确定」且决定决策形状时：论文明示假设则实现最接近变体并记偏差，否则该臂不跑，写入 `summary.json` 的 `skipped_arms`。

### 4.3 第二批与不做

| 论文 | 处理 | 理由 |
|---|---|---|
| X10 风电—电池自调度 | 第二批：第一批完成后再做 | 去掉风电后退化为点价格＋激活点策略的两阶段随机优化，与 `B2_FB0_gate` 同类 |
| X15 北欧多市场 BESS | 第二批 | GAM 点预测＋另建场景，与 `B3_fb1` 同类 |
| X22 丹麦风电—退役电池 | 第二批 | Monte Carlo 独立场景，与 `X14_rebuild` 同类 |
| X02 西班牙一体化框架 | 第二批 | 多产品点预测＋确定性优化，与 `B2_FB0` 同类 |
| X11 BESS-PV 调频规划 | 第二批 | 时步独立概率预测，与 `B3_fb2_no_temporal` 思路同类 |
| L03、X19 | P3 | 决策导向训练，不属 P1 |
| M08、M07、X12、X17 等其余候选 | 不做 | 见论文规划第 6 节候选表理由 |

第二批只在第一批全部通过验收后才开始，且另起提交。

### 4.4 只读参照行

B0、B1、Oracle 读 `p1_paper/results/s2_dk1_baselines/summary.json`；`B2_F01`、`B2_F08`、`oracle_price_reserve` 读 `p1_paper/results/s2c_decision_layer/summary.json`。结果表中注明 `B2_F01`、`B2_F08` 的预测窗口为 116 天，与本任务新臂不同；干净的点预测对照是 `B2_FB0`。

## 5. 评价

| 维度 | 指标 |
|---|---|
| 收益 | 总收益及能量、容量、激活、恢复、未履约五项；日均 |
| 交付 | 未交付激活占比、有未交付的天数、功率缺口与能量缺口 MWh、压力小时（S1 冻结阈值）未交付占比 |
| 尾部 | 日收益最差 5% 的平均值（CVaR5%） |
| 物理 | 终端 SOC 偏差最大值、槽内双向位置计数（沿用 `activation_ledger` 的列） |
| 可实施性 | 求解时间中位数与最大值、后备触发天数 |
| 成对比较 | 每个臂对 `B2_FB0`、对 `X14_rebuild` 的逐日收益差与未交付差：7 天块自助法 95% 置信区间（2000 次，种子 1）与 DM 检验（Newey–West 6 阶） |
| 前沿 | 各臂（含 X14 的三个 χ）在「总收益—未交付占比」平面上的点，及是否被其他臂支配 |
| 分月 | 2026-02 至 2026-08 逐月总收益与未交付占比 |

判定实验结论按预先写定的规则给出：`B3_hist_paired` 相对 `B2_FB0` 的日收益差置信区间下界 > 0，或未交付占比差置信区间上界 < 0 且收益差下界 > −1%·`B2_FB0` 日均，记为「分布有决策价值」；`B3_hist_paired` 与 `B3_hist_independent` 的差给出跨目标配对的价值。

## 6. 文件与接口

| 文件 | 写入者 | 内容 |
|---|---|---|
| `configs/s3_comparison.yaml` | 主线程 | 储能引用、求解器、树、风险参数、臂清单、`published_rules` 规格 |
| `src/s3_comparison/__init__.py` | 主线程 | 空 |
| `src/s3_comparison/scenario_io.py` | `scenario_io` 实例 | 第 3.4 节 |
| `src/s3_comparison/hand_checks.py` | `hand_checks` 实例 | 第 7 节第 1–3 项 |
| `src/s3_comparison/three_stage.py` | 主线程 | 07:30 三阶段与 12:00 两阶段规划，每个函数不超过 60 行 |
| `src/s3_comparison/published_rules.py` | 主线程 | X06、X07、X09、X13、X14 的差异化部分 |
| `experiments/s3_comparison.py` | 主线程 | `--config`、`--check`；按日多进程 |
| `p1_paper/results/s3_comparison/` | 主线程 | `summary.json`（环境指纹、种子、臂清单、偏差、跳过的臂、后备天数、判定实验结论、验收标志、文件 SHA-256，不写墙钟时间）、`arm_summary.csv`、`daily_results.csv`（列与 `s2c_decision_layer/daily_results.csv` 相同并加 `strategy`）、`paired_differences.csv`、`frontier.csv`、`monthly.csv`、`stress_hours.csv`、`scenario_io_audit.csv` |

## 7. 正确性检查（`--check`，全量运行前必须通过）

1. **退化一致**：1 个场景、K=1、激活为零、价格取实现值时，07:30 三阶段规划的目标值与 `solve_day`（自由备用）一致，误差不超过 yaml 容差。
2. **两场景手算**：有激活与无激活两场景、各 50%，解析最优与 LP 一致。
3. **非预期性**：同簇场景的能量计划逐槽相等；12:00 冻结的 r_up 与 07:30 输出逐小时相等。
4. **完美信息一致**：以当日实现值作为唯一场景时，B3 对 211 天的结算总收益与 S2 Oracle 的 824,744 EUR 之差写入 `summary.json`；相对差超过 0.1% 时停下，逐项说明来源再继续。
5. **结算未改**：`B2_FB0` 以 F08 的 `forecasts.csv` 代入时，211 天总收益等于 S2-C 的 `B2_F08` 560,782 EUR（误差不超过 1e-6 EUR）。
6. **物理**：全部臂日末 SOC 偏差不超过 yaml 容差；毛功率不超过 4 MW。
7. **可复现**：全量运行两次，所有 csv 的 SHA-256 一致。

## 8. 执行步骤与提交

1. 波次 0 与波次 1 按第 2.2 节。
2. 第一次提交：`scenario_io.py`、`hand_checks.py`、yaml 骨架与审计表，写明实例名。
3. 第二次提交：规划与规则模块、实验脚本与全部结果；提交信息写明判定实验结论、各已发表重建的收益与未交付占比、`B3_fb2` 相对 `X14_rebuild` 与 `B2_FB0` 的成对差。
4. 第三次提交：
   - 锚定表：`experiments/anchor_table.py` 与 `configs/anchor_table.yaml` 中的占位行 `FB2_B3` 改为 `FB2_B4`；为 X06、X07、X09、X13、X14 各加一行，状态为「原论文数据不公开，不可做原任务锚定」；重跑锚定表，`git diff` 中只允许这些行与 `summary.json` 对应条目变化。
   - [执行计划](research-foundation-2026/EXECUTION_PLAN.md) 第 10.1 节 S3 行与第 10.4 节，合计不超过 15 行；
   - [论文规划](research-foundation-2026/PAPER_PLAN.md) 第 2 节「戊」行状态；判定实验若为「无决策价值」，同时在第 5 节停止条件下写一行触发记录；
   - [研究框架](research-foundation-2026/RESEARCH_FRAMEWORK.md) 第 8 节末加不超过 8 行的「S3-B 结果」。
5. 审查后的修正单独提交。第二批（第 4.3 节）另起提交。

## 9. 不要做

1. 不改冻结协议与 `src/epf_harness/`、`src/f01_lear_dk1/settlement.py` 的任何逻辑；不改已有实验与结果（锚定表按第 8 节例外）。
2. 不用测试期结果选 χ、K、场景数、ε、可靠性或任何参数。
3. 不让任何臂获得他臂没有的信息；`B3_fb2_plus_weather` 除外且显式标注。
4. 不实现 B4 或任何在线更新规则。
5. 不把重建称为复现；原论文数字只能标 `paper-reported`，不得与本任务结果放在同一列比较。
6. 不引入整数规划或新求解器依赖。
7. 不提交付费全文 PDF；不新建 AGENTS.md 禁止的文件；不用 `git add -A` 或 `git add .`；不直接推 `main`。
8. 研究文档任一文件超过 400 行时停下报告。

## 10. 验收

```bash
cd /public/ZLCODE/electricity-price-forecasting-research
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/s3_comparison.py --config configs/s3_comparison.yaml --check
/public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/s3_comparison.py --config configs/s3_comparison.yaml
find p1_paper/results/s3_comparison -name '*.csv' -exec sha256sum {} + | sort > /tmp/s3b_run1.sha
/public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/s3_comparison.py --config configs/s3_comparison.yaml
find p1_paper/results/s3_comparison -name '*.csv' -exec sha256sum {} + | sort | diff - /tmp/s3b_run1.sha
/public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/anchor_table.py --config configs/anchor_table.yaml
/public/ZLCODE/.venvs/r1_lear_de/bin/python scripts/audit_multi_market_resources.py
wc -l research-foundation-2026/*.md
git diff --stat main -- p1_paper/results/ src/epf_harness/ src/f01_lear_dk1/ src/s2c_decision_layer/
```

通过条件：`--check` 七项通过；两次运行哈希一致；资源审计退出码 0；研究文档均不超过 400 行；`p1_paper/results/` 只出现 `s3_comparison/` 与 `anchor_table/` 的变化；`src/epf_harness/`、`src/f01_lear_dk1/`、`src/s2c_decision_layer/` 无改动。

## 11. 停止条件与 PR

立即停下并报告：S3-A 产物缺失或校验失败；第 7 节第 4、5 项不通过；需要改动第 9 节第 1 条所列冻结项；后备触发超过 yaml `max_fallback_share`（写定 5%）。

判定实验为「无决策价值」时照常完成全部臂并如实报告，不调参挽救。

PR（`codex/s3-comparison` → `main`）描述用三到五句：判定实验结论；五个已发表重建的收益与未交付占比排名；`B3_fb2` 相对 `X14_rebuild` 与 `B2_FB0` 的成对差及置信区间；被跳过的臂及原因。**开 PR 后停下，等用户确认后再进入 S3-C。**
