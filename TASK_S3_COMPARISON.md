# 任务 S3-B：已发表方法在 DK1 上的同协议对比（子代理分工版）

目标：在同一 07:30/12:00 信息集、同一可行域与同一结算层上，重建已发表的电能量—备用储能决策方法，与自制对照和本文未校准的 FB2＋B3 一起按实际激活结算 211 个交割日，报告收益—未交付率—尾部损失，以及预测风险与实际结算是否一致。
改哪里：新建 `configs/s3_comparison.yaml`、`src/s3_comparison/`、`experiments/s3_comparison.py`、`p1_paper/results/s3_comparison/`；修改 `experiments/anchor_table.py` 与 `configs/anchor_table.yaml`；按第 8 节更新三份研究文档的指定位置。
不要做：第 9 节全部条目。
验收：第 10 节命令全部通过，`redline_reviewer` 交回表中无「阻塞」「需修正」。

**前提**（不存在即停）：
- `p1_paper/results/s3_forecast_side/`：`summary.json`、`point_0730/fb0_lgbm.csv`、`point_metrics.csv`、`scenarios_0730/` 下的 `hist_paired`、`hist_independent`、`climatology`、`x09_block_copula`、`fb2_gate`、`fb2_plus_weather`，`scenarios_1200/` 下的 `fb2_gate`、`fb2_plus_weather`；
- `p1_paper/results/s3_cross_market/`：`summary.json`、`daily_joint_scores.csv`、`daily_conditional_scores.csv`；
- `src/s3_forecast_side/`、`src/s3_cross_market/` 与 `configs/s3_cross_market.yaml`。

基线：`main` 最新提交。环境：Python 3.9.25 虚拟环境，本机在 `/home/zl/nvme/.venvs/r1_lear_de`，按 `requirements-r1.txt` 的固定版本重建；不升级、不新装依赖。下文命令一律用 `$PY` 指代它的解释器，定义在第 10 节开头，换机器只改那一行。scipy 1.7.3 **没有 `milp`**，只能用 `linprog(method="highs")` 解线性规划；论文中的二进制变量一律按本仓库冻结的毛功率约束处理并记为偏差。

**分支与 PR**：从 `main` 开 `codex/s3-comparison`，按第 8 节分次提交，开 PR 后停下等用户确认。本文件即已确认的改动计划，执行中不因 AGENTS.md 第 6 节再次停下；停点只有开出 PR 之后或第 11 节停止条件。

修订记录：2026-09-14 按执行计划第 10.5 节与用户决定重写——FB2 输入改为 `own_pooled` 边际两种耦合并由本任务导出；12:00 比较三种产品；X14 按明示假设重建不跳过；B3 加一档风险厌恶；评价加预测风险超越率与交叉评分；RQ4 只作复验。2026-09-15 修正 X07、X14 的证据边界：本机已有用户通过机构订阅取得的未跟踪全文，方法规格以全文核读为准，来源卡只承担书目、获取与入库边界记录。2026-09-15 按用户决定修订 B3 追索：主线程 A 的完美信息检查中，B3 结算 813,010.785 EUR，Oracle 824,743.787 EUR，相对差 1.42%，原因是 B3 允许经济性短缺，而 Oracle 与 `settle_day` 都优先交付。现改为 07:30 每个场景全额交付、12:00 先求最大交付再优化目标；原定义保留为敏感性臂 `B3_hist_paired_shortfall`。第 7 节第 4 项阈值不变。

---

## 0. 已确定的定位（不要重新论证）

P1 主张见 [论文规划](research-foundation-2026/PAPER_PLAN.md) 第 6 节，预测侧冻结依据见 [执行计划](research-foundation-2026/EXECUTION_PLAN.md) 第 10.5 节。本任务回答三件事：

1. **同协议外部基准**：已发表方法在后改制 DK1 上、按实际激活与不平衡价结算时各自表现如何。原论文多为单日（X07）或模型内期望收益（X14）评估，本任务给出样本外逐日结算。
2. **判定实验**：只用历史日作场景的三阶段随机优化，相对点预测联合优化（B2）有没有决策价值；打乱跨目标配对后价值还剩多少；12:00 按已公布容量价更新有没有决策价值。它决定「闸门条件联合场景」主张是否保留。
3. **RQ4 复验（次要）**：预测层联合评分的排序与决策收益排序是否一致。该结论已有 F07、X20，本任务只报告，不作主张。

本任务**不实现**交付风险在线校准（B4），那是 S3-C。B3 在本任务中使用预先写定的固定风险系数。

## 1. 名称与代号

| 代号 | 名称 | 本任务中的定义 |
|---|---|---|
| B0、B1、Oracle | 只做能量、顺序规则、条件完美预知 | 读 `p1_paper/results/s2_dk1_baselines/`，不重跑 |
| B2 | 点预测＋联合优化 | 已有 `B2_F01`、`B2_F08` 只读；新增 `B2_FB0`、`B2_FB0_gate` |
| B3 | 三阶段随机联合优化，固定风险系数 | 第 3.3 节；阶段结构引用 X14，不是本文贡献；计划阶段按交付义务追索（07:30 全额交付，12:00 优先交付，见第 3.2、3.3 节）；风险系数 χ ∈ yaml `risk.chi_levels: [0.0, 0.5]` |
| FB2 | 闸门条件联合概率场景 | 本任务中实现为 `own_pooled` 边际配经验 Copula（`pooled_empirical`）或低秩高斯 Copula（`pooled_gaussian`） |
| B4 | B3＋按已实现未交付事件在线校准风险系数 | 不在本任务 |

## 2. 子代理编排

### 2.1 角色与写入范围

| 子代理（实例） | 权限 | 负责 | 允许写入 |
|---|---|---|---|
| `library_paper_reviewer` | 只读 | 核读 X14、X07 的本机付费全文并抽取决策规格；另抽取 X06（第 II–III 节机会约束与分位近似）、X13（第 II 节联合机会约束、式 (3a)–(3e)、可靠性与样本数）、X09（决策模型全文）的**决策部分**规格，字段见第 4.2 节；从 F07（`paper/.../03_preprints/F07_...pdf`）抽取式 (29)–(30) 的交叉评分与式 (40) 的 VaR/CVaR 联合评分函数 | 无 |
| `structure_scout` | 只读 | 通读 `src/epf_harness/dk1_storage.py`、`src/f01_lear_dk1/settlement.py`、`src/s2c_decision_layer/arms.py`，给出第 3.3 节三阶段规划与现有可行域、恢复层、结算层之间的逐约束映射，指出重复占用风险，并给出两个可解析求解的手算案例 | 无 |
| `literature_searcher` | 只读 | 检索问题：X06、X07、X09、X13、X14 是否有官方代码、数据、勘误或后续同协议重建；有则给链接与许可证 | 无 |
| `dk1_data_profiler`（实例 `scenario_io`） | 可写 | 读取并校验全部场景文件；第 3.4 节的共同场景处理 | `src/s3_comparison/scenario_io.py`；`p1_paper/results/s3_comparison/scenario_io_audit.csv` |
| `dk1_data_profiler`（实例 `hand_checks`） | 可写 | 第 7 节第 1–3 项与第 10 项手算案例的独立实现（不 import 主线程的求解与评分模块，只 import `src/epf_harness/`） | `src/s3_comparison/hand_checks.py` |
| `redline_reviewer` | 只读 | 开 PR 前审查全部改动 | 无 |
| **主线程** | 可写 | yaml、`scenario_export.py`、`three_stage.py`、`published_rules.py`、`risk_scoring.py`、实验脚本、结果、锚定表、研究文档、提交与 PR | 除上面两个实例专属文件外本任务涉及的全部文件 |

X07、X14 的付费全文由用户通过机构订阅取得，当前分别位于 `paper/multi_market_energy_reserve/01_intersection_frontier/02_peer_reviewed_specialized/X07_2022_IEEE_Systems_VPP_Stochastic.pdf`（SHA-256 `7aef32d4189f7f63a9c84497483def776b9274ef11ece563c2f8bf4ddeb65198`）和同目录 `X14_2024_EEM_Wind_Battery_aFRR.pdf`（SHA-256 `2ffa534e0ff11d79b4e2d6fa2993d526431df652d037d7a75360b495e19f3998`）。二者只作本机只读证据，不加入 git；来源卡只记录书目、获取方式与不入库原因，不替代全文方法证据。每个抽取字段必须附全文页码及公式、表或章节定位；全文未写明的字段才记「未确定」。任一 PDF 缺失或散列不符即停止对应臂的抽取并报告，不退回来源卡补定方法字段。X07 按第 4.2 节规则处理；**X14 为例外**：按第 4.1 节写定的明示假设实现最接近变体，逐条写入 yaml 偏差，不跳过（2026-09-14 用户决定）。

### 2.2 执行顺序

| 波次 | 谁 | 做什么 | 前提 |
|---|---|---|---|
| 0 | 主线程 | 核对前提；开分支；写 `configs/s3_comparison.yaml` 的 storage、solver、risk、tree、export、arms 键与 `src/s3_comparison/__init__.py`；实现 `scenario_export.py` 并运行 `--export`，第 7 节第 8、9 项通过后提交导出文件 | 无 |
| 1（并行） | `library_paper_reviewer`、`structure_scout`、`literature_searcher`、`dk1_data_profiler` 的 `scenario_io` 与 `hand_checks` 两个实例 | 按第 2.1 节 | 波次 0 |
| 主线程 A | 主线程 | 把第 4.2 节规格与偏差写入 yaml `published_rules`；实现 `three_stage.py`、`published_rules.py`、`risk_scoring.py`、实验脚本；跑 `--check`；全量运行两次 | 波次 1 全部交回 |
| 主线程 B | 主线程 | 锚定表与研究文档（第 8 节） | 主线程 A |
| 2 | `redline_reviewer` | 审查全部改动 | 主线程 B |
| 主线程 C | 主线程 | 修正后开 PR，停下 | 审查交回 |

整合规则：只采用子代理交回的、带出处的内容；标为「未确定」「未核验」「未检索」的原样保留，不得自行补全；两个子代理结论冲突时回到原文核对，并在提交信息里说明；子代理失败时该部分记入 `summary.json` 的 `incomplete`，不得凭记忆补编。

## 3. 决策协议

### 3.1 共同可行域与结算（冻结，不改）

储能参数、效率、交付时长 T=1 h、初始与终端 SOC 4.4 MWh、毛功率约束、备用能量预留约束一律取自 `configs/s2c_decision_layer.yaml`，并调用 `src/epf_harness/dk1_storage.py` 的现有函数构造；实际面板一律由 `src/f01_lear_dk1/settlement.py` 的 `build_panels` 构建，测试日与 B1 阈值由 `settlement_protocol` 给出（211 天、6.69 EUR/MW）；结算一律调用 `settle_day`，不改其任何逻辑。规划中的采购量沿用 S2 口径（面板值），在结果中声明。

### 3.2 两次求解的时序

| 时刻 | 已知 | 求解 | 冻结 |
|---|---|---|---|
| 07:30 (D-1) | 07:30 场景 | 第 3.3 节三阶段规划 | 备用承诺 r_up（24 维） |
| 12:00 (D-1) | r_up；D 日已公布容量价；12:00 场景 | 两阶段规划：能量计划为第一阶段，每个场景的交付与恢复为追索，交付量 ∈ [0, required]；能量计划须满足 `solve_day` 同样的无激活日末 SOC 等式。**第一步**在全部物理约束下最大化 Σ_s π_s Σ_t delivered_s,t，得 D*。**第二步**加约束 Σ_s π_s Σ_t delivered_s,t ≥ D* − yaml `recourse.delivery_tolerance_mwh`，最大化与 07:30 同一 χ 的目标，剩余短缺按第 3.3 节收益式中的不平衡价计。第一步按概率加权的总交付量取最大，不是逐场景取最大，这一近似在结果中声明 | 能量计划 p_ch、p_dis（96 维）及其无激活 `soc_plan` |
| 交割日 | 实际激活、激活价、不平衡价 | `settle_day` | — |

12:00 场景由第 3.5 节给出；无 12:00 产品的臂沿用 07:30 场景，这一差别即「容量价条件化」的检验对象。

### 3.3 三阶段随机联合优化（B3）

- 场景树：把 50 个 07:30 场景按容量价路径（按 F(D) 标准差标准化的 24 维）用 `sklearn.cluster.KMeans` 聚成 K 簇（yaml `tree.clusters: 10`，`tree.seed` 写 yaml），簇概率为场景占比。
- 第一阶段：r_up(h)，全部场景共用。
- 第二阶段：每簇一套能量计划 p_ch,k、p_dis,k；同簇场景共用（非预期性）。
- 追索：每个场景 s、每个 15 分钟槽 t 的交付量 delivered_s,t = required_s,t（required = r_up(h)·激活份额_s,t，**全额交付**）与恢复充电 recovery ≥ 0；约束与 `recovery_constraints` 同构（SOC 上下界、剩余毛功率、备用能量预留、终端 SOC 回到初值）。r_up = 0 始终可行，交付不了的小时由模型自动减少承诺。
- 场景收益：R_s = 容量价_s·r_up ＋ 日前价_s·(p_dis,k − p_ch,k)·Δt ＋ 激活价_s·delivered − 不平衡价_s·(recovery·Δt ＋ required − delivered)。07:30 规划中 required − delivered = 0；12:00 规划中按第 3.2 节取值。
- 目标（最大化）：(1−χ)·Σ_s π_s R_s ＋ χ·CVaR_β(R)，β = yaml `risk.cvar_level: 0.95`（最差 5% 场景），Rockafellar–Uryasev 线性化：CVaR = η − Σ_s π_s u_s /(1−β)，u_s ≥ η − R_s，u_s ≥ 0。χ 与 β 只在 yaml 写一次，测试期不改。
- 求解：`linprog(method="highs")`，时限写 yaml。超时或不可行时当日改用 B1 的冻结规则决策，记入 `fallback_days`，并计入结果。

### 3.4 场景的共同处理（`scenario_io` 实例实现，所有臂相同）

| 项 | 规则 |
|---|---|
| 场景来源 | 读第 3.5 节列出的 npz，校验交割日与 211 天逐日一致、形状、`information_set`、`evaluated_days` |
| 小时 → 15 分钟 | 日前价、激活价在小时内四槽相同；激活量四槽均分；激活份额 = 槽激活量 / 面板采购量 |
| 不平衡价 | 不平衡价_s,t = 日前价_s,h ＋ 溢价；溢价从 F(D)（拟合块 [D-119, D-36]）同一四小时时段、同一激活状态（该小时激活量是否为正）的实际 15 分钟溢价中按种子抽取 |
| 激活价缺失 | 场景中激活价维照常存在（表示「若被激活时的价格」）；该维为 NaN 的场景值只在 required=0 时允许，其余情形报错 |
| 期望型激活 | `activation_representation: expected_value` 的场景（X14 重建）在追索中用确定性 required，不生成路径 |
| 场景数 | 50；不在测试期上选择 |

### 3.5 场景输入与导出（`scenario_export.py`，主线程，波次 0）

`p1_paper/results/s3_cross_market/` 没有场景文件，`fb0_lgbm_cap1200` 点预测也未由 S3-A 导出，均由本任务重算后导出，不改 `s3_cross_market`、`s3_forecast_side` 的代码与结果。

| 导出 id | 来源 | 07:30 | 12:00 |
|---|---|---|---|
| `pooled_independent` | `own_pooled` 边际，独立抽样（`fb1_qr` 键） | 是 | 无（沿用 07:30） |
| `pooled_gaussian` | `own_pooled` 边际，低秩高斯 Copula（`fb2_gate` 键） | 是 | `gaussian_24_hour`（`s3_cross_market` 的高斯 24 小时条件化）；`direct_qr`（见下） |
| `pooled_gaussian_no_cross_target` | 同上，相关阵跨目标块置零（`fb2_no_cross_target` 键） | 是 | 无 |
| `pooled_empirical` | `own_pooled` 边际，经验 Copula（`empirical_copula` 键） | 是 | `analog`（类比日条件化）；`direct_qr`（见下） |
| `x14_rebuild` | 第 4.1 节 X14 场景规则，由 `published_rules.py` 生成 | 是（`activation_representation: expected_value`） | 按第 4.1 节簇选择，不另存 |
| `fb0_lgbm_cap1200` | S3-A 的 FB0 加 D 日实际容量价同小时值与日均值 | — | 点预测 csv |

规则：
- 生成调用 `src/s3_cross_market/models.py` 的 `prepare_origin`、`sample_origin` 与 `src/s3_forecast_side/generators.py` 的现有函数，种子 = `configs/s3_cross_market.yaml` 的 `sampling_seed` ＋ 测试日序号，每日先生成 100 个场景并做第 7 节第 8 项核对，再导出前 50 个。
- `direct_qr`：沿用生成 07:30 场景时的同一组均匀数，只把日前价 24 维的分位网格换成 12:00 直接分位数回归网格（`direct_conditional_grid`），激活量与激活价两维取值不变。
- 文件：`p1_paper/results/s3_comparison/scenarios_0730/<id>.npz`（`values` float32，(211, 50, 4, 24)）、`scenarios_1200/<id>__<product>.npz`（(211, 50, 3, 24)，目标为日前价、激活量、激活价）、`point_1200/fb0_lgbm_cap1200.csv`（列同 S3-A 点预测 csv）；字段同 S3-A（`delivery_days`、`targets`、`seed`、`information_set`、`activation_representation`、`evaluated_days`），另存 `source_config_sha256`。每个文件不超过 20 MB。
- 沿用 S3-A 文件：`hist_paired`、`hist_independent`、`climatology`、`x09_block_copula`（边际为 S3-A 按小时 `fb1_qr`，在结果中声明）；衔接行 `fb2_gate` 与上界 `fb2_plus_weather`（S3-A 按小时边际，只用于第 4.1 节两行）。

## 4. 对比臂

### 4.1 总表

B3 类臂（带 `B3_` 前缀）各在 `risk.chi_levels` 的两档上各跑一次，臂 id 后缀 `__chi0` / `__chi05`。

| 臂 id | 类别 | 07:30 场景 | 07:30 规划 | 12:00 规划 | 风险处理 |
|---|---|---|---|---|---|
| `B2_FB0` | 自制对照 | `s3_forecast_side/point_0730/fb0_lgbm.csv` | `solve_day` 自由备用（同 S2-C） | 不重排 | 无 |
| `B2_FB0_gate` | 自制对照 | 同上 | 同上 | `solve_day` 固定 r_up，日前价用 `point_1200/fb0_lgbm_cap1200.csv` | 无 |
| `B3_hist_paired` | 判定实验 | `hist_paired` | B3 | 无条件，沿用 07:30 场景 | χ 两档 |
| `B3_hist_independent` | 判定实验 | `hist_independent` | B3 | 同上 | χ 两档 |
| `B3_pooled_independent` | 消融：去依赖 | `pooled_independent` | B3 | 同上 | χ 两档 |
| `B3_pooled_gaussian_no_cross` | 消融：去跨目标依赖 | `pooled_gaussian_no_cross_target` | B3 | 同上 | χ 两档 |
| `B3_pooled_gaussian` | 本文未校准 | `pooled_gaussian` | B3 | 无条件 | χ 两档 |
| `B3_pooled_gaussian_gate` | 本文未校准 | 同上 | B3 | `pooled_gaussian__gaussian_24_hour` | χ 两档 |
| `B3_pooled_gaussian_directqr` | 本文未校准 | 同上 | B3 | `pooled_gaussian__direct_qr` | χ 两档 |
| `B3_pooled_empirical` | 本文未校准 | `pooled_empirical` | B3 | 无条件 | χ 两档 |
| `B3_pooled_empirical_analog` | 本文未校准 | 同上 | B3 | `pooled_empirical__analog` | χ 两档 |
| `B3_pooled_empirical_directqr` | 本文未校准 | 同上 | B3 | `pooled_empirical__direct_qr` | χ 两档 |
| `B3_fb2_s3a` | S3-A 衔接 | S3-A `fb2_gate` | B3 | S3-A `scenarios_1200/fb2_gate` | χ 两档 |
| `B3_fb2_plus_weather_s3a` | 上界 | S3-A `fb2_plus_weather` | B3 | S3-A `scenarios_1200/fb2_plus_weather` | χ 两档，只在 209 天评价，标注不可部署 |
| `B3_hist_paired_shortfall` | 敏感性：原追索定义 | `hist_paired` | B3，但交付量 ∈ [0, required]，两个闸门都不做第一步 | 无条件，沿用 07:30 场景 | χ 两档；只报告，不参与判定实验 |
| `X14_rebuild` | 已发表重建 | `x14_rebuild` | B3 结构，激活用期望型 | 以 D 日已公布容量价路径选欧氏距离最近的簇（标准化同第 3.3 节），用该簇场景做第 3.2 节 12:00 规划 | CVaR：χ ∈ {0, 0.1, 0.5}，主行 χ=0；β 用 `risk.cvar_level` |
| `X07_rebuild` | 已发表重建 | 按第 4.2 节抽取结果；决定性字段未定且非明示假设时跳过 | 单次决策：K=1，r_up 与能量计划在 07:30 一起定，12:00 不重排 | 不重排 | CVaR β=0.5；每个场景须全额交付，不可行时按第 3.3 节后备 |
| `X09_rebuild` | 已发表重建 | `x09_block_copula` | 按原文顺序两阶段随机模型改到 07:30 | 按原文 | 按原文 |
| `X06_rebuild` | 已发表重建 | 日前价、容量价用 `fb0_lgbm`；激活用 `climatology` 的逐时经验分位 | 确定性 LP＋逐时机会约束，ε=0.2 按原文在各分位层均分 | 不重排 | 事先设定的 ε |
| `X13_rebuild` | 已发表重建 | 价格用 `fb0_lgbm`（原文为代表性价格曲线，记偏差）；激活比例与持续时间的联合经验分布由 F(D) 的 15 分钟激活事件估计 | 确定性 LP＋联合机会约束化为逐约束机会约束：可靠性 98%，β=99%，样本数按原文式 (3d)，逐约束水平按式 (3b)–(3c) 蒙特卡洛收紧 | 不重排 | 事先设定的可靠性 |

**X14 场景的明示假设**（全文核读值 → DK1 取值；全部写入 yaml `published_rules.x14.deviation`）：

| 字段 | 全文核读值 | DK1 取值 | 状态 |
|---|---|---|---|
| 容量价 | 按 4 小时块从历史均价均匀抽样 | 每小时从 F(D) 同小时容量价中独立均匀抽样 | 产品单元不同，偏差 |
| 日前价、激活价 | 取历史实现值 | 每个场景从 F(D) 均匀抽一个历史日，取其整日日前价路径；激活价另抽一日取整日路径 | 抽样单元未确定，明示假设 |
| 激活 | 历史激活概率 × 0.5 得平均激活功率 | required_t = r_up(h)·α_h·0.5·Δt，α_h = F(D) 中该小时 15 分钟激活量为正的槽占比 | α 定义与窗口未确定，明示假设 |
| 不平衡价 | 日前价加历史溢价 | 同第 3.4 节 | 一致 |
| 场景数与权重 | 未确定 | 50，等权 | 明示假设 |
| 备用持续时长 | 德国规则 1.5 h | 冻结 T = 1 h | 协议差异，偏差 |
| 风电资产 | 400 MW 风电场 | 删除 | 偏差 |
| CVaR 水平 | 未确定 | `risk.cvar_level` | 明示假设 |

除表中各项与上表明确写定的规则外，任何 B3 类臂的 χ、β、K、场景数、时限都在 yaml 预先写定；不得用测试期结果选择。

### 4.2 已发表方法决策部分的抽取字段

`library_paper_reviewer` 按下列字段交回，主线程写入 yaml `published_rules.<id>`（`source`、`source_locator`、`paper_value`、`dk1_value`、`deviation`、`undetermined`）；`source_locator` 必须是全文页码，并在可用时附公式号、表号或章节号：决策时刻与阶段；决策变量；目标函数各项；SOC 与备用持续约束；激活表示方式；非预期性结构；风险度量及其参数原值；场景数；二进制变量及其作用；需删除的资产（风电、光伏、常规机组、需求响应）；市场产品差异（德国 4 小时块、PICASSO、FCR-N、比利时产品）；原文评估方式。

某字段「未确定」且决定决策形状时：论文明示假设则实现最接近变体并记偏差，否则该臂不跑，写入 `summary.json` 的 `skipped_arms`。X14 为例外，按第 4.1 节明示假设执行。

### 4.3 第二批与不做

| 论文 | 处理 | 理由 |
|---|---|---|
| X10 风电—电池自调度 | 第二批：第一批完成后再做 | 去掉风电后退化为点价格＋激活点策略的两阶段随机优化，与 `B2_FB0_gate` 同类 |
| X15 北欧多市场 BESS | 第二批 | GAM 点预测＋另建场景，与 `B3_pooled_independent` 同类 |
| X22 丹麦风电—退役电池 | 第二批 | Monte Carlo 独立场景，与 `X14_rebuild` 同类 |
| X02 西班牙一体化框架 | 第二批 | 多产品点预测＋确定性优化，与 `B2_FB0` 同类 |
| X11 BESS-PV 调频规划 | 第二批 | 时步独立概率预测 |
| L03、X19 | P3 | 决策导向训练，不属 P1 |
| M08、M07、X12、X17 等其余候选 | 不做 | 见论文规划第 6 节候选表理由 |

第二批只在第一批全部通过验收后才开始，且另起提交。

### 4.4 只读参照行

B0、B1、Oracle 读 `p1_paper/results/s2_dk1_baselines/summary.json`；`B2_F01`、`B2_F08`、`oracle_price_reserve` 读 `p1_paper/results/s2c_decision_layer/summary.json`。结果表中注明 `B2_F01`、`B2_F08` 的预测窗口为 116 天，与本任务新臂不同；干净的点预测对照是 `B2_FB0`。

## 5. 评价

### 5.1 指标

| 维度 | 指标 |
|---|---|
| 收益 | 总收益及能量、容量、激活、恢复、未履约五项；日均 |
| 交付 | 未交付激活占比、有未交付的天数、功率缺口与能量缺口 MWh、压力小时（S1 冻结阈值）未交付占比 |
| 尾部 | 日收益最差 5% 的平均值（CVaR5%） |
| 预测风险与实际结算 | 对每个 B3 类臂与 `X14_rebuild`：以冻结计划在其自身 12:00 场景下逐场景调用 `settle_day`（场景作为 `realised`）得到预测日收益分布，取 VaR_5% 与 CVaR_5%；报告实际结算日收益低于预测 VaR_5% 的天数占比（名义 5%）、超越日上实际收益与预测 CVaR_5% 之差的均值，以及 F07 式 (40)（Fissler、Ziegel、Gneiting 2015 的 VaR/CVaR 联合评分，收益以千欧计）的逐日评分 |
| 交叉评分 | 在 `hist_paired`、`hist_independent`、`pooled_independent`、`pooled_gaussian_no_cross_target`、`pooled_gaussian`、`pooled_empirical` 六个 07:30 来源、χ=0.5、12:00 无条件的臂之间：臂 m 的冻结计划放到臂 i 的场景下按上行方法得预测 VaR/CVaR，与臂 m 的实际结算收益计算联合评分，形成 6×6 矩阵；按 F07 式 (30) 对每个 (i, m) 做「i 的评分不劣于 m 自身」的 DM 检验（Newey–West 6 阶） |
| 物理 | 终端 SOC 偏差最大值、槽内双向位置计数（沿用 `activation_ledger` 的列） |
| 可实施性 | 求解时间中位数与最大值、后备触发天数 |
| RQ4 复验 | 对有预测层逐日联合评分的臂（`s3_forecast_side`、`s3_cross_market` 的 `daily_scores` / `daily_joint_scores`，seed 1），按 Energy Score、Variogram Score 与 211 天总收益、未交付占比分别排序，报告 Kendall τ；只报告，不作主张 |
| 分月 | 2026-02 至 2026-08 逐月总收益与未交付占比 |

### 5.2 成对比较与判定

成对比较一律为逐日收益差与未交付差：7 天块自助法 95% 置信区间（2000 次，种子 1）与 DM 检验（Newey–West 6 阶）。预先写定的比较（同一 χ 档内）：

| 比较 | 回答 |
|---|---|
| 每个臂对 `B2_FB0`；每个臂对 `X14_rebuild`（χ=0） | 外部基准 |
| `B3_hist_paired` 对 `B3_hist_independent` | 跨目标配对的决策价值 |
| `B3_pooled_gaussian`、`B3_pooled_empirical` 各对 `B3_pooled_independent` | 依赖结构的决策价值 |
| `B3_pooled_gaussian` 对 `B3_pooled_gaussian_no_cross` | 跨目标依赖的决策价值 |
| `B3_pooled_gaussian` 对 `B3_pooled_empirical` | 两种耦合的决策差别 |
| `_gate`、`_directqr` 对同耦合无条件臂；`_analog`、`_directqr` 对 `B3_pooled_empirical` | 12:00 按已公布容量价更新的决策价值 |
| `B3_pooled_gaussian` 对 `B3_fb2_s3a` | 合并边际相对 S3-A 按小时边际的决策差别 |
| `B3_fb2_plus_weather_s3a` 对 `B3_fb2_s3a`（209 天） | 交割日气象信息缺口的经济代价 |
| 同一臂 χ=0.5 对 χ=0 | 固定风险系数的收益—未交付权衡 |
| `B3_hist_paired` 对 `B3_hist_paired_shortfall`（同 χ） | 交付义务的收益代价：收益差、未交付差、承诺量差 |

前沿：各臂（含 X14 的三个 χ 与 B3 类的两档 χ）在「总收益—未交付占比」平面上的点，及是否被其他臂支配。

判定实验按预先写定的规则**在 χ 两档上分别给出**：`B3_hist_paired` 相对 `B2_FB0` 的日收益差置信区间下界 > 0，或未交付占比差置信区间上界 < 0 且收益差下界 > −1%·`B2_FB0` 日均，记为该档「分布有决策价值」。两档都不满足才记为「分布无决策价值」并触发论文规划第 5 节停止条件。

## 6. 文件与接口

| 文件 | 写入者 | 内容 |
|---|---|---|
| `configs/s3_comparison.yaml` | 主线程 | 储能引用、求解器、`risk`（`chi_levels`、`cvar_level`、X14 的 χ 扫描）、`tree`、`export`、`recourse`（`delivery_0730: full`、`delivery_1200: delivery_first`、`delivery_tolerance_mwh`）、臂清单、`published_rules` 规格 |
| `src/s3_comparison/__init__.py` | 主线程 | 空 |
| `src/s3_comparison/scenario_export.py` | 主线程 | 第 3.5 节 |
| `src/s3_comparison/scenario_io.py` | `scenario_io` 实例 | 第 3.4 节 |
| `src/s3_comparison/hand_checks.py` | `hand_checks` 实例 | 第 7 节第 1–3、10 项 |
| `src/s3_comparison/three_stage.py` | 主线程 | 07:30 三阶段与 12:00 两阶段规划（含 CVaR），每个函数不超过 60 行 |
| `src/s3_comparison/published_rules.py` | 主线程 | X06、X07、X09、X13、X14 的差异化部分与 X14 场景生成 |
| `src/s3_comparison/risk_scoring.py` | 主线程 | 第 5.1 节预测风险、联合评分与交叉评分 |
| `experiments/s3_comparison.py` | 主线程 | `--config`、`--export`、`--check`；按日多进程 |
| `experiments/anchor_table.py`、`configs/anchor_table.yaml` | 主线程 | 第 8 节第 4 步 |
| `p1_paper/results/s3_comparison/` | 主线程 | `summary.json`（环境指纹、种子、臂清单、偏差、跳过的臂、后备天数、判定实验两档结论、验收标志、文件 SHA-256，不写墙钟时间）、`arm_summary.csv`、`daily_results.csv`（列与 `s2c_decision_layer/daily_results.csv` 相同并加 `strategy`、`chi`）、`paired_differences.csv`、`frontier.csv`、`monthly.csv`、`stress_hours.csv`、`risk_calibration.csv`、`cross_scoring.csv`、`rq4_rank_agreement.csv`、`scenario_io_audit.csv`，以及第 3.5 节导出目录 |

## 7. 正确性检查（`--export` 与 `--check`，全量运行前必须通过）

1. **退化一致**：1 个场景、K=1、χ=0、激活为零、价格取实现值时，07:30 三阶段规划的目标值与 `solve_day`（自由备用）一致，误差不超过 yaml 容差。
2. **两场景手算**：有激活与无激活两场景、各 50%，χ=0 与 χ=1 下的解析最优与 LP 一致。
3. **非预期性**：同簇场景的能量计划逐槽相等；12:00 冻结的 r_up 与 07:30 输出逐小时相等。
4. **完美信息一致**：以当日实现值（15 分钟日前价、实现不平衡价、实现激活份额与激活价）作为唯一场景，K=1、χ=0，依次做 07:30 规划、12:00 规划和 `settle_day` 结算。211 天结算总收益与 S2 Oracle 824,744 EUR 之差写入 `summary.json`，相对差超过 0.1% 时停下，逐项说明来源再继续。同时报告 `B3_hist_paired_shortfall` 追索下的同一数值，不设阈值。
5. **结算未改**：`B2_FB0` 以 F08 的 `forecasts.csv` 代入时，211 天总收益等于 S2-C 的 `B2_F08` 560,782 EUR（误差不超过 1e-6 EUR）。
6. **物理**：全部臂日末 SOC 偏差不超过 yaml 容差；毛功率不超过 4 MW。
7. **可复现**：全量运行两次，所有 csv 与 npz 的 SHA-256 一致。
8. **导出复现**：重算的 100 场景版本逐日联合评分与 `s3_cross_market/daily_joint_scores.csv` 中 `own_pooled__independent`、`own_pooled__gaussian`、`own_pooled__empirical` 的 Energy Score、Variogram Score 一致，12:00 产品的日前价逐日 CRPS 与 `daily_conditional_scores.csv` 中 `own_pooled__unconditional`、`__gaussian_24_hour`、`__analog`、`__direct_quantile_regression` 一致，最大绝对误差不超过 1e-9；`__direct_quantile_regression` 的复现在偏移 31 的 100 场景中间样本上核对，该样本不导出；`fb0_lgbm_cap1200` 的日前价 MAE 等于 S3-A `point_metrics.csv` 中的值（误差不超过 1e-9）。
9. **`direct_qr` 组装**： (a) 每个交割日、每个小时，若场景 i 的 07:30 日前价严格小于场景 j，则 i 的 `direct_qr` 日前价不大于 j（逆序对计数为 0）；在 100 场景数据与导出的 50 场景文件上各查一次。 (b) 激活量、激活价两维与同耦合的 07:30 场景逐值相等，NaN 位置相同。 (c) 用重新生成的 07:30 均匀数查 07:30 网格，结果与已存的 07:30 场景逐值相等；`direct_qr` 日前价由同一组均匀数查直接分位数回归网格得到。 (d) Spearman 最小值与并列组数按高斯、经验两个来源分别写入 `summary.json`，只作诊断，不设阈值。
10. **风险评分手算**：已知分布合成收益上，VaR_5%、CVaR_5%、超越率与式 (40) 评分与解析值误差在 yaml 容差内；按真实分布给出的 (VaR, CVaR) 的期望评分不高于偏移后的 (VaR, CVaR)。
11. **X14 生成**：`x14_rebuild` 的容量价每小时取值都属于 F(D) 同小时实际值集合；日前价与激活价每个场景都是某个 F(D) 历史日的整日路径；α_h ∈ [0, 1] 且只由 F(D) 计算。
12. **12:00 优先交付**：每个 B3 类臂、每个交割日，第二步解的 Σ_s π_s Σ_t delivered ≥ D* − yaml 容差；两步求解状态写入 `daily_results.csv`。

## 8. 执行步骤与提交

1. 波次 0：导出通过第 7 节第 8、9 项后，**第一次提交**：yaml 骨架、`scenario_export.py`、导出文件，提交信息写明第 8 项的最大误差。
2. 波次 1 按第 2.2 节。**第二次提交**：`scenario_io.py`、`hand_checks.py` 与审计表，写明实例名。
3. **第三次提交**：规划、规则与风险评分模块、实验脚本与全部结果；提交信息写明判定实验两档结论、各已发表重建的收益与未交付占比、`B3_pooled_gaussian` 与 `B3_pooled_empirical` 相对 `X14_rebuild` 与 `B2_FB0` 的成对差、12:00 条件化的成对差、预测 VaR 超越率、`B3_hist_paired` 与 `B3_hist_paired_shortfall` 的收益、未交付占比与承诺量对比。
4. **第四次提交**：
   - 锚定表：`experiments/anchor_table.py` 与 `configs/anchor_table.yaml` 中的占位行 `FB2_B3` 改为 `FB2_B4`；为 X06、X07、X09、X13、X14 各加一行，`verification_type` 为 `no_anchor`，`note` 为「原论文数据不公开，不可做原任务锚定」；同步修改 `validate` 中的行数（7 → 12）与 `no_anchor` 计数（0 → 5）；重跑锚定表，`p1_paper/results/anchor_table/` 的 `git diff` 只允许出现这六行及 `summary.json` 对应条目。
   - [执行计划](research-foundation-2026/EXECUTION_PLAN.md) 第 10.1 节 S3 行与新增第 10.6 节「S3-B 结论（实测）」，合计不超过 20 行；
   - [论文规划](research-foundation-2026/PAPER_PLAN.md) 第 2 节「戊」行状态；判定实验若两档均为「无决策价值」，同时在第 5 节停止条件下写一行触发记录；
   - [研究框架](research-foundation-2026/RESEARCH_FRAMEWORK.md) 第 8 节末加不超过 8 行的「S3-B 结果」。
5. 审查后的修正单独提交。第二批（第 4.3 节）另起提交。

## 9. 不要做

1. 不改冻结协议与 `src/epf_harness/`、`src/f01_lear_dk1/settlement.py` 的任何逻辑；不改 `src/s3_forecast_side/`、`src/s3_cross_market/`、`src/s3_joint_surfaces/` 与已有实验、结果（锚定表按第 8 节例外；导出只通过新文件复用上述模块）。
2. 不用测试期结果选 χ、β、K、场景数、ε、可靠性或任何参数；不在 211 天上重新挑选预测侧规格或耦合方式。
3. 不让任何臂获得他臂没有的信息；`B3_fb2_plus_weather_s3a` 除外且显式标注。
4. 不实现 B4 或任何在线更新规则。
5. 不把重建称为复现；原论文数字只能标 `paper-reported`，不得与本任务结果放在同一列比较。
6. 不引入整数规划或新求解器依赖。
7. 不把 RQ4 复验写成本文贡献。
8. 不提交付费全文 PDF；不新建 AGENTS.md 禁止的文件；不用 `git add -A` 或 `git add .`；不直接推 `main`。
9. 研究文档任一文件超过 400 行时停下报告。

## 10. 验收

```bash
cd "$(git rev-parse --show-toplevel)"
export PY=/home/zl/nvme/.venvs/r1_lear_de/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" experiments/s3_comparison.py --config configs/s3_comparison.yaml --export
"$PY" experiments/s3_comparison.py --config configs/s3_comparison.yaml --check
"$PY" experiments/s3_comparison.py --config configs/s3_comparison.yaml
find p1_paper/results/s3_comparison -type f \( -name '*.csv' -o -name '*.npz' \) -exec sha256sum {} + | sort > /tmp/s3b_run1.sha
"$PY" experiments/s3_comparison.py --config configs/s3_comparison.yaml
find p1_paper/results/s3_comparison -type f \( -name '*.csv' -o -name '*.npz' \) -exec sha256sum {} + | sort | diff - /tmp/s3b_run1.sha
find p1_paper/results/s3_comparison -size +20M
"$PY" experiments/anchor_table.py --config configs/anchor_table.yaml
"$PY" scripts/audit_multi_market_resources.py
wc -l research-foundation-2026/*.md
git diff --stat main -- p1_paper/results/ src/epf_harness/ src/f01_lear_dk1/ src/s2c_decision_layer/ src/s3_forecast_side/ src/s3_cross_market/ src/s3_joint_surfaces/
```

通过条件：`--export` 第 8、9 项与 `--check` 其余各项通过；两次运行哈希一致；无超过 20 MB 的文件；资源审计退出码 0（`day_ahead_prices_raw.json` 缺失时按 `download_manifest.json` 的 `resolved_url` 重新下载并核对 sha256，不放宽断言）；研究文档均不超过 400 行；`p1_paper/results/` 只出现 `s3_comparison/` 与 `anchor_table/` 的变化；上列 `src/` 目录无改动。

## 11. 停止条件与 PR

立即停下并报告：前提文件缺失或校验失败；第 7 节第 4、5、8、12 项不通过；需要改动第 9 节第 1 条所列冻结项；后备触发超过 yaml `max_fallback_share`（写定 5%）；导出文件超过 20 MB。

判定实验两档均为「无决策价值」时照常完成全部臂并如实报告，不调参挽救。

PR（`codex/s3-comparison` → `main`）描述用三到五句：判定实验两档结论；五个已发表重建的收益与未交付占比排名；`B3_pooled_gaussian`、`B3_pooled_empirical` 相对 `X14_rebuild` 与 `B2_FB0` 的成对差及置信区间，以及 12:00 条件化的成对差；预测 VaR 超越率；交付义务敏感性（`B3_hist_paired` 对 `_shortfall`）的收益差与未交付差；被跳过的臂及原因。**开 PR 后停下，等用户确认后再进入 S3-C。**
