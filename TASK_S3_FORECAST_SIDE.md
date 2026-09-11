# 任务 S3-A：预测侧完整实验（子代理分工版）

目标：在冻结的 DK1 协议上一次跑完预测侧全部臂——自制对照、已发表方法的场景生成器重建、本文的闸门条件联合场景——并导出任务 S3-B 直接读取的场景文件。
改哪里：新建 `configs/s3_forecast_side.yaml`、`src/s3_forecast_side/`、`experiments/s3_forecast_side.py`、`p1_paper/results/s3_forecast_side/`；按第 8 节更新三份研究文档的指定位置。
不要做：第 9 节全部条目。
验收：第 10 节命令全部通过，`redline_reviewer` 交回表中无「阻塞」「需修正」。

基线：`main` 最新提交。环境固定 `/public/ZLCODE/.venvs/r1_lear_de/bin/python`，不升级、不新装依赖。已核实可用：numpy 1.21.6、pandas 1.3.5、scipy 1.7.3（`linprog` 含 HiGHS）、scikit-learn 0.24.2（无 `QuantileRegressor`）、statsmodels 0.13.2（`QuantReg`）、lightgbm 4.6.0、PyYAML；**没有 pyarrow**，结果只用 csv 与 npz。

**分支与 PR**：从 `main` 开 `codex/s3-forecast-side`，按第 8 节分次提交，开 PR 后停下等用户确认，不直接推 `main`。本文件即已确认的改动计划：执行中不因 AGENTS.md 第 6 节「超过 30 行先停」再次停下；停点只有两个——开出 PR 之后，或触发第 11 节停止条件。

**启动方式**：Codex 不会自动派生子代理，必须按第 2.3 节明确派出。运行时若限制并发，按表中顺序错峰派出。

---

## 0. 已确定的定位（不要重新论证）

X14 已给出「容量闸门 → 日前闸门 → 实时」的三阶段随机决策，X07 已给出「分位数边际＋静态经验 Copula」的联合场景。P1 的主张收缩为三条，定义见 [论文规划](research-foundation-2026/PAPER_PLAN.md) 第 6 节：

| 主张 | 本任务的作用 |
|---|---|
| 按已实现未交付事件在线校准风险系数（主） | 不在本任务；本任务产出它需要的场景 |
| 闸门条件联合场景：日前价以 09:20 已公布容量价为条件，激活保留日内路径 | **本任务检验其预测层价值** |
| 同协议外部基准：在后改制 DK1 上按实际结算重建已发表方法 | 本任务重建其**场景生成部分**（X14、X07、X09、X06），决策部分在 S3-B |

三类臂必须分清：

| 角色 | 本任务中的臂 | 论文里的位置 |
|---|---|---|
| 自制对照 | 持续法、FB0、FB1、气候态、历史日两种抽样 | 主结果表，隔离增益来源 |
| 已发表方法重建 | F01、F08（读已有结果）、X14 式、X07 式、X09 式、X06 式 | 主结果表；一律称「同协议重建」，不称复现 |
| 本文 | FB2 及其两项去依赖消融；FB2+ 为不可部署上界 | 主结果表与消融 |

顶会结构（F18、F20、F22、F28、F31 等）只作 FB2 的零件，不作为对比方法。

## 1. 名称与代号

首次出现写「名称（代号）」。编号定义以 [CONVENTIONS.md](CONVENTIONS.md) 第 2 节为准。

| 代号 | 名称 | 臂 id |
|---|---|---|
| FB0 | 独立点预测 | `fb0_lgbm` |
| FB1 | 独立概率预测（与 FB2 同边际，目标间、时段间独立） | `fb1_qr` |
| FB2 | 闸门条件联合概率场景（本文） | `fb2_gate` |
| FB2+ | FB2 加交割日风光日前预测（不可部署上界） | `fb2_plus_weather` |

## 2. 子代理编排

### 2.1 拆分原则

读得多、交回摘要的工作交给只读子代理；互不依赖、边界清楚的模块交给写入子代理，每个实例只写自己名下的文件；强顺序、写共享文件的工作（yaml、模型、实验编排、研究文档、提交、PR）留在主线程。

### 2.2 角色与写入范围

定义文件在 `.codex/agents/`。子代理只写本表列在其实例名下的文件。

| 子代理（实例） | 权限 | 负责 | 允许写入 |
|---|---|---|---|
| `library_paper_reviewer` | 只读 | 抽取 X14、X07、X09、X06 场景生成部分的可重建规格，字段见第 4.4 节 | 无 |
| `structure_scout` | 只读 | 方向 a：从 F18、F20、F22、F23、F04 抽取边际两段式、低秩高斯 Copula 估计与条件化、两阶段训练、概率评分的公式与短窗陷阱；另核验 Energy Score、Variogram Score、DM 检验的原始出处与载体 | 无 |
| `literature_searcher` | 只读 | 检索问题：是否已有工作（1）以已公布的备用容量出清结果为条件生成日前价场景，或（2）按已实现的备用未交付事件在线校准储能承诺的风险参数 | 无 |
| `dk1_data_profiler`（实例 `panel`） | 可写 | 第 3 节的小时面板、可见性、窗口与特征 | `src/s3_forecast_side/panel.py`；`p1_paper/results/s3_forecast_side/panel_audit.csv` |
| `dk1_data_profiler`（实例 `scoring`） | 可写 | 第 5 节全部指标与第 7 节中的评分手算 | `src/s3_forecast_side/scoring.py` |
| `redline_reviewer` | 只读 | 开 PR 前审查分支相对 `main` 的全部改动 | 无 |
| **主线程** | 可写 | yaml、`__init__.py`、`marginals.py`、`copulas.py`、`generators.py`、实验脚本、结果、研究文档、提交与 PR | 除上面两个实例专属文件外本任务涉及的全部文件 |

X07、X14 的全文 PDF 为付费全文，只在主检出目录本地未跟踪存放，**不得提交**：
`/public/ZLCODE/electricity-price-forecasting-research/paper/multi_market_energy_reserve/01_intersection_frontier/02_peer_reviewed_specialized/` 下文件名以 `X07_2022_`、`X14_2022_` 开头的两个 PDF。读不到时以同目录来源卡为准并注明。

### 2.3 执行顺序

| 波次 | 谁 | 做什么 | 前提 |
|---|---|---|---|
| 0 | 主线程 | 开分支；写 `configs/s3_forecast_side.yaml` 的 data、protocol、window、split、targets、visibility、features、scoring 键与 `src/s3_forecast_side/__init__.py` | 无 |
| 1（并行） | `library_paper_reviewer`、`structure_scout`、`literature_searcher`、`dk1_data_profiler` 的 `panel` 与 `scoring` 两个实例 | 按第 2.2 节 | 波次 0 |
| 主线程 A | 主线程 | 把第 4.4 节规格与偏差写入 yaml `published_generators`；实现第 4 节全部臂与实验脚本；跑 `--check`；全量运行两次 | 波次 1 全部交回 |
| 主线程 B | 主线程 | 按第 8 节写研究文档；把 `literature_searcher` 结论写入论文规划第 6 节主张边界表的「已有工作」列 | 主线程 A |
| 2 | `redline_reviewer` | 审查全部改动 | 主线程 B |
| 主线程 C | 主线程 | 修正「阻塞」「需修正」条目，开 PR，停下 | 审查交回 |

### 2.4 主线程整合规则

- 只采用子代理交回的、带出处的内容；标为「未确定」「未核验」「未检索」的原样保留，不得自行补全。
- 两个子代理结论冲突时回到原文核对，并在提交信息里说明。
- 子代理失败或无法联网：该部分记为「未完成」写入 `summary.json` 的 `incomplete`，不得凭记忆补编。

## 3. 数据与协议

### 3.1 预测目标（小时级，按民用交割日 24 期）

| 目标 | 来源列（PriceArea=DK1） | 小时聚合 | 07:30 可见规则 |
|---|---|---|---|
| `day_ahead_price` | `day_ahead_prices.csv` 的 `DayAheadPriceEUR` | 四个 15 分钟值取均值 | 交割日 d 的整日值于 d-1 12:57 公布 |
| `capacity_price` | `afrr_capacity_market.csv` 的 `UpPriceEUR` | 原值 | 交割日 d 的整日值于 d-1 09:20 公布 |
| `activation_volume` | `imbalance_and_activation.csv` 的 `aFRRUpMW` | 四个区间求和，不乘 0.25 | 该小时结束后可见 |
| `activation_price` | 同表 `aFRRVWAUpEUR` | 按区间激活量加权；小时激活量为 0 时**缺失** | 该小时结束后可见 |

夏令时切换日按冻结规则 `truncate_or_repeat_last` 规整为 24 期（同 `src/f01_lear_dk1/data.py`）；可见时刻必须先在原始 UTC 行上计算，再规整。可见规则与 `configs/s3_dk1_target_profile.yaml` 的 `availability_kind` 一致，`panel.py` 须断言两者相同。

### 3.2 两个决策时刻

- **07:30 快照**：交割日 D 的决策时刻为 D-1 07:30 Europe/Copenhagen，逐日换算 UTC。
- **12:00 快照**：D-1 12:00；与 07:30 相比只多出 D 日容量价（已于 09:20 公布）。激活特征仍用 07:30 快照，记为保守设定。

每个特征、标签与超参选择都要有 `available_at <= decision_time` 断言。

### 3.3 滚动窗口（冻结协议第 8.1 节的逐日口径，按民用日历日计）

| 块 | 相对测试日 D 的范围 | 长度 | 用途 |
|---|---|---|---|
| 拟合块 F(D) | [D-119, D-36] | 84 天 | 拟合全部边际模型、Copula、FB0 |
| 概率校准块 C(D) | [D-35, D-8] | 28 天 | 用 F(D) 拟合的模型预测这 28 天，做重校准与超参选择 |
| 清除间隔 | [D-7, D-1] | 7 天 | 不参与任何估计；保证 C(D) 标签在 D-1 07:30 已完整可见 |

D 日预测使用 F(D) 拟合的同一套模型加 C(D) 得到的重校准映射。每个测试日重估一次。块内已剔除交割日自然缺席，实际天数写入 `panel_audit.csv`。断言：D=2026-01-28 时 F(D) 为 2025-10-01 至 2025-12-23、含 81 个可用日，C(D) 为 2025-12-24 至 2026-01-20。

与 F01/F08 的差别必须在结果中声明：二者用截至 D-1 的 116 天窗且无概率校准块。本任务所有新臂统一使用上表口径，不得混用。

### 3.4 测试日

评价集合 = S2 结算的 211 个交割日（`p1_paper/results/s2_dk1_baselines/daily_results.csv` 的唯一交割日），断言逐日相同。2026-08-26 因 D-1 落在已剔除日而不在集合内。

### 3.5 特征（所有基于边际模型的臂共用）

| 目标 | 模型粒度 | 特征 |
|---|---|---|
| `day_ahead_price`、`capacity_price` | 每目标每小时一个模型 | 最近可见同小时值（d-1）；d-7 同小时值；最近 7 个可见同小时值均值（有效值不少于 5 个）；交割日是否周末 |
| `activation_volume` | 每小时一个模型 | 最近 7 个可见同小时值均值（有效值不少于 5 个）；同一窗口内该小时零值占比；是否周末 |
| `activation_price` | 24 小时合并一个模型 | 六个四小时时段哑变量；最近 7 个可见交割日的量加权激活价；是否周末 |

「最近可见同小时值」对激活类目标：若 d-1 该小时已在 07:30 前结束则取 d-1，否则取 d-2。特征缺失的行不进入拟合与校准，计数写入审计；测试日任一特征缺失即断言失败。

### 3.6 缺失

缺失保持缺失，不补零、不插值、不前向填充。`activation_price` 在零激活小时缺失：不进入其边际拟合与评价；Copula 估计用成对完整样本；场景中该维照常生成（表示「若被激活时的价格」）。

## 4. 预测臂

### 4.1 总表

| 臂 id | 角色 | 边际 | 依赖 | 12:00 产品 | 重校准 | 导出场景 |
|---|---|---|---|---|---|---|
| `persistence` | 自制对照 | 点：最近可见同小时值 | — | — | 否 | 点 csv |
| `fb0_lgbm` | FB0 | 点：LightGBM，参数逐项等于 `configs/f08_lightgbm_dk1.yaml` 的 `model.params` | — | `fb0_lgbm_cap1200`：日前价模型加入 D 日同小时与日均容量价 | 否 | 点 csv |
| `f01_lear`、`f08_lightgbm` | 已发表重建（读已有结果） | 点：读 `p1_paper/results/{f01_lear_dk1,f08_lightgbm_dk1}/forecasts.csv` | — | — | 否 | 不重跑 |
| `climatology` | 自制对照；激活量部分即 X06 式逐时经验分布 | F(D) 逐时经验分位数 | 独立 | — | 否 | 是 |
| `fb1_qr` | FB1 | 第 4.2 节 | 独立 | — | 是 | 是 |
| `hist_paired` | 自制对照（判定实验输入） | 无模型：F(D) 中均匀抽取历史日，四个目标取同一天 | 历史日内全部保留 | — | 否 | 是 |
| `hist_independent` | 自制对照 | 同上，但每个目标各自独立抽日 | 仅目标内日内依赖 | — | 否 | 是 |
| `empirical_copula` | 自制对照（简单联合基线） | `fb1_qr` | F(D) 经验 Copula，全部 96 维 | 类比条件化：F(D) 中容量价路径最接近的 10 天 | 是 | 是 |
| `fb2_gate` | FB2（本文） | `fb1_qr` | 第 4.3 节低秩高斯 Copula | 精确高斯条件化 | 是 | 是 |
| `fb2_no_cross_target` | FB2 消融 | 同上 | 相关阵中跨目标块置零 | 无（条件化失效） | 是 | 是 |
| `fb2_no_temporal` | FB2 消融 | 同上 | 相关阵中同目标跨小时块置零 | 精确高斯条件化 | 是 | 是 |
| `x14_hist_independent` | X14 式重建 | 第 4.4 节 | 各变量独立抽历史日 | — | 否 | 是 |
| `x07_static_copula` | X07 式重建 | 第 4.4 节 | 静态经验 Copula | — | 否 | 是 |
| `x09_block_copula` | X09 式重建 | 第 4.4 节 | 价格块联合、激活块独立 | — | 按原文 | 是 |
| `fb2_plus_weather` | FB2+ 上界 | `fb1_qr` 加 D 日 `ForecastDayAhead`（陆上风、海上风、光伏小时值） | 同 `fb2_gate` | 精确高斯条件化 | 是 | 是 |

`fb2_plus_weather` 是唯一允许越过 07:30 信息集的臂：只有它可以读 `wind_solar_forecasts.csv` 的 D 日 `ForecastDayAhead`，代码须断言其他臂未读，结果标注 `information_set: beyond_gate_upper_bound`。

### 4.2 共同边际 `fb1_qr`

- 分位点：`[0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995]`，写在 yaml。
- 价格类与激活价：statsmodels `QuantReg` 线性分位数回归；分位数交叉用排序重排（Chernozhukov 重排）。未收敛直接报错，不改用其他估计器。
- 激活量：两段式（F18 结构）。零概率 p0 用 `LogisticRegression`（参数写 yaml）；正值部分对 log(激活量) 做 `QuantReg`。混合分位函数：τ ≤ p0 时为 0，否则为 exp(Q_pos((τ−p0)/(1−p0)))。
- 分位点之间线性插值；[0.005, 0.995] 之外取端点值，并在尾部指标中报告这一截断。
- 重校准：每个目标把 C(D) 的 28×24 个 PIT（零值随机化 PIT）合并，估计其经验分布 R，D 日在名义水平 τ 处取 Q(R⁻¹(τ))。随机化使用 yaml 种子。

### 4.3 本文 `fb2_gate`

1. 在 F(D) 上计算每行在 F(D) 拟合边际下的 PIT（零值随机化），转成正态分数 z，截断在 Φ⁻¹(0.001) 与 Φ⁻¹(0.999)。
2. 用成对完整样本估计 96×96 相关阵（缺失的激活价维只用成对完整对）。
3. 主轴因子法拟合 Σ = ΛΛᵀ + Ψ，秩 r ∈ yaml `rank_candidates: [1, 2, 3]`；每个 D 用 C(D) 上的平均 Energy Score 选秩，只看 C(D)。
4. 07:30：从 N(0, Σ) 抽样 → Φ → 重校准后的边际分位函数，得到联合场景。
5. 12:00：以 D 日已公布容量价在 07:30 边际下的正态分数为条件，按高斯条件分布抽取日前价、激活量、激活价的 72 维。
6. 报告每个 D 的秩、可训练参数数与有效样本/参数比。

### 4.4 已发表方法的场景生成重建

规格以 `library_paper_reviewer` 交回为准，每项写入 yaml `published_generators.<id>`：`source`（文件、页、式号）、`paper_value`、`dk1_value`、`deviation`、`undetermined`。下列为已核读的起点：

| 臂 | 论文做法（据全文或来源卡） | DK1 重建 | 需抽取确认的字段 |
|---|---|---|---|
| `x14_hist_independent` | 容量价按 4 小时块从历史均价均匀抽样；日前价与激活价取历史实现值；激活为历史激活概率 α 乘 δ=0.5 的平均功率 | 容量价按 1 小时从 F(D) 抽历史日；日前价独立抽日；激活价按小时从 F(D) 有效值中独立抽；激活量为确定性期望 α_h·δ·F(D) 同小时平均采购量，标注 `activation_representation: expected_value` | α 的精确定义与统计窗口；δ 的含义；各变量抽样是否整日；场景数与概率权重 |
| `x07_static_copula` | 每变量 BLSTM 分位数回归（1%–99%）后，用全部历史日轨迹拟合分段线性经验 Copula 抽样；容量价与激活价为常数 | 日前价与激活量用 `QuantReg`（不做两段式、不重校准，负分位按物理下界 0 截断并记录）；经验 Copula 拟合于 F(D) 的 48 维；容量价与激活价取 F(D) 均值常数 | 分位点集合；Copula 构造细节；激活比例的定义；是否有重校准 |
| `x09_block_copula` | 七类价格组成整日联合场景，风、价格块与调节需求块相互独立 | 价格块（日前价、容量价、激活价）按原文场景方法生成；原文方法在 84 日窗不可重建时，改为 `fb1_qr` 边际＋价格块经验 Copula，并记为偏差；激活量块独立生成、只保留自身日内依赖 | 价格场景生成方法、场景数、是否用历史日、块的划分 |
| `climatology` 的激活量部分 | X06 以历史观测构造逐时激活经验分布并取分位数 | F(D) 逐时经验分位数 | 累计量还是逐时量；分位水平 |

原文某字段为「未确定」且决定场景形状时：仅当论文把它写成明示假设才实现最接近的变体并记偏差；否则该臂不跑，写入 `summary.json` 的 `skipped_arms` 及原因。

### 4.5 场景数与种子

每个测试日生成 S=100 个场景用于评分；导出前 50 个。抽样种子 yaml `sampling_seeds: [1, 2, 3]`，主表用种子 1，另报三种子均值与标准差。逐日种子 = 基础种子 + 测试日序号。

## 5. 评价

### 5.1 指标

| 层 | 指标 | 口径 |
|---|---|---|
| 点 | MAE、RMSE、rMAE（相对 D-7 同小时朴素预测） | 四个目标；概率臂用中位数；激活价只在正激活小时 |
| 边际 | 17 个分位点平均 pinball loss；样本 CRPS；50/80/90/98% 中心区间覆盖率与平均宽度；随机化 PIT 的 KS 距离 | 每目标；激活量另报零事件 Brier score 与正值小时 CRPS |
| 尾部 | 超过 q0.95、q0.99 的频率 | 全部小时，及 S1 冻结压力小时：容量价 ≥ 56.53 EUR/MW、激活量 ≥ 38.70 MWh/h |
| 联合 | Energy Score、Variogram Score（p=0.5，等权） | 每日 72 维（日前价、容量价、激活量），各目标除以 F(D) 实现值标准差；另报价格块 48 维与激活块 24 维 |
| 依赖复现 | 场景与实现值的 Spearman ρ 之差：日前价—容量价、日前价—激活量、容量价—激活量（同小时与滞后 1 小时）；P(容量价>q0.9 ∣ 日前价>q0.9) 之差；各目标滞后 1 小时自相关之差 | 211 天汇总 |
| 12:00 条件 | 日前价 CRPS 与中位数 MAE：`fb2_gate` 条件 vs 其 07:30 无条件、`empirical_copula` 类比条件、`fb1_qr`、`fb0_lgbm_cap1200`、`f08_lightgbm` | 211 天 |
| 漂移 | 逐月 90% 区间覆盖偏差 | 2026-02 至 2026-08 |
| 规模 | 可训练参数数、有效样本/参数比 | 每臂每日 |

### 5.2 显著性

- 以日为单位的损失差序列做 DM 检验，方差用 Newey–West（6 阶滞后），双侧。
- 7 天块自助法，2000 次，种子 1，给出平均差的 95% 置信区间。
- 主比较：`fb2_gate` 对 `fb1_qr`、`empirical_copula`、`x07_static_copula`、`x14_hist_independent`、`x09_block_copula`；12:00 条件对无条件。

### 5.3 与研究框架第 11 节要求的对应

| 要求 | 本任务检验 |
|---|---|
| 1 零质量与重尾 | 零事件 Brier、正值 CRPS、q95/q99 超越频率 |
| 2 目标对依赖不同 | 三个目标对的 ρ 与尾部条件频率误差 |
| 3 完整日路径 | Variogram Score、滞后自相关误差、`fb2_no_temporal` 消融 |
| 4 可校准不确定性 | 覆盖率、PIT、与同信息集点预测的成对比较 |
| 5 短窗可估计 | 参数数、有效样本/参数比、秩选择分布 |
| 6 漂移 | 逐月覆盖偏差；断言每次拟合终点早于 D-7 |
| 7 显式联合＋独立对照 | `fb2_gate` 对 `fb1_qr`、`fb2_no_cross_target`；12:00 条件化 |
| 8 决策风险校准 | 不在本任务，见 S3-B 与 S3-C |

## 6. 文件与接口

| 文件 | 写入者 | 内容 |
|---|---|---|
| `configs/s3_forecast_side.yaml` | 主线程 | 全部超参、窗口、分位点、种子、阈值、`published_generators` 规格 |
| `src/s3_forecast_side/__init__.py` | 主线程 | 空 |
| `src/s3_forecast_side/panel.py` | `panel` 实例 | 小时面板、可见时刻、两个快照、窗口块、特征、审计表 |
| `src/s3_forecast_side/scoring.py` | `scoring` 实例 | 第 5 节全部指标、DM、块自助法、评分手算 `hand_checks()` |
| `src/s3_forecast_side/marginals.py` | 主线程 | 持续法、FB0、`fb1_qr` 两段式、气候态、重校准 |
| `src/s3_forecast_side/copulas.py` | 主线程 | 经验 Copula、低秩高斯 Copula、高斯条件化、类比条件化 |
| `src/s3_forecast_side/generators.py` | 主线程 | 第 4.1 节全部臂的组装与已发表重建 |
| `experiments/s3_forecast_side.py` | 主线程 | `--config`、`--check`；多进程按日并行，进程数写 yaml |
| `p1_paper/results/s3_forecast_side/` | 主线程 | 见下 |

结果目录：`summary.json`（环境指纹含 lightgbm 版本、种子、窗口口径、测试日哈希、臂清单、偏差、跳过的臂、参数规模、验收标志、各文件 SHA-256；**不写墙钟时间**）、`point_metrics.csv`、`marginal_metrics.csv`、`tail_metrics.csv`、`joint_metrics.csv`、`dependence_errors.csv`、`conditional_1200.csv`、`significance.csv`、`monthly_calibration.csv`、`daily_scores.csv`、`panel_audit.csv`、`point_0730/<arm>.csv`、`scenarios_0730/<arm>.npz`、`scenarios_1200/<arm>.npz`。

点预测 csv 列：`delivery_day, target, hour, forecast, actual`（与 S2-C 读取格式一致）。

`scenarios_0730/<arm>.npz`：`delivery_days`（211，字符串）、`targets`（上表四个名称）、`values` float32 形状 (211, 50, 4, 24)、`seed`、`information_set`、`activation_representation`（`path` 或 `expected_value`）。`scenarios_1200/<arm>.npz`：`values` 形状 (211, 50, 3, 24)，目标为日前价、激活量、激活价，另存 `conditioning: realized_capacity_price`。每个文件不超过 20 MB，超过即停。

## 7. 正确性检查（`--check`，全量运行前必须通过）

1. 退化场景（全部等于一个点）的样本 CRPS 等于绝对误差；一维 Energy Score 等于 CRPS；场景全等于观测时 Energy Score 为 0。
2. 已知分布合成数据上 pinball、覆盖率、PIT 与解析值误差在 yaml 容差内。
3. 合成 Σ 上高斯条件抽样的样本均值、协方差与解析条件分布一致。
4. 在特征中注入 D 日数值或 07:30 后才结束的激活小时，可见性断言必须报错。
5. D=2026-01-28 的三个窗口块与第 3.3 节断言一致；测试日集合与 S2 的 211 天逐日一致。
6. `fb0_lgbm` 参数与 `configs/f08_lightgbm_dk1.yaml` 逐项相等。
7. 全量运行两次，所有 csv 与 npz 的 SHA-256 一致。

## 8. 执行步骤与提交

1. 波次 0 与波次 1 按第 2.3 节。
2. 第一次提交：`panel.py`、`scoring.py`、yaml 骨架与审计表；提交信息写明由哪个子代理实例完成。
3. 第二次提交：模型、生成器、实验脚本与全部结果；提交信息写明关键发现（FB2 相对各对照的 Energy Score 与 12:00 条件化增益、激活尾部覆盖）。
4. 第三次提交：研究文档。只改这几处，数字从结果文件读出，不手算：
   - [执行计划](research-foundation-2026/EXECUTION_PLAN.md) 第 10.1 节 S3 行与第 10.4 节，合计不超过 15 行；
   - [论文规划](research-foundation-2026/PAPER_PLAN.md) 第 2 节「丁」行状态，以及第 6 节主张边界表的「已有工作」列（写入 `literature_searcher` 结论，保留「未检索」）；
   - [研究框架](research-foundation-2026/RESEARCH_FRAMEWORK.md) 第 11 节要求表之后加不超过 10 行的「S3-A 检验结果」，逐条对应要求 1–7。
5. 审查后的修正单独提交。

## 9. 不要做

1. 不改冻结协议：窗口长度、重校准周期、划分、指标定义、缺失规则、储能参数、结算公式；不改 `src/epf_harness/` 与任何已有实验、结果。
2. 不用测试期（211 天）的任何结果选模型、选秩、选分位点、选场景数或调参；选择只看 C(D)。
3. 不让任何臂获得他臂没有的信息；`fb2_plus_weather` 除外且须显式标注。
4. 不补缺失值，不对时间序列随机切分，不把芬兰、德国、RTS-GMLC 数据与 DK1 行级拼接。
5. 不实现决策层、不做结算、不实现风险校准（S3-B、S3-C 的范围）。
6. 不引入深度模型或基础模型（TACTiS、TimeGrad、CSDI、Moirai、Chronos 等）；它们不在本任务范围。
7. 不把已发表方法重建称为「复现」；不把顶会结构列为对比方法。
8. 不提交付费全文 PDF；不新建 README、SUMMARY、CHANGELOG、`*_GUIDE.md`、`*_NOTES.md`、`*_REPORT.md`；不用 `git add -A` 或 `git add .`；不直接推 `main`。
9. 研究文档任一文件超过 400 行时停下报告。

## 10. 验收

```bash
cd /public/ZLCODE/electricity-price-forecasting-research
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/s3_forecast_side.py --config configs/s3_forecast_side.yaml --check
/public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/s3_forecast_side.py --config configs/s3_forecast_side.yaml
find p1_paper/results/s3_forecast_side -type f \( -name '*.csv' -o -name '*.npz' \) -exec sha256sum {} + | sort > /tmp/s3a_run1.sha
/public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/s3_forecast_side.py --config configs/s3_forecast_side.yaml
find p1_paper/results/s3_forecast_side -type f \( -name '*.csv' -o -name '*.npz' \) -exec sha256sum {} + | sort | diff - /tmp/s3a_run1.sha
find p1_paper/results/s3_forecast_side -size +20M
/public/ZLCODE/.venvs/r1_lear_de/bin/python scripts/audit_multi_market_resources.py
wc -l research-foundation-2026/*.md
git diff --stat main -- p1_paper/results/ src/epf_harness/ experiments/ configs/
```

通过条件：`--check` 全部通过；两次运行哈希无差异；无超过 20 MB 的文件；资源审计退出码 0（工作树缺 `day_ahead_prices_raw.json` 时从主检出硬链接补齐，不放宽断言）；三份研究文档均不超过 400 行；`git diff --stat` 中 `p1_paper/results/` 只出现 `s3_forecast_side/`，`src/epf_harness/` 与已有实验、配置无改动。

## 11. 停止条件与 PR

立即停下并报告：
- 测试日集合不等于 S2 的 211 天，或任一可见性断言失败；
- 需要改动第 9 节第 1 条所列任何冻结项；
- 某个已发表方法的决定性字段「未确定」且论文未把它写成明示假设（该臂按第 4.4 节跳过并继续其余工作，不算停止）。

FB2 不优于对照时照常完成并如实报告，不在测试期上调参。

PR（`codex/s3-forecast-side` → `main`）描述用三到五句：Energy Score 排名与 FB2 相对 `fb1_qr`、`empirical_copula`、X07 式、X14 式的显著性；12:00 条件化对日前价 CRPS 的增益；激活量零事件与 q99 覆盖；被跳过的臂及原因。**开 PR 后停下，等用户确认后再进入 S3-B。**
