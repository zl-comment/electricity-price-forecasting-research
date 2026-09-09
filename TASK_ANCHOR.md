# 任务 A：锚定表　｜　任务 B：R2 芬兰 aFRR（部分复现）

目标：建一张表，对每条外部基线同时回答「这个实现在它原来的任务上对得上吗」和「搬到 DK1 后表现如何」，用来挡住「你把别人的方法做成了稻草人」这一质疑。

基线：`main` = `4821d21`。环境固定为 `/public/ZLCODE/.venvs/r1_lear_de/bin/python`。本文件在两个任务都结束后删除。

---

## 0. F06 全文已取得，协议逐项核对的结果

PDF 已进仓库（`paper/.../F06_2026_Energy_AI_Finnish_aFRR_Forecasting.pdf`，SHA-256 `395e74e6…d29cbc`，Åbo Akademi 机构仓储 CC BY 正式发表版）。全文 17 页已逐页读过，B1 闸门那五项的核对结果如下。

| # | 项目 | 结果 | 出处 |
|---|---|---|---|
| 1 | 用哪个 CSV | ✅ **v2**（`extended_data_v2.csv`，6,456 数据行） | 第 4 节与第 8 节两处写「约 6456 个小时样本」 |
| 2 | 时间划分 | ✅ 70/15/15 按时间不打乱；训练 4,497 / 验证 976 / 测试 960；测试期 **2025-02-05 至 2025-03-16** | 第 6.1 节；测试期见 Fig. C.12 图注 |
| 3 | MAE 口径 | ✅ 式 (4) 标准 MAE，**Up 与 Down 分开两个目标**，EUR/MWh，**不剔除异常值**，单次确定性运行，**随机种子 42** | 第 5.5 节、第 4 节、Table 4 表题 |
| 4 | 特征集 | ⚠️ **半给** | 见 0.2 |
| 5 | LightGBM 超参 | ❌ **没给** | 见 0.3 |

### 0.1 精确靶子（Table 4，连续测试块，LightGBM）

**此前把「17.5–17.8 / 55.7–60.1」当成不确定性区间是错的**，它是四个模型之间的分布（LightGBM / XGBoost / TFT / TiDE）。LightGBM 有精确点值：

| 目标 | MAE | nMAE | RMSE | MASE | RMSSE |
|---|---:|---:|---:|---:|---:|
| Down | **17.58** | 0.963 | 26.32 | 0.482 | 0.569 |
| Up | **55.74** | 0.603 | 71.17 | 0.498 | 0.553 |

参考量级：论文报告 Down 均价 8.67 EUR/MWh、Up 均价 64.70 EUR/MWh。Down 的 MAE 大于其均价，nMAE 0.963 与此一致——**不要以为 MAE 17.58 就算准**。

### 0.2 特征集为什么只算半给

- Table A.1 列全了约 96 个特征的名称与定义，可照抄。
- Table 2 给五个特征组的规模：气象 7、市场 8、派生市场 30、派生气象 31、时间 20。
- Table 3 给五个特征集各组取多少个（Set 1 共 40 … Set 5 共 83）。
- **但全文没说 Table 4 用的是哪一套，也没给「集合 → 具体特征名」的映射。** 论文自己说全组合搜索（2^95）不可行，靠的是相关性、SHAP 与专家判断的混合迭代，过程未公开。

结论：特征词表可复制，具体子集不可复制。

### 0.3 超参完全没有

第 6.2 节只说用 Optuna 按验证集 MAE 调了学习率、树数、树深与正则强度。**调出的值全文和四个附录（A 特征表、B 特征构造、C 补充图、D 评价洞见）都没有**，Optuna 搜索空间与试验次数也没有。

### 0.4 论文自身的一处算术不一致

第 6.1 节说三部分「共 6457 条」，但 4,497 + 976 + 960 = **6,433**。往回推 `4497 + 968 + 968 = 6433` 是自洽的（后从测试挪 8 点到验证，使测试 960 = 20×48），所以 6457 是笔误。实际文件 6,456 行，中间 23 行的去向未交代，大概率是滞后特征构造丢的头部行。**这会让划分边界差约一天，必须在结果里记为已知偏差，不要为了对齐而反推。**

### 0.5 由此定级：部分复现，两个自由参数

数据、划分、指标、种子四项可精确对齐；特征子集与超参必须自选。所以 **R2 是「部分复现」，不是严格复现**：我们与 17.58 / 55.74 的差距无法区分是实现问题还是特征/超参选择问题。这句限制必须写进结果文件。

---

# 任务 A：锚定表

## A1. 表的结构

关键设计是 `verification_type` 列——它让「不是所有格子都是同一种证据」写在表里，而不是靠读者体会。

| 列 | 含义 |
|---|---|
| `method` | F01_LEAR / F08_LightGBM / …，含 LEAR 的窗口后缀 |
| `original_task` | 原论文的任务与数据集；无则 `none` |
| `original_target_product` | **原任务预测的是什么产品**，见 A2 的产品差异说明 |
| `paper_reported` / `local_on_original_task` / `deviation_percent` | 原论文报告值、本仓库在原任务上的值、偏差 |
| `verification_type` | 五个允许值之一，**全表最重要的一列** |
| `dk1_day_ahead_mae` / `dk1_capacity_mae` | DK1 同协议迁移下的预测误差 |
| `dk1_threshold_arm_eur` / `dk1_free_arm_eur` | 冻结阈值臂与自由备用臂的结算收益 |
| `note` | 不通过原因、产品差异、自由参数等 |

`verification_type` 只允许这五个值：

| 值 | 含义 | 谁能用 |
|---|---|---|
| `point_level_reproduction_passed` | 与作者预测序列逐点比对通过 | F01 窗口 1092、1456 |
| `point_level_reproduction_failed` | 逐点比对不通过，原因写进 `note` | F01 窗口 56、84、Ensemble |
| `partial_reproduction_free_hyperparameters` | 数据/划分/指标/种子对齐，特征子集与超参自选 | F08，任务 B 完成后 |
| `no_anchor` | 原论文没有可对照的任务 | F08，任务 B 完成前 |
| `no_prior_work` | 该问题无先例 | FB2＋B3 |

**任务 A 先按 `no_anchor` 出表。** 任务 B 完成后再改成 `partial_reproduction_free_hyperparameters` 并回填三列。

## A2. 两条不能省的口径约束

**一、跨市场禁止比 MAE，必须用 MASE / RMSSE。** F06 的 Table 9 拿自己与西班牙、德国的 aFRR 研究比较时明确说明：数据粒度、市场结构与基线定义都不同，只能用尺度无关指标。因此锚定表里 DK1 的容量价 MAE（F01 12.27、F08 8.81）与芬兰的 17.58 **不得放在同一列**。DK1 列与原任务列必须分栏，且表头注明两者不可横向比较。

**二、产品不同，必须写明。** F06 预测的是 aFRR **能量**（激活边际）价 `Up`/`Down`（EUR/MWh），容量价 `Up_Cap`/`Down_Cap` 在它那里是**特征**；而 DK1 上 F08 的目标是 `afrr_capacity_market.UpPriceEUR`，即**容量**价（EUR/MW）。锚定的作用是「我们的 LightGBM 实现合格吗」，不是「同一个任务」，所以锚点仍成立，但 `original_target_product` 列与 `note` 必须写清差异。

## A3. 表的内容（全部由代码从已提交的 summary 读取，**禁止手抄常数**）

| method | original_task | paper_reported | local | dev% | verification_type |
|---|---|---|---|---|---|
| F01_LEAR_cw1092 | EPF-DE 日前价 | MAE 3.9298 | 3.9306 | 0.020 | `point_level_reproduction_passed` |
| F01_LEAR_cw1456 | EPF-DE 日前价 | MAE 3.9878 | 3.9889 | 0.027 | `point_level_reproduction_passed` |
| F01_LEAR_cw56 | EPF-DE 日前价 | MAE 4.2826 | 4.0694 | 4.977 | `point_level_reproduction_failed` |
| F01_LEAR_cw84 | EPF-DE 日前价 | MAE 4.1796 | 3.9833 | 4.698 | `point_level_reproduction_failed` |
| F01_LEAR_ensemble | EPF-DE 日前价 | MAE 3.6091 | 3.5703 | 1.075 | `point_level_reproduction_failed` |
| F08_LightGBM | `none` | | | | `no_anchor` |
| FB2_B3 | `none` | | | | `no_prior_work` |

取值来源：

- 前五行来自 `p1_paper/results/r1_lear_de/summary.json` 的 `methods.<window>`：`target_mae`、`local.mae`、`mae_bias_percent`、`accepted`、`not_reproducible_reason`。**`verification_type` 由 `accepted` 决定，不要另写判断逻辑**，`note` 填 `not_reproducible_reason`。
- DK1 列：F01 取 `f01_lear_dk1/summary.json`，F08 取 `f08_lightgbm_dk1/summary.json` 的 `prediction_metrics_211_common_days` 与 `settlement`；自由臂收益取 `s2c_decision_layer/summary.json` 的 `arms.B2_F01` / `arms.B2_F08`。
- F08 的 `original_task` 填 `none`，理由取 `paper_data_alignment.json` 里 F08 的 `original_data_status`（`method_paper_no_electricity_dataset`），写进 `note`。
- `FB2_B3` 行现在只有 `verification_type`，其余留空。**这一行必须存在**——空格子就是贡献声明，也是软肋，不能因为「还没做」而不列。

## A4. 文件

| 文件 | 内容 | 新建 |
|---|---|---|
| `experiments/anchor_table.py` | 读五个 summary，生成表 | 是 |
| `configs/anchor_table.yaml` | 输入输出路径、`verification_type` 允许值、随机种子 | 是 |
| `p1_paper/results/anchor_table/anchor_table.csv` | 表本身 | 是 |
| `p1_paper/results/anchor_table/summary.json` | 环境指纹、随机种子、逐行来源文件与字段路径 | 是 |

**不建 `src/anchor_table/` 模块。** 纯聚合约 80 行，拆一层属于 [AGENTS.md](AGENTS.md) 第 3 条禁止的预留抽象层。

`summary.json` 必须逐行记录每个数字取自哪个文件的哪个字段路径（例如 `r1_lear_de/summary.json → methods.1092.local.mae`）。这张表的作用就是可追溯。

## A5. 验收

```bash
cd /public/ZLCODE/electricity-price-forecasting-research && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/anchor_table.py --config configs/anchor_table.yaml
```

- 表有 7 行，`verification_type` 取值全部落在 A1 的五个允许值内，越界抛异常；
- 五个 LEAR 行的 `local` 与 `deviation_percent` 与 `r1_lear_de/summary.json` 逐位一致；
- 恰好 2 行 `point_level_reproduction_passed`、3 行 `point_level_reproduction_failed`；**算出别的数量说明读错了 `accepted`，停下查**；
- F08 行 `verification_type` 为 `no_anchor` 且 `paper_reported` 为空；
- DK1 列与原任务列分栏，表头写明不可横向比较；
- 其他实验重跑后结果目录零差异（`git diff --stat p1_paper/results/` 只应出现 `anchor_table/`）。

## A6. 顺带要改的一处文档

`research-foundation-2026/PAPER_PLAN.md` 第 47 行补一句：F08 无原任务锚点，任务 B 完成后升级为部分复现，并链接锚定表。**一句话，不要展开**，该文件不得超 400 行。

---

# 任务 B：R2 芬兰 aFRR，部分复现

## B1. 可以照抄的协议（来自全文，逐条落实，不要再推断）

| 项 | 值 |
|---|---|
| 数据 | `data/multi_market_energy_reserve/finland_afrr_zenodo_17494556/extracted/extended_data_v2.csv` |
| 缺失值 | 聚合前在 15 分钟序列上用**前 96 个观测的后向滚动均值**填补（式 1）。**不补零**——论文实测零价占比 Down 0.22%、Up 0%，补零会注入经济上不可能的信号 |
| 聚合 | 15 分钟 → 小时。论文记录该聚合把 2024 年 9 月样本的 Up 峰值从 556.10 削到 305.46 EUR/MWh（−45.1%），标准差 94.12 → 63.37 |
| 异常值 | **不剔除**，前后都不做 |
| 目标 | `Up` 与 `Down` **分开两个模型**，48 小时horizon，直接多步（一次出 48 步，不迭代） |
| 划分 | 70/15/15 按时间，训练 4,497 / 验证 976 / 测试 960，测试期 2025-02-05 至 2025-03-16 |
| 指标 | MAE、RMSE、nMAE、MASE、RMSSE 五项全报；MASE/RMSSE 的季节朴素周期 **m = 48** |
| 种子 | **42** |
| 靶子 | LightGBM Down MAE **17.58**、Up MAE **55.74**（见 0.1 完整五指标） |

评价只做**连续测试块**（Table 4 那一套）。论文另有分段 48 小时与滑动窗两套评价，**本次不做**——它们不产生 0.1 的靶子，做了只增加自由度。

## B2. 必须自选的两项，以及怎么处理

| 自由参数 | 处理方式 |
|---|---|
| 特征子集 | 用 Table A.1 的完整词表实现全部特征，**取全集**，不做选择。理由：论文的子集不可复制，取全集是唯一无需再做一次不可复现选择的方案。在结果里记 `feature_selection: full_table_a1_superset` |
| LightGBM 超参 | 用 `configs/f08_lightgbm_dk1.yaml` 的 `model.params` **原样**，不调参。理由：R2 的目的是核验**我们的 F08 实现**，换一套超参就核验不了它。在结果里记 `hyperparameters: inherited_from_f08_dk1_not_tuned` |

**禁止用 Optuna 调参去逼近 17.58。** 那会把「我们的实现合格吗」变成「我们能不能凑出这个数」，两者不是一回事，后者没有信息量。

## B3. 判据：不是「落进区间就算过」

R2 不设通过/失败门槛，只报告并解释。在结果与 commit message 里必须同时给出：

- 五个指标的实测值与论文值并列；
- **MASE 与 RMSSE 是重点**，因为它们对价格水平不敏感，最能说明实现是否合格。论文 LightGBM 是 Down 0.482 / 0.498，Up 0.569 / 0.553，全部低于 1，即优于季节朴素基线。**我们的 MASE 若也明显低于 1，实现即可判定合格，即使 MAE 对不上 17.58。**
- 差距归因只能写到「特征子集与超参不同」这一层，**不得声称实现有误或无误之外的结论**。

## B4. 文件与产物

| 文件 | 内容 |
|---|---|
| `src/r2_lightgbm_fi/features.py` | 按 Table A.1 构造全部特征；滞后、滚动、循环编码、风寒与体感指数按论文脚注 1、2 的公式 |
| `src/r2_lightgbm_fi/data.py` | 读 v2、15 分钟滚动均值填补、聚合到小时、按 70/15/15 切分 |
| `experiments/r2_lightgbm_fi.py` | 运行入口 |
| `configs/r2_lightgbm_fi.yaml` | 上述全部协议参数，含 `paper_reported` 五指标供代码断言比对 |
| `p1_paper/results/r2_lightgbm_fi/` | `summary.json`（环境指纹、种子、五指标对照、两个自由参数的声明）、`forecasts.csv` |

复用 `src/f08_lightgbm_dk1/model.py` 的 `fit_predict_hourly`，**不要另写一份 LightGBM 调用**——共用同一份实现正是 R2 存在的意义。

## B5. 完成后回填锚定表

F08 行：`verification_type` 改 `partial_reproduction_free_hyperparameters`，`original_task` 填芬兰 aFRR 能量价，`original_target_product` 填 `aFRR energy (activation) price`，`paper_reported` 与 `local_on_original_task` 填 MASE（不是 MAE，见 A2），`note` 写明两个自由参数与产品差异。

## B6. 术语

全文写「部分复现」或「独立重实现」。**禁止写「复现了 F06 的结果」**——`PAPER_RESULTS.md` 第 24 行已把「本仓库已复现这些数值」列为该论文的禁止表述，且我们有两个自由参数。

芬兰窗口 2024-06 至 2025-03 与 DK1 的 2025-10 至 2026-08 **无重叠**，只能各自期内比较，不得跨期对比或合并。

---

## 不要做

1. 不要改 `src/epf_harness/`、`src/f01_lear_dk1/`、`src/f08_lightgbm_dk1/`、`src/s2c_decision_layer/` 下任何文件。
2. 不要重跑 R1、F01、F08、S2、S2-C，锚定表只读它们的 summary。
3. 不要手抄数字进 CSV 或文档，全部由代码读取。
4. 不要为逼近 17.58 而调超参或挑特征子集。
5. 不要做分段 48 小时与滑动窗两套评价。
6. 不要把跨市场的 MAE 放进同一列比较，用 MASE/RMSSE。
7. 不要把 `partial_reproduction_free_hyperparameters` 写成复现，不要把 `no_anchor` 说成「暂未复现」——后者暗示做得到。
8. 不要因为 `FB2_B3` 行是空的就删掉它。
9. 不要为对齐 0.4 的 23 行差异而反推划分边界，记为已知偏差即可。
10. 不要 `git add -A` / `git add .`。
11. 不要新建 README、SUMMARY、CHANGELOG、`*_GUIDE.md`、`*_NOTES.md`、`*_REPORT.md`。

## 提交

任务 A 一次提交，commit message 写表建在哪、五个 `verification_type` 各命中几行、F08 无锚点的处理。

任务 B 单独提交，commit message 写五指标对照、MASE 判断、两个自由参数，以及锚定表 F08 行的回填。
