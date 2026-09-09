# 任务 A：锚定表　｜　任务 B：R2（芬兰 aFRR，**当前被阻塞**）

目标：建一张表，对每条外部基线同时回答「这个实现在它原来的任务上对得上吗」和「搬到 DK1 后表现如何」，用来挡住「你把别人的方法做成了稻草人」这一质疑。

基线：`main` = `4fdf86c`。环境固定为 `/public/ZLCODE/.venvs/r1_lear_de/bin/python`。本文件在两个任务都结束后删除。

---

## 0. 先读这一节：R2 不是严格复现，我此前说错了

任务书作者在上一轮把 F06 称为「论文库里唯一剩下的严格复现机会」。核查后**这个判断是错的**，三条硬事实：

| 事实 | 依据 |
|---|---|
| F06 全文从未取得 | `paper/.../F06_SOURCE_...md` 明写「出版社全文未能由自动采集器取得」；`download_manifest.json` 里没有 F06 条目 |
| Zenodo 数据集**不含作者预测序列** | `extracted/extended_data_v{1,2}.csv` 的列只有 `datetime, Up, Down, sp, 气象七项, Down_Cap, Up_Cap, is_public_holiday, 消费三项`。没有任何模型输出列 |
| 作者报告的是**区间**不是点值 | 下调 MAE 约 17.5–17.8，上调约 55.7–60.1，且 `PAPER_RESULTS.md` 第 24 行已把「本仓库已复现这些数值」列为**禁止表述** |

R1 之所以成立，是因为 EPF-DE 同时公开了输入数据**和**作者的预测时间序列（`Forecasts_DE_DNN_LEAR_ensembles.csv`），才能做逐点比对（中位点差 0.011，99 分位 0.106）。芬兰数据集没有这一半，**逐点比对不可能**。

再加上没有全文，就不知道：用的是 v1（3,936 行）还是 v2（6,456 行）、训练/测试怎么划、MAE 的口径、特征集、超参。**光是划分口径不同就能让 MAE 移动超过 17.5–17.8 这个区间的宽度**，所以在拿到全文之前，任何数字都不可比。

结论：**任务 A 现在就做；任务 B 被阻塞，见第 3 节的闸门。**

---

# 任务 A：锚定表

## A1. 表的结构

关键设计是 `verification_type` 列——它让「不是所有格子都是同一种证据」这件事写在表里，而不是靠读者自己体会。

| 列 | 含义 |
|---|---|
| `method` | F01_LEAR / F08_LightGBM / …，含 LEAR 的窗口后缀 |
| `original_task` | 原论文的任务与数据集；无则填 `none` |
| `paper_reported` | 原论文报告值；无则空 |
| `local_on_original_task` | 本仓库在原任务上跑出的值 |
| `deviation_percent` | 两者偏差 |
| `verification_type` | 见下表，**这是全表最重要的一列** |
| `dk1_day_ahead_mae` / `dk1_capacity_mae` | DK1 同协议迁移下的预测误差 |
| `dk1_threshold_arm_eur` / `dk1_free_arm_eur` | 冻结阈值臂与自由备用臂的结算收益 |

`verification_type` 只允许这五个值：

| 值 | 含义 | 谁能用 |
|---|---|---|
| `point_level_reproduction_passed` | 与作者预测序列逐点比对通过 | F01 窗口 1092、1456 |
| `point_level_reproduction_failed` | 逐点比对不通过，原因须在 `note` 列写明 | F01 窗口 56、84、Ensemble |
| `range_consistency_only` | 只能与论文报告的区间比对，无逐点证据 | 任务 B 若解锁 |
| `no_anchor` | 原论文没有可对照的任务 | F08 |
| `no_prior_work` | 该问题无先例 | FB2＋B3 |

## A2. 表的内容（全部取自已提交的结果文件，**禁止手抄常数**）

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

- 前五行全部来自 `p1_paper/results/r1_lear_de/summary.json` 的 `methods.<window>` —— `target_mae`、`local.mae`、`mae_bias_percent`、`accepted`、`not_reproducible_reason`。**`verification_type` 由 `accepted` 决定，不要另写判断逻辑**，`note` 列填 `not_reproducible_reason`。
- DK1 列：F01 取 `p1_paper/results/f01_lear_dk1/summary.json`，F08 取 `p1_paper/results/f08_lightgbm_dk1/summary.json` 的 `prediction_metrics_211_common_days` 与 `settlement`；自由臂收益取 `p1_paper/results/s2c_decision_layer/summary.json` 的 `arms.B2_F01` / `arms.B2_F08`。
- F08 的 `original_task` 填 `none`，理由取 `paper_data_alignment.json` 里 F08 的 `original_data_status`（`method_paper_no_electricity_dataset`），写进 `note` 列。
- `FB2_B3` 行现在只有 `verification_type`，其余留空。**这一行必须存在**——空格子就是贡献声明，也是软肋，不能因为「还没做」而不列。

## A3. 文件

| 文件 | 内容 | 新建 |
|---|---|---|
| `experiments/anchor_table.py` | 读五个 summary，生成表 | 是 |
| `configs/anchor_table.yaml` | 五个输入路径、输出路径、`verification_type` 允许值、随机种子 | 是 |
| `p1_paper/results/anchor_table/anchor_table.csv` | 表本身 | 是 |
| `p1_paper/results/anchor_table/summary.json` | 环境指纹、随机种子、各行来源文件与字段路径 | 是 |

**不建 `src/anchor_table/` 模块。** 这是纯聚合，逻辑约 80 行，拆一层包属于 [AGENTS.md](AGENTS.md) 第 3 条禁止的「为将来可能需要而写的抽象层」。其他任务建 `src/` 子包是因为那里有真实的模型逻辑，这里没有。

`summary.json` 里必须逐行记录每个数字取自哪个文件的哪个字段路径（例如 `r1_lear_de/summary.json → methods.1092.local.mae`）。这张表的作用就是可追溯，来源不写清楚它就没有意义。

## A4. 验收

```bash
cd /public/ZLCODE/electricity-price-forecasting-research && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/anchor_table.py --config configs/anchor_table.yaml
```

通过条件：

- 表有 7 行，`verification_type` 的取值全部落在 A1 的五个允许值内，越界直接抛异常；
- 五个 LEAR 行的 `local` 与 `deviation_percent` 与 `r1_lear_de/summary.json` 逐位一致；
- 恰好 2 行是 `point_level_reproduction_passed`，3 行是 `point_level_reproduction_failed`；**若代码算出别的数量，说明读错了 `accepted` 字段，停下查**；
- F08 行的 `verification_type` 是 `no_anchor` 且 `paper_reported` 为空；
- 其他四个实验重跑后结果目录零差异（`git diff --stat p1_paper/results/` 只应出现 `anchor_table/`）。

## A5. 顺带要改的一处文档

`research-foundation-2026/PAPER_PLAN.md` 第 47 行现在写「尚需第三条外部基线，且至少一条外部基线需覆盖概率预测」。补一句：F08 无原任务锚点是已知且暂时无法消除的缺口，理由是其原论文没有电力数据集，并链接到锚定表。**一句话，不要展开**，该文件不得因此超 400 行。

---

# 任务 B：R2 芬兰 aFRR —— 当前被阻塞

## B1. 闸门：先拿到 F06 全文，否则不要开工

`Energy and AI` 24 (2026) 100724，DOI `10.1016/j.egyai.2026.100724`。自动采集器取不到，需要人工通过机构权限获取。

拿到后必须能从全文里抽出这五项，**缺任何一项 R2 都不能声称是复现**：

1. 用的是 `extended_data_v1.csv`（3,936 行）还是 `v2.csv`（6,456 行）；
2. 训练/验证/测试的时间划分；
3. MAE 的口径：对 `Up` 和 `Down` 分别算还是合算，单位，是否剔除任何时段；
4. 特征集：气象七项与 `electricity_consumption_forecast` 用了哪些，滞后阶数；
5. LightGBM 的超参与随机种子。

**这五项不齐就停下报告，不要用默认值猜。** 划分口径不同造成的 MAE 位移会超过 17.5–17.8 这个区间的宽度，猜出来的数字对不上不能说明实现有问题，对上了也不能说明实现正确——两个方向都无信息量。

## B2. 闸门通过后做什么

目标降级为**区间一致性核验**，不是复现：

- 用与 F08 完全相同的 LightGBM 实现（`src/f08_lightgbm_dk1/model.py` 的 `fit_predict_hourly`），按全文抽出的划分与特征集，在芬兰数据上对 `Up` 与 `Down` 各出一组 MAE；
- 判据：落在作者报告的 17.5–17.8（下调）与 55.7–60.1（上调）内算通过；
- 锚定表里 F08 行的 `verification_type` 从 `no_anchor` 改为 `range_consistency_only`，`original_task` 改为芬兰 aFRR，并在 `note` 写明这是区间核验、非逐点复现；
- 结果写 `p1_paper/results/r2_lightgbm_fi/`，含环境指纹与随机种子。

**术语**：全文写「区间一致性核验」。禁止写「复现 F06 的结果」——`PAPER_RESULTS.md` 第 24 行已把「本仓库已复现这些数值」列为该论文的禁止表述。

窗口 2024-06 至 2025-03 与 DK1 的 2025-10 至 2026-08 **无重叠**，只能各自期内比较，不得跨期对比或合并（[EXECUTION_PLAN.md](research-foundation-2026/EXECUTION_PLAN.md) 第 11 节已记录该约束）。

## B3. 闸门取不到怎么办

取不到全文是完全可能的结果，那就**取消 R2**，不要用猜的划分硬跑一版充数。届时：

- 锚定表 F08 行永久保持 `no_anchor`；
- 在 `PAPER_PLAN.md` 的缺口表里把它写成已知限制：F08 是算法来源论文，没有电力任务可锚定，其在 DK1 上的可信度只能靠与 F01 共用同一未缩放设计矩阵这一条公平性约束支撑；
- 这是一个诚实的限制陈述，不是失败。审稿人接受「我们说明了做不到什么」，不接受「我们假装做到了」。

---

## 不要做

1. 不要改 `src/epf_harness/`、`src/f01_lear_dk1/`、`src/f08_lightgbm_dk1/`、`src/s2c_decision_layer/` 下任何文件。
2. 不要重跑 R1、F01、F08、S2、S2-C，锚定表只读它们的 summary。
3. 不要手抄任何数字进 CSV 或文档，全部由代码从 summary 读取。
4. 不要在 B1 闸门未通过时开始写 R2 的代码。
5. 不要把 `range_consistency_only` 写成复现，不要把 `no_anchor` 说成「暂未复现」——后者暗示做得到，实际做不到。
6. 不要因为 `FB2_B3` 行是空的就把它删掉。
7. 不要 `git add -A` / `git add .`。
8. 不要新建 README、SUMMARY、CHANGELOG、`*_GUIDE.md`、`*_NOTES.md`、`*_REPORT.md`。

## 提交

任务 A 一次提交。commit message 写：表建在哪、`verification_type` 的五个取值各命中几行、以及 F08 无锚点这一缺口的处理方式。

任务 B 的闸门结果单独报告，不要和任务 A 混在一个提交里。
