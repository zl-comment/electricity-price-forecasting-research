# 任务 R2-T：给 R2 补一个验证集调参变体

目标：让 R2 真正锚定 F08 的实现能力。当前运行的模型退化成近似常数预测器，锚不住任何东西；补一个按原论文协议在**验证集**上调参的变体，两个变体并列报告，差值即「超参」这一自由参数的代价。

基线：`main` = `60a3d62`。环境固定为 `/public/ZLCODE/.venvs/r1_lear_de/bin/python`。本文件在任务完成并合并后删除。

---

## 0. 为什么要补

### 0.1 当前运行是退化的

`p1_paper/results/r2_lightgbm_fi/summary.json` 的 `degenerate_fit_audit`：

| 目标 | 常数预测器 MAE<sup>*</sup> | 模型 MAE | 模型相对常数改善 | 预测 std / 实际 std |
|---|---:|---:|---:|---:|
| Down | 25.076 | 23.625 | **5.8%** | 0.377 |
| Up | 58.540 | 56.427 | **3.6%** | 0.279 |

<sup>*</sup> 常数 ＝ 训练集中位数，即 L1 目标的退化解。

**原论文 Up MAE 是 55.74，而常数预测器就能拿到 58.540。** 所以「本地 56.427 与原论文仅差 1.2%」是目标高波动下的巧合，不构成实现正确的证据。这次运行没有锚定任何东西。

### 0.2 根源是上一份任务书写错了

上一份任务书写「超参原样沿用 F08，不许调参」，把两件不同的事混为一谈：

- **在测试集上调参去逼近 17.58** → 作弊，必须禁止；
- **在验证集上调参** → **这就是原论文自己的协议**。F06 第 6.2 节原文：用 Optuna「minimizing the validation MAE on the 70/15 split」。

复刻它的调参**流程**是忠实复现，不是作弊。禁令把后者也堵死，直接造成退化。这是任务书的错，不是执行的错。

---

## 1. 前置：先修掉 20 倍冗余拟合

`experiments/r2_lightgbm_fi.py` 的 `forecast_target` 在 20 个测试原点的循环里反复调用 `fit_predict_hourly`，但每次传入的 `Xtrain, Ytrain` **完全相同**：

```python
for origin in test_origins(frame, splits["test"], horizon):
    prediction = fit_predict_hourly(Xtrain, Ytrain, features.iloc[[origin]]..., ...)
```

即 48 个模型被重复拟合了 20 遍，每个目标 960 次拟合而非 48 次。结果没错（种子确定），但白烧 20 倍算力，而调参需要跑几十次完整训练，不修就没法做。

**改法**：把「拟合」与「预测」拆开——先按当前参数拟合一次得到 48 个 booster，再对 20 个原点各预测一次。

- **不要修改 `src/f08_lightgbm_dk1/model.py`。** 那是 DK1 任务共用的实现，改它会影响已冻结的 F08 结果。在 `src/r2_lightgbm_fi/` 下新增一个薄封装，内部逐 horizon 调 `lgb.train` 与 `booster.predict`，**参数、种子、`num_threads` 与 `fit_predict_hourly` 逐字相同**。
- 在结果里把 `shared_f08_fit_calls_per_target` 改为记录实际拟合次数（应为 48），并新增 `predict_calls_per_target: 20`。

**硬校验：修完后重跑，`forecasts.csv` 必须与已提交版本逐字节相同，五个指标逐位相同。** 不相同说明封装与 `fit_predict_hourly` 不等价，停下查，不要调容差。

---

## 2. 依赖：不装 Optuna，用带种子的随机搜索

Optuna 5.0.0 支持 python ≥ 3.9，但需要额外装 `colorlog`、`alembic`、`sqlalchemy`、`tqdm` 四个包（`PyYAML`、`packaging`、`numpy` 已有）。往锁定的科学计算环境里塞四个包，只为跑一个四维搜索，代价不划算。

**更要紧的是：原论文没有公布搜索空间、试验次数和采样器种子**，所以无论用 TPE 还是随机搜索，这三项都是我们自选的自由参数，换算法并不会让结果更接近原文。真正需要忠实的是「**在验证集 MAE 上、对论文点名的那四类参数调参**」这个协议，带种子的随机搜索完全满足。

**结论：用 `numpy.random.RandomState` 实现随机搜索，不新增任何依赖。** 在结果里记 `tuner: seeded_random_search`，并写明未使用 Optuna 及其理由。

---

## 3. 调参协议

### 3.1 搜索空间：只覆盖论文点名的四类

原论文第 6.2 节原话：tuned「learning rate, number of trees, tree depth, and regularization strength」。**只调这四类，不要自行扩充**——多调一项就多一个论文没有的自由度。

| 参数 | 对应论文措辞 | 建议范围 | 采样 |
|---|---|---|---|
| `learning_rate` | learning rate | [0.01, 0.30] | 对数均匀 |
| `num_iterations` | number of trees | [100, 2000] | 整数对数均匀 |
| `num_leaves` | tree depth | [15, 255] | 整数对数均匀 |
| `max_depth` | tree depth | {−1, 4, 6, 8, 12} | 均匀 |
| `lambda_l1` | regularization strength | [1e-3, 10] 及 0 | 对数均匀，含 0 |
| `lambda_l2` | regularization strength | [1e-3, 10] 及 0 | 对数均匀，含 0 |
| `min_data_in_leaf` | — | **不调**，固定 20 | — |

其余参数（`objective: regression_l1`、`metric: l1`、`deterministic: true`、`force_col_wise: true`、`num_threads: 1`、三个子种子）**全部保持 `configs/f08_lightgbm_dk1.yaml` 的取值不变**。

全部范围写进 `configs/r2_lightgbm_fi.yaml` 的新块 `tuning:`，代码里不许有字面量。

### 3.2 目标函数

- **验证集 MAE**，与原论文一致。
- **Down 与 Up 分开调**，各得一套超参——它们是两个独立模型。
- 验证集用 0.4 节确定的划分中段：训练 4,497 / **验证 976** / 测试 960。
- 验证集上的预测同样按 48 小时直接多步、原点不重叠地铺满；976 不能被 48 整除，**取能整除的最大前缀（20 × 48 ＝ 960），余下 16 小时丢弃并记录**，不要为凑整改动划分。

### 3.3 测试集纪律

- **调参全程只允许读训练段与验证段。** 测试段在最终一次评估前不得以任何形式进入代码路径。
- 选出最优超参后，**用训练段重新拟合一次**，再对测试段做且仅做**一次**评估。
- **禁止看到测试结果后回头改搜索空间、加试验次数或换种子重跑。** 一次定稿。若结果不理想，如实报告，那也是结论。

### 3.4 预算与种子

- 试验次数 `n_trials: 60`（每个目标），写进 config。
- 搜索种子 `tuning_seed: 42`，与原论文的确定性运行种子一致。
- 修完第 1 节后单次训练约 20 秒量级，60 × 2 个目标预计 30–60 分钟，可接受。若实测远超，**减少 `n_trials` 并在结果里如实记录，不要中途改协议**。

---

## 4. 两个变体都要报

结果目录 `p1_paper/results/r2_lightgbm_fi/` 下 `summary.json` 顶层新增 `variants`，含两个键：

| 变体 | 超参来源 | 作用 |
|---|---|---|
| `inherited` | `configs/f08_lightgbm_dk1.yaml` 原值，不调 | 消融，量化「不调参」的代价 |
| `tuned` | 3.1 的搜索空间在验证集上选出 | 锚点，表 6 与表 7 以它为准 |

两个变体都必须给出：

- 五个指标（MAE、RMSE、nMAE、MASE、RMSSE）× 两个目标，与原论文并列；
- **`degenerate_fit_audit` 全套**（常数预测器 MAE、相对常数改善、预测/实际标准差比、均值偏差）——调参后若仍接近常数预测器，那才是关键结论，必须能看出来；
- 选中的超参全文（`tuned` 变体）与验证集 MAE。

新增顶层字段 `hyperparameter_cost_eur_free`：`tuned` 与 `inherited` 在每个目标上的 MAE 差，即「超参」这一自由参数的代价。

`forecasts.csv` 增加一列 `variant`，两个变体的预测都写进去。

---

## 5. 回填表 6 与表 7

任务完成后，锚定表 `p1_paper/results/anchor_table/anchor_table.csv` 的 F08 行按 `tuned` 变体更新：

- `local_on_original_task` 与 `deviation_percent` 改用 `tuned` 的 MASE；
- `verification_type` 保持 `partial_reproduction_free_hyperparameters`（**仍有两个自由参数：特征子集，以及搜索空间/预算/种子**，不要改成更强的类型）；
- `note` 补一句：超参由验证集调参选出，搜索空间与预算由本仓库自定。

`experiments/anchor_table.py` 从 `variants.tuned` 读取，**不要手抄**。

---

## 6. 判据：报告，不设通过门槛

沿用上一份任务书的口径，并加一条：

- **MASE 与 RMSSE 是重点**，它们对价格水平不敏感。原论文 LightGBM 为 Down 0.482 / 0.569、Up 0.498 / 0.553。
- **`degenerate_fit_audit` 是先决条件**：若 `tuned` 变体的「相对常数改善」仍低于 10%，则无论 MAE 多接近 17.58，都**不能**声称锚定成功，须在结果与 commit message 中写明「调参后仍退化，F08 实现能力未获验证」。
- 差距归因只能写到「特征子集不同」「搜索空间与预算不同」这一层，不得推断实现有误或无误之外的结论。

原论文尺度指标的分母存疑（见已提交的 `paper_metric_consistency_audit`），**该审计保持原样，不得据此调整本仓库任何数值**。

---

## 7. 不要做

1. 不要修改 `src/f08_lightgbm_dk1/model.py`、`src/f01_lear_dk1/`、`src/epf_harness/`、`src/s2c_decision_layer/`。
2. 不要安装 Optuna 或任何新依赖。
3. 不要在测试集上调参，不要在看到测试结果后回头改搜索空间、预算或种子。
4. 不要扩充 3.1 之外的搜索维度。
5. 不要为凑整而改动 70/15/15 划分或 6,457 点小时网格。
6. 不要删除 `inherited` 变体——没有它就量化不了超参的代价。
7. 不要把 `verification_type` 升级成比 `partial_reproduction_free_hyperparameters` 更强的类型。
8. 不要重跑或修改 R1、F01、F08、S2、S2-C 的结果。
9. 不要手抄数字进 CSV 或文档。
10. 不要 `git add -A` / `git add .`。
11. 不要新建 README、SUMMARY、CHANGELOG、`*_GUIDE.md`、`*_NOTES.md`、`*_REPORT.md`。
12. **术语**：全程写「部分复现」，禁止写「复现了 F06 的结果」。

---

## 8. 验收

**第 1 步 冗余拟合修复不改变结果**
```bash
cd /public/ZLCODE/electricity-price-forecasting-research && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/r2_lightgbm_fi.py --config configs/r2_lightgbm_fi.yaml
```
先只做第 1 节的改动，此时 config 尚无 `tuning:` 块。通过条件：`git diff` 对 `p1_paper/results/r2_lightgbm_fi/forecasts.csv` **为空**，`summary.json` 只应出现 `shared_f08_fit_calls_per_target` 由 20 变 48 与新增 `predict_calls_per_target` 两处差异。单次运行耗时应从约 12 分钟降到 1 分钟量级。

**第 2 步 调参变体**

同一命令，config 加上 `tuning:` 块后重跑。通过条件：

- `variants` 含 `inherited` 与 `tuned` 两个键，两者五指标 × 两目标齐全；
- `inherited` 的五指标与第 1 步逐位相同；
- `tuned` 的超参全部落在 3.1 声明的范围内；
- 两个变体的 `degenerate_fit_audit` 齐全；
- `hyperparameter_cost_eur_free` 已计算；
- 结果里记录了 `tuner: seeded_random_search`、`n_trials`、`tuning_seed`、验证集丢弃的 16 小时。

**第 3 步 回归**
```bash
cd /public/ZLCODE/electricity-price-forecasting-research && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/anchor_table.py --config configs/anchor_table.yaml
```
锚定表仍 7 行 × 13 列，计数仍为 2 passed / 3 failed / 1 partial / 1 no_prior_work，F08 行按第 5 节更新。`git diff --stat p1_paper/results/` 只应出现 `r2_lightgbm_fi/` 与 `anchor_table/`。

**第 4 步 交付判断**

给出这张表，三种结论按实际填，不要预设：

| 变体 | Down MAE | Up MAE | Down MASE | Up MASE | 相对常数改善 (Down / Up) |
|---|---|---|---|---|---|
| 原论文 | 17.58 | 55.74 | 0.482 | 0.498 | — |
| `inherited` | 23.625 | 56.427 | 1.085 | 0.904 | 5.8% / 3.6% |
| `tuned` | 待测 | 待测 | 待测 | 待测 | 待测 |

| 情形 | 结论 |
|---|---|
| `tuned` 的 MASE 明显低于 1 且相对常数改善远高于 10% | 实现合格，F08 锚定成立，表 6/7 按此改写 |
| MASE 改善但相对常数改善仍低 | 仍退化，锚定不成立，如实写明并保留 `no_anchor` 级别的表述 |
| 与 `inherited` 差别不大 | 说明退化不是超参造成的，需另查特征构造或标签对齐；停下报告，不要继续调 |

**第 5 步 提交**

逐路径 `git add`，一次提交。commit message 写：冗余拟合修复前后结果是否逐字节一致、`tuned` 的五指标与相对常数改善、落在上表哪一种情形、以及锚定表 F08 行的最终取值。
