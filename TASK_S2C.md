# 任务 S2-C：先拆决策层缺口，再决定 FB 侧规模

目标：把 `B1 → Oracle` 的 391,126 EUR 缺口中占 86.6% 的决策层部分拆成「备用联合优化」与「激活完美信息」两块，判断其中多少是可部署信息拿得到的，据此决定 FB0–FB3 的建设规模；同时把已经产生但没有任何文档记录的 RQ4 证据落到纸面。

基线：`main` = `bdce144`。环境固定为 `/public/ZLCODE/.venvs/r1_lear_de/bin/python`。本文件在 S2-C 全部完成并合并后删除。

---

## 0. 两个已经拿到但没写下来的结果

### 0.1 缺口分解：86.6% 在决策层

211 个共同交割日，数据见 [`arm_comparison.csv`](p1_paper/results/f08_lightgbm_dk1/arm_comparison.csv)：

| 缺口来源 | EUR | 占 391,126 |
|---|---:|---:|
| 价格预测误差 | 36,546 | 9.3% |
| 时间分辨率 | 15,835 | 4.0% |
| 决策层 | 338,745 | **86.6%** |

F01 已吃掉预测误差那 9.3% 中的 77.2%，只剩 8,341 EUR。继续在预测侧堆模型最多再拿几千欧元，所以原顺序「先做 FB0，再 FB1、FB2＋B3」推迟。

**Oracle 同时拥有完美价格、自由备用优化、完美激活量三样东西，其中第三样永远不可部署。** 若 338,745 主要来自第三样，论文必须改用可部署上界报 regret，而不是 Oracle。这个结论现在拿，不要等 S3 做完被审稿人指出。

### 0.2 RQ4 的一个干净反例（**当前无任何文档或结果文件记录，本次必须补上**）

211 个共同交割日上：

| 方法 | 日前价 MAE | 日前价 RMSE | 容量价 MAE | 容量价 RMSE | 结算收益 EUR |
|---|---:|---:|---:|---:|---:|
| 持续法（B1） | 28.875 | 43.73 | 9.080 | 25.19 | 433,618 |
| F01 LEAR | 25.458 | 49.77 | 12.272 | 30.88 | **461,823** |
| F08 LightGBM | **24.842** | **37.82** | **8.812** | **23.12** | 445,055 |

**F08 在两个目标、两个指标上四项全胜，收益却比 F01 少 16,768 EUR。** 按 MAE 或 RMSE 选模型会选 F08，然后少赚 3.6%。

这是 [RESEARCH_FRAMEWORK.md](research-foundation-2026/RESEARCH_FRAMEWORK.md) 第 90 行 RQ4 可证伪主张「预测精度排名与决策收益排名不一致」的直接正面证据，不需要任何消融。数字已在两个 `summary.json` 里，但 `grep -rn "RQ4" research-foundation-2026/` 只命中问题定义本身，没有一处引用这个证据。

---

## 1. 臂的定义

四个臂，除信息集外全部共用同一 arm（4 MW / 8 MWh）、同一 storage、同一 solver 容差、同一 211 天。

| 臂 | 计划时相信的价格 | 备用承诺 | 激活信息 | 状态 |
|---|---|---|---|---|
| `perfect_quarter_price` | 实测 15 分钟 | 冻结阈值 6.69 | 无 | 已有 = 485,999 |
| **`B2_F01`** | F01 预测 | **自由优化** | 无 | 新增，**可部署** |
| **`B2_F08`** | F08 预测 | **自由优化** | 无 | 新增，**可部署** |
| **`oracle_price_reserve`** | 实测 15 分钟 | **自由优化** | 无 | 新增，不可部署 |
| `Oracle` | 实测 15 分钟 | 自由优化 | 完美 | 已有 = 824,744 |

拆解口径：

- `oracle_price_reserve − perfect_quarter_price` = **备用联合优化的价值**（方向可部署）
- `Oracle − oracle_price_reserve` = **激活完美信息的价值**（不可部署，只能作上界）
- `B2_F01 − 461,823` = 同一信息集下「阈值规则 → 自由优化」实际拿到多少
- `B2_F01 / oracle_price_reserve` = 真实预测能吃掉备用优化空间的比例
- **`B2_F01` 与 `B2_F08` 的排序**——若自由优化后 F08 反超 F01，说明 0.2 的反例是阈值规则造成的；若仍不反超，说明精度—收益错位比阈值规则更深。**两种结果都要照实报，不要预期任何一种。**

**编号**：`B2` 已在 [CONVENTIONS.md](CONVENTIONS.md) 第 2 节的 `B0`–`B3`、`Oracle` 体系内登记（点预测＋联合优化，消融臂），直接用，不新增前缀。`oracle_price_reserve` 与已有的 `perfect_quarter_price` 一样按具名 Oracle 变体处理，**不引入新编号前缀**。

---

## 2. S2-C0：文档同步（先做，分钟级）

### 2.1 三处陈旧的「下一步」

都还写着「先纳入/先复现 F01 LEAR」，且其中一处用了术语红线词「复现」——按 [CONVENTIONS.md](CONVENTIONS.md) 第 1 节，DK1 上只能写「同协议迁移评测」。

**`research-foundation-2026/PAPER_PLAN.md` 第 47 行**

> | **无任何外部基线** | 仅与自制对照臂比较，RQ1 站不住；FB0 是本项目自己的代码 | 至少纳入一条预测侧已发表方法。成本最低的是 F01 LEAR——R1 已逐点复现该实现 |

改为反映已完成状态：F01 与 F08 已作为外部基线纳入同协议评测，两者受 DK1 数据长度约束（小时级、纯价格滞后、n/p = 109/103 = 1.06）；剩余缺口改为「尚无第三条外部基线，且两条均非概率预测」。

**`research-foundation-2026/EXECUTION_PLAN.md` 第 377 行**

> 决策侧的激活闭环、恢复成本与条件 Oracle 已固定。接下来先复现 F01 LEAR 预测外部基线，再建立 FB0 独立点、FB1 多任务点、FB2 联合概率和 FB3 决策导向候选；全部限定在 07:30 信息集上，共用第 8.1 节的测试集与本节的结算层。

改为：F01/F08 已纳入；接下来先做 S2-C1/C2 拆决策层缺口，再按结果决定 FB0–FB3 规模。**删掉「复现」二字。**

**`research-foundation-2026/EXECUTION_PLAN.md` 第 385 行（第 11 节第 5 条）**

> 5. **下一步**：先纳入 F01 LEAR 外部预测基线，再建立 FB0–FB3，全部限定在 07:30 信息集上；

第 5 条标为已完成，新的下一步是 S2-C1/C2。第 6 条已写「在相同激活闭环与恢复层上实现 B2/B3」，与 S2-C1 重合，**合并为一条**，不要留两处互相矛盾的下一步。

第 307 行与第 381 行提到 R1「逐点复现」是**正确的**（R1 在 EPF-DE 上确实是复现，且已注明只有 1092/1456 窗通过），不要改。

### 2.2 把 0.2 的 RQ4 反例落到纸面

三处，缺一不可：

1. **`p1_paper/results/f08_lightgbm_dk1/summary.json`** —— 新增 `forecast_versus_settlement` 段，记录三条方法在 211 天上的四个预测指标与结算收益，以及 `f08_dominates_f01_on_all_forecast_metrics`（布尔）与 `f01_minus_f08_eur`。由 `experiments/f08_lightgbm_dk1.py` 生成，值从已读入的 F01 summary 与本地指标取，**不要手写常数**。
2. **`p1_paper/results/f08_lightgbm_dk1/arm_comparison.csv`** —— 已有的 `arm` 行补两列 `day_ahead_mae`、`capacity_mae`，可填的填，诊断臂与 Oracle 留空。让「精度更好、收益更低」在同一张表里可见。
3. **`research-foundation-2026/EXECUTION_PLAN.md` 6.3 节** —— 在缺口分解那段结论后加一句，指出该反例的存在与它支持 RQ4，并链接到 `arm_comparison.csv`。**控制在两句以内**，且改完全文件必须仍 < 400 行（现 391 行，[AGENTS.md](AGENTS.md) 第 2 条上限）。

不要为此新建 md 文件。不要在 `RESEARCH_FRAMEWORK.md` 里改 RQ4 的定义——问题和主张是冻结的，只补证据。

---

## 3. S2-C1 与 S2-C2：实现

两个任务共用一份代码，一次跑出四个新臂结果（`B2_F01`、`B2_F08`、`oracle_price_reserve`，外加复算 `perfect_quarter_price` 做一致性校验）。

### 3.1 关键实现事实（已核实，不要重新探查）

`src/epf_harness/dk1_storage.py:83` 的 `solve_day(prices, capacity_prices, procured_mw, hour_of_slot, arm, storage, solver, r_up_fixed)`：

- **传 `r_up_fixed=None` 即得自由备用优化。** `day_bounds`（第 71–80 行）在该分支下把 `r_up` 的上界设为 `min(P_max, market_share_cap · procured_mw)`，下界 0。目标函数第 94 行已含 `-capacity_prices`，第 60–61 行的 `-soc + hours · reserve_delivery_hours / eta_d <= soc_0` 已做能量预留。
- **因此 `src/epf_harness/dk1_storage.py` 一行都不要改。**
- `settle_day` 在 `plan` 不含 `p_recovery` 时自动走 `solve_recovery`，与 B0/B1 路径相同。
- 自由备用臂**不使用**冻结阈值 6.69。阈值只属于 B1 与 `perfect_quarter_price`。

### 3.2 文件

| 文件 | 内容 | 新建 |
|---|---|---|
| `src/s2c_decision_layer/__init__.py` | 一行 docstring | 是 |
| `src/s2c_decision_layer/arms.py` | 四个臂的构造与求解，全部复用 `epf_harness.dk1_storage` | 是 |
| `configs/s2c_decision_layer.yaml` | `data`/`window`/`split`/`storage`/`solver`/`settlement`/`datasets` 块**逐字**抄自 `configs/s2_dk1_baselines.yaml`，不改任何数值 | 是 |
| `experiments/s2c_decision_layer.py` | 运行入口，结构照抄 `experiments/f01_lear_dk1.py` | 是 |
| `experiments/f08_lightgbm_dk1.py` | 改现有：`arm_comparison_table` 加入三个新臂、拆分决策层行、补两列 MAE；新增 `forecast_versus_settlement` | 否 |

面板构造、211 天与阈值的重建**直接 import** `f01_lear_dk1.settlement` 的 `build_panels`、`settlement_protocol`、`resize`，不要重写——那是保证与 F01/F08 同口径的唯一可靠办法。预测值从 `p1_paper/results/f01_lear_dk1/forecasts.csv` 与 `p1_paper/results/f08_lightgbm_dk1/forecasts.csv` 读入，**不重跑预测**。

### 3.3 结果产物

```
p1_paper/results/s2c_decision_layer/
  summary.json        环境指纹、随机种子、四臂收益、两块拆解、与 B1/F01/F08/Oracle 的对照
  daily_results.csv   逐日逐臂，列与 p1_paper/results/f01_lear_dk1/daily_results.csv 对齐
```

`arm_comparison.csv` 的 `headroom_component` 行把 `decision_layer` 拆成 `reserve_cooptimisation` 与 `activation_foresight` 两行，值从 `s2c_decision_layer/summary.json` 读入。

---

## 4. 不要做

1. 不要改 `src/epf_harness/` 下任何文件，不要改 `configs/s2_dk1_baselines.yaml`。
2. 不要改滚动窗长、测试期起止、剔除日列表、评价指标定义、缺失值规则、arm 参数（4 MW / 8 MWh）。
3. 不要重跑或修改 F01/F08 的预测，直接读它们的 `forecasts.csv`。
4. 不要给自由备用臂加冻结阈值，也不要给阈值臂放开 `r_up`。
5. 不要新增编号前缀。`B2` 用现有体系，Oracle 变体用具名方式。
6. 不要改 `RESEARCH_FRAMEWORK.md` 里 RQ 的定义或可证伪主张。
7. 不要为了让 `B2` 好看而调 arm 参数或放宽可行域。
8. `exclusivity_violations`、`slots_with_shortfall`、`recovery_overlap_slots` 照实报告，不隐藏、不消化。自由备用优化预计会让互斥违例上升，这是 LP 松弛的已知代价，报出来。
9. 不要 `git add -A` / `git add .`。
10. 不要新建 README、SUMMARY、CHANGELOG、`*_GUIDE.md`、`*_NOTES.md`、`*_REPORT.md`。
11. **术语**：F01 在 DK1 上是同协议迁移评测，F08 是外部基线，唯一的「复现」是 EPF-DE 上的 R1。

---

## 5. 验收

**第 1 步 回归**
```bash
cd /public/ZLCODE/electricity-price-forecasting-research && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/s2_dk1_baselines.py --config configs/s2_dk1_baselines.yaml
```
`git diff --stat p1_paper/results/s2_dk1_baselines/` 必须为空，B1 仍是 `433617.76291309856`。

**第 2 步 新臂**
```bash
cd /public/ZLCODE/electricity-price-forecasting-research && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/s2c_decision_layer.py --config configs/s2c_decision_layer.yaml
```
通过条件：

- 四臂各 211 天；
- 复算的 `perfect_quarter_price` 等于 `485998.67801063386`（容差 `1e-6`），**这是与已有结果同口径的硬校验，不等就停下查，不要调容差**；
- 逐日 `|terminal_soc_deviation_mwh|` < `solver.soc_tolerance_mwh`；
- `oracle_price_reserve` ≥ `perfect_quarter_price` 且 ≤ `Oracle`，否则说明可行域或目标写错了；
- `B2_F01` ≤ `oracle_price_reserve` 且 `B2_F08` ≤ `oracle_price_reserve`，否则说明预测臂拿到了实测信息。

**第 3 步 F08 表更新**
```bash
cd /public/ZLCODE/electricity-price-forecasting-research && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /public/ZLCODE/.venvs/r1_lear_de/bin/python experiments/f08_lightgbm_dk1.py --config configs/f08_lightgbm_dk1.yaml
```
`forecasts.csv`、`daily_results.csv`、`day_index_map.csv` 必须字节不变；`summary.json` 只应新增 `forecast_versus_settlement`，已有字段值不变；`arm_comparison.csv` 按 3.3 更新。

**第 4 步 交付判断**

| 臂 | 价格 | 备用 | 激活 | EUR | 占 Oracle |
|---|---|---|---|---:|---:|
| B1 | 持续法 | 阈值 | 无 | 433,618 | 52.6% |
| F08 | 预测 | 阈值 | 无 | 445,055 | 54.0% |
| F01 | 预测 | 阈值 | 无 | 461,823 | 56.0% |
| **B2_F08** | 预测 | 自由 | 无 | 待测 | |
| **B2_F01** | 预测 | 自由 | 无 | 待测 | |
| perfect_quarter_price | 实测 | 阈值 | 无 | 485,999 | 58.9% |
| **oracle_price_reserve** | 实测 | 自由 | 无 | 待测 | |
| Oracle | 实测 | 自由 | 完美 | 824,744 | 100% |

按结果分三种情况写结论，**不要事先预设哪种**：

| 结果 | 结论 |
|---|---|
| 备用联合优化价值大、激活信息价值小 | 按原计划建 FB0 → FB1 → FB2＋B3；FB2 的联合分布重点放在容量价与激活量的联合尾部 |
| 两者都大 | 建 FB0＋FB2，但论文上界改用 `oracle_price_reserve` 报 regret，Oracle 只作参考并注明含不可部署信息 |
| 备用联合优化价值小 | **停下重议。** 说明阈值规则已接近该可行域下的最优，`FB2＋B3` 对 `FB0＋B1` 拿不到显著增益，需重新审视 arm 参数或研究问题本身 |

无论哪种，FB0 仍必须自建——它是主结论格的左半边，且 F01/F08 是外部基线、FB0 是自制对照臂，作用不同，不能顶替。

**第 5 步 提交**

逐路径 `git add`，一次提交。commit message 写：新增了什么臂、两块拆解各是多少、`B2_F01` 与 `B2_F08` 的排序是否与 0.2 的反例一致、落在上表哪一种情况、以及据此对 FB0–FB3 规模的决定。
