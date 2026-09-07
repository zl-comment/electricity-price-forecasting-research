# DK1 多市场数据字典

术语、数据角色和免责规则见 [CONVENTIONS.md](../CONVENTIONS.md)，本文件不重述。

结构性检查由 `experiments/s0_dk1_audit.py` 生成，结果在 `p1_paper/results/s0_dk1_audit/`。本文件中标注「实测」的数字全部来自该结果文件，标注「规则」的来自外部规则文档并给出链接，标注「待查」的尚无来源。

## 1. 市场时序

aFRR 容量闸门早于日前闸门 4.5 小时。储能必须在日前价格不可观测时承诺备用容量。这是市场规则强加的顺序，不是建模选择。

| 产品 | 交割单元 | 闸门关闭 | 结果公布 | 来源 |
|---|---|---|---|---|
| aFRR 容量（DK1 本地市场，上/下分别采购） | 1 小时（实测） | 07:30 CET (D-1) | 规则要求缓解情形下最迟 09:10 CET 可得 | 规则：[Nordic FRR CM Market Handbook](https://nordicbalancingmodel.net/wp-content/uploads/2023/06/Market-handbook-FRR-CM.pdf)、[Energinet aFRR 容量市场](https://energinet.dk/el/balancering-og-systemydelser/markeder-og-udbud/afrr-kapacitetsmarked/) |
| 日前电能量（SDAC） | 15 分钟（实测） | 12:00 CET (D-1) | 约 12:57 CET | 规则：[ENTSO-E SDAC](https://www.entsoe.eu/network_codes/cacm/implementation/sdac/)、[EPEX SDAC Timings](https://www.epexspot.com/sites/default/files/download_center_files/Day-Ahead%20MRC%20Processes%20(02.07.2019).pdf) |
| 不平衡结算 | 15 分钟（实测） | — | 修正价次一工作日 15:00 丹麦本地时 | 规则：[Energinet Imbalance price design](https://en.energinet.dk/electricity/balancing-and-ancillary-services/imbalance-price-design/)、[NBS Handbook v5.2](https://www.esett.com/app/uploads/2025/10/NBS-Handbook-v5.2.pdf) |
| 供需结算 | 1 小时（实测） | — | 末端滞后 75.75 小时（实测） | 实测 |
| 风光预测五档 | 1 小时（实测） | — | 各档签发时刻**待查** | — |

aFRR 容量报价开门为 00:00 (D-7)，ACER 决定 19/2020 规定容量市场时段为 07:00–10:00 CET (D-1)，容量闸门对 aFRR 与 mFRR 相同。

DK1 用 2024 年 10 月自建的本地 aFRR 容量市场，DK2 在北欧联合市场。07:30 这一数值的 DK1 适用性来自 Energinet 丹麦语页面的二手摘要，直接抓取该页返回 HTTP 403，**尚未一手确认**。

窗口起点 2025-10-01 交割日恰为 SDAC 15 分钟 MTU 上线日。不得为拉长历史而前移起点，此前为小时制。

## 2. 五张表的结构（实测）

价区全部为 DK1。UTC 时标全部无重复。

| 表 | 主键 | 声明分辨率 | 实测众数步长 | 唯一时标 | 缺时标 | 本地钟点重复 |
|---|---|---|---|---|---|---|
| `day_ahead_prices` | TimeUTC + PriceArea | PT15M | 900 s | 32,160 | 0 | 4 |
| `afrr_capacity_market` | TimeUTC + PriceArea | PT1H | 3600 s | 8,040 | 0 | 1 |
| `imbalance_and_activation` | TimeUTC + PriceArea | PT15M | 900 s | 32,160 | 0 | 4 |
| `wind_solar_forecasts` | HourUTC + PriceArea + ForecastType | PT1H | 3600 s | 7,977 | 63 | 1 |
| `production_consumption_settlement` | HourUTC + PriceArea | PT1H | 3600 s | 7,941 | 24 | 1 |

**本地钟点不是主键**。`TimeDK` / `HourDK` 在 2025-10-26 回拨时重复：15 分钟表各 4 个，小时表各 1 个。UTC 列在两次切换（2025-10-26、2026-03-29）前后连续无重复。所有连接一律用 UTC 列。

共同窗口 2025-09-30T22:00Z 至 2026-08-28T18:00Z，7,964 小时，**末端由 `production_consumption_settlement` 限制**。各表相对最晚可得时点的末端亏空：日前价与不平衡 0 小时，aFRR 容量与风光预测 0.75 小时，供需结算 75.75 小时。

## 3. 缺失与缺口（实测）

缺失一律保持缺失，不补零、不插值、不外推。

| 表 | 缺口位置 | 规模 |
|---|---|---|
| `wind_solar_forecasts` | 2025-11-21T22:00Z → 2025-11-24T14:00Z | 连续缺 63 个小时时标 |
| `production_consumption_settlement` | 2026-08-24T21:00Z → 2026-08-25T22:00Z | 连续缺 24 个小时时标 |

`wind_solar_forecasts` 三个类型的时标覆盖不同：海上风 7,940、陆上风 7,976、光伏 7,976（完整网格 8,040）。海上风比另两者另缺 36 个时标。

逐列缺失值：`ForecastDayAhead` 108、`Forecast5Hour` 77、`Forecast1Hour` 77、`ForecastIntraday` 36、`ForecastCurrent` 0。`imbalance_and_activation` 的 `SatisfiedDemand` 缺 18、`DominatingDirection` 缺 3。

## 4. 可见性

`ForecastCurrent` 及事后结算数据不得作为日前任务输入。

`wind_solar_forecasts` 的 `TimestampUTC` 是**该行最后写入时刻，不是签发时刻**：与 `HourUTC` 之差中位数为 +0.01 小时，23,892 行中 23,760 行落在目标小时 ±1 小时内。该表 `updateFrequency` 为 PT5M，历史版本不保留。五个视野列共用这一个时标，日前列的签发时刻在文件中无任何记录。

因此每个特征的 `available_at` 只能由市场规则推导，不能从数据读出。`available_at <= decision_time` 断言的阈值依赖第 1 节，第 1 节留白的格子填上之前该断言无法实现。

`ForecastCurrent` 与 `ForecastIntraday` 在同时非空的行中有 34.9% 数值相等，二者不是同一量。

## 5. 15 分钟与小时的聚合规则

日前价与不平衡为 15 分钟，aFRR 容量为小时。规则须预先固定，各类量用各自的统计量，不混用。**本节待定，由第 1 节确认后填写。**

## 6. 待查项与向 Energinet 的问询清单

前两项阻塞 `available_at <= decision_time` 断言。

| 编号 | 待查 | 阻塞什么 |
|---|---|---|
| Q1 | 风光预测五档各自的签发时刻 | `available_at` 阈值 |
| Q2 | DK1 本地 aFRR 容量市场闸门在 2025-10 至 2026-08 窗口内是否变更 | 时序图在窗口内的有效性 |
| Q3 | 不平衡价初值与终值的差异及修订规则 | 该价格能否作为实时市场标签 |

问询清单（发往 energidataservice.dk 支持渠道，逐条引用数据集编号）：

1. 数据集 40（Forecast Wind and Solar Power, Hour Resolution）中，`ForecastDayAhead`、`ForecastIntraday`、`Forecast5Hour`、`Forecast1Hour`、`ForecastCurrent` 五列各自的签发时刻如何定义？相对交割小时分别提前多久？
2. 数据集 40 的 `TimestampUTC` 记录的是该行最后更新时刻还是某一预测的签发时刻？若为最后更新时刻，是否存在保留各档签发时间戳的历史版本或归档接口？
3. 数据集 40 在 2025-11-21T22:00Z 至 2025-11-24T14:00Z 缺 63 个小时时标，原因为何？是否会补发？海上风比陆上风与光伏另缺 36 个时标，原因为何？
4. 数据集 145（aFRR Capacity Market）适用于 DK1 的报价截止时刻是否为 07:30 CET (D-1)？该时刻在 2025-10-01 至 2026-08-31 期间是否发生过变更？出清结果在当日几点对外发布？
5. 数据集 145 的上、下调是否为独立采购与独立出清？`UpProcuredMW` 与 `DownProcuredMW` 是否可由同一机组同时提供？
6. 数据集 160（Imbalance Price）的价格是初值还是终值？若存在修订，修订发布时刻与最长修订窗口为何？`SatisfiedDemand` 的 18 个缺失值与 `DominatingDirection` 的 3 个缺失值代表什么状态？
7. 数据集 57（Production and Consumption - Settlement）的发布滞后是否有承诺上限？本窗口实测末端滞后 75.75 小时，2026-08-24T21:00Z 至 2026-08-25T22:00Z 另缺 24 个小时时标，原因为何？
8. 数据集 129（Day-Ahead Prices）自 2025-10-01 交割日起为 15 分钟分辨率。2025-10-01 之前的小时值与之后的 15 分钟值是否可在同一序列中使用，还是应视为两个不同产品？
