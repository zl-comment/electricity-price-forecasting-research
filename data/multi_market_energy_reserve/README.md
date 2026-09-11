# 电能量—备用多市场数据入口

更新日期：2026-09-10。该目录围绕新的论文目标重建，数据角色分为：真实多市场主数据、备用预测外部基准、透明联合出清机制基准和公开电价预测基准。四套数据不得拼接成同一个真实物理系统。

“论文—数据一一对应”在本库中表示：72 篇/项文献中的每一项都有明确的 `original_data_status` 和至少一条本地数据关系；关系区分“论文精确公开数据”“同市场部分字段”“当前项目迁移基准”和“机制仿真基准”。这不表示为每篇论文复制一份相同 CSV，也不把未公开的原论文数据伪装成本地已有数据。正向映射见[论文—数据对应表](paper_data_alignment.json)，反向映射见[数据目录](catalog.json)，论文目录中也保存相同的 `data_links`。

## D1：Energinet DK1——真实多市场主数据

本地目录：[energinet_dk1](energinet_dk1/)。数据来自 [Energinet Energy Data Service](https://www.energidataservice.dk/)，统一查询 DK1、起始日 2025-10-01（含）、结束日 2026-09-01（不含）。官方数据依 [CC BY 4.0 条款](https://www.energidataservice.dk/terms-and-conditions) 使用。

| 文件 | 记录数 | 粒度与内容 | 当前异常 |
|---|---:|---|---|
| [day_ahead_prices.csv](energinet_dk1/day_ahead_prices.csv) | 32,160 | 15 分钟日前价格，EUR/DKK | 无空值 |
| [afrr_capacity_market.csv](energinet_dk1/afrr_capacity_market.csv) | 8,040 | 小时级 aFRR 上/下需求、采购量和容量价格 | 无空值 |
| [imbalance_and_activation.csv](energinet_dk1/imbalance_and_activation.csv) | 32,160 | 15 分钟不平衡价格、aFRR 激活量/加权价格、mFRR 边际价格 | `SatisfiedDemand` 缺 18，`DominatingDirection` 缺 3 |
| [wind_solar_forecasts.csv](energinet_dk1/wind_solar_forecasts.csv) | 23,892 | 三类风光的日前、日内、5 小时、1 小时和当前预测 | 四个预测期限分别缺 108、36、77、77 个值 |
| [production_consumption_settlement.csv](energinet_dk1/production_consumption_settlement.csv) | 7,941 | 小时级负荷、风光发电、交换和网损 | 最新仅到 2026-08-28 18:00 UTC，比完整窗口少 99 小时，不补值 |

API 返回采用倒序；同时保留 UTC 与丹麦本地时间。夏令时切换和 15 分钟/小时级连接必须以 UTC 为主键，不能按本地钟点直接去重。原始 JSON、官方字段元数据和无损 CSV 序列化均保留；CSV 没有插值、聚合或数值换算。

这套数据能支持：日前能量价格、aFRR 容量价格、实时不平衡/激活和风光预测的同期研究。它**不包含真实储能报价、成交、SOC 或设备控制记录**，所以首轮储能实验应明确为“使用真实市场观测驱动的价格接受者仿真”，不能写成现场运营复现。

## D2：芬兰 aFRR——备用预测外部基准

本地目录：[finland_afrr_zenodo_17494556](finland_afrr_zenodo_17494556/)，来源为 [Zenodo 数据集 10.5281/zenodo.17494556](https://doi.org/10.5281/zenodo.17494556)。建议使用 `extended_data_v2.csv`：6,456 个小时记录，覆盖 2024-06-20 22:00 UTC 至 2025-03-16 22:00 UTC，无空值；`v1` 保留用于核验早期版本。

该数据包含 aFRR 上/下价格、现货价格、容量价格、天气和用电量，适合复现备用价格预测并测试跨区域稳健性。它不是完整的电能量—备用联合交易记录，也不应与 DK1 行级拼接。

## D3：RTS-GMLC——联合出清机制基准

本地目录：[rts_gmlc_system](rts_gmlc_system/)，固定上游提交 `3ece0d3725c844056132393ee252b3083dd4eab4`，共保留 37 个系统参数及日前/实时时间序列文件。

RTS-GMLC 用于建立透明的电能量—备用联合出清、网络约束、SOC 连续性和压力场景。它不是电网运营商的真实历史市场记录；由优化器生成的价格、成交量、弃电和缺口必须标为**模型出清结果**，并公开目标函数、约束和价格提取方式。

当前通用审计还发现上游 `gen.csv` 的 `Output_pct_4`、`HR_incr_4` 各有 157 个空值，`storage.csv` 的 `Start Energy` 有 21 个空值。这些很可能对应分段曲线或起始能量字段的不适用项，但在完成字段语义核验前只记录为空，不擅自补零或默认值。

## D4：EPF-DE——论文精确公开预测基准

本地目录：[epf_de_benchmark](epf_de_benchmark/)。`DE.csv` 来自 EPFtoolbox 配套 [Zenodo 4624805](https://zenodo.org/records/4624805)，包含 52,416 个小时的德国日前电价、负荷预测和风光预测；同时保存固定上游提交中的 LEAR/DNN 作者预测。

它与 F01 是精确论文基准关系，并为从旧库迁入的 LightGBM、NBEATSx、TimeXer、CrossLinear、ProtoTS 和 UniCA 提供共同的项目评测数据。后六者在 D4 上运行时只能称为“同协议迁移评测”，不能称为复现其原论文结果。D4 也不能与 DK1 逐行拼接来制造一个多市场系统。

## 数据使用顺序

1. 先审计 DK1 市场时序、共同标签窗口、缺失值和信息可见时间；
2. 用简单持久性/线性/树模型建立各目标独立预测与联合预测基线；
3. 用统一储能模型比较单市场、顺序多市场和联合决策；
4. 用 RTS 验证联合出清与物理可交付机制；
5. 用芬兰数据做备用价格预测的跨市场外部检验。
6. 用 EPF-DE 检验预测骨干和决策导向训练是否在统一公开协议下成立，再迁移到 DK1。

当前不得静默填补任何缺失值，也不得随机切分时间序列。训练、验证、测试必须按时间划分，并保留每个特征在决策时刻是否已经可见的记录。

## 清单、审计与复现

- [下载清单](download_manifest.json)：来源 URL、查询窗口、版本、文件大小和 SHA-256。
- [数据目录](catalog.json)：四套数据的角色、边界及对应论文 ID。
- [论文—数据对应表](paper_data_alignment.json)：逐篇原始数据状态和本地数据关系。
- [数据审计](data_audit.json)：131 项论文/数据清单完整性校验、43 个 CSV 审计，以及 72 篇论文—4 套数据、77 条关系的双向对应检查。

```bash
python3 scripts/collect_multi_market_resources.py data
python3 scripts/build_paper_data_alignment.py
python3 scripts/audit_multi_market_resources.py
```

重新采集 Energinet 会得到当时的官方回溯版本；若官方修订历史值，应保留新旧清单和校验和，不覆盖后冒充同一数据快照。
