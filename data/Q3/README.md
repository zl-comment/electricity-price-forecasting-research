# Q3 数据入口

用途：固定储能容量下，研究午间新能源吸收与晚峰电量预留的跨时段调度。

| 目录 | 下载内容 | 用途与限制 |
|---|---|---|
| [rts_gmlc](rts_gmlc/README.md) | 73 节点系统参数、2020 标签年度的小时日前/5 分钟实时序列及备用资料 | 系统级主实验；合成测试系统，不是实网遥测 |
| [citylearn_2022](citylearn_2022/schema.json) | 17 栋建筑各 8,760 小时、天气、电价、碳强度、储能/PV 参数 | 小规模原型；缺真实保供事件与完整时间戳 |
| [elia](elia/download_manifest.json) | 官方负荷、风电、光伏；2023 全年与 2026-08-01 至 09-05 UTC 窗口，15 分钟 | 真实测量/外推及预测；无配套电池动作/网络，存在历史修订 |

完整分析、字段需求、论文匹配及数据缺口见 [讨论文档](../../research-foundation-2026/Q3_PAPERS_AND_DATA_DISCUSSION_2026-09-06.md)。

原始文件未补零或重采样。来源、固定提交、SHA-256 见 [基础下载清单](download_manifest.json) 与 [Elia 清单](elia/download_manifest.json)，字段与缺失检查见 [审计报告](data_audit.json)。Elia 近期光伏实际只返回至 2026-09-05 21:45 UTC，请求窗口末尾少 8 个时间点；2023 光伏有 304 个测量空值，近期风电有 40 个分类行的测量空值。下载完整性不等于数据无缺失。

MW 与 MWh、kW 与 kWh、GWh 必须分别处理；CityLearn 归一化光伏列、效率字段须按版本语义转换。Elia 下载为 UTC，午间/晚峰分析应转换当地时区。CityLearn 的 3 个室内环境字段全空，不影响当前只研究电储能的任务，但不得当作真实测量输入。

上游声明保留在 RTS README、CityLearn upstream_LICENSE 和 Elia upstream_licence.html。不要因这些文件可公开下载就默认具有同一种许可。
