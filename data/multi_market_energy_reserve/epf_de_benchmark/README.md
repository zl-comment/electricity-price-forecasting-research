# D4：EPF-DE 公开基准

该目录从旧项目 `zl-comment/epf-frontier-study@8c6b0ee720c60ed85bed6f138201228951ca4c76` 的研究资产迁入，但重新绑定到可公开复现的第一方来源：

- [DE.csv](DE.csv)：来自 [EPFtoolbox Zenodo 4624805](https://zenodo.org/records/4624805)，SHA-256 为 `e421cf4eb160cecfec6c7c8222e7168b3c751d0dd6771ba21401784f3f2a4565`；
- [Forecasts_DE_DNN_LEAR_ensembles.csv](Forecasts_DE_DNN_LEAR_ensembles.csv)：来自固定提交 `jeslago/epftoolbox@47d6e0629f65ebd19d3c12cb5689dbad0c2ea078`，SHA-256 为 `1577f8ae0851fb536e654b69608c94840a9813aa5776d04691dd5ec1a2a3ea13`。

`DE.csv` 共 52,416 个小时，覆盖 2012-01-09 至 2017-12-31，字段为德国日前电价、Amprion 负荷预测和 PV+Wind 预测。预测文件包含 F01 论文协议下的 LEAR/DNN 作者预测，不能写成本仓库重新训练结果。

这套数据与 F01 是“论文精确公开基准”关系；对 F08--F13 等迁移模型只是当前项目的共同评测数据，并不等于复现这些方法论文的原始数据和结果。
