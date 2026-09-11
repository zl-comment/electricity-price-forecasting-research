# X14 来源卡：德国风电—电池日前与 aFRR 三阶段竞价

- 题目：*Wind-Battery Pool Optimal Bidding in German Energy and Secondary Control Reserve Markets*
- 作者：Gianluca Mancini, Stefanos Delikaraoglou, Eleni Stai, Ognjen Stanojev, Gabriela Hug
- 会议：*20th International Conference on the European Energy Market* (2024), 1–9
- DOI：[10.1109/EEM60825.2024.10608835](https://doi.org/10.1109/EEM60825.2024.10608835)
- 分类：交叉前沿 / 专业同行评审

| 项 | 全文核读结果 |
|---|---|
| 市场与样本 | 德国日前、aFRR 容量与 PICASSO 激活能量市场；400 MW 风电场与 30 MW / 45 MWh 电池组成价格接受者组合，两者不同址；2022-07 至 2022-12 逐日求解 |
| 决策时序 | 三阶段：09:00 (D-1) 报 aFRR 容量 → 12:00 (D-1) 报日前能量 → 交割日 15 分钟实时充放电；场景树按容量价分簇，同簇内日前决策须一致 |
| 容量价场景 | 按 4 小时块从历史均价中均匀抽样 |
| 日前价与激活价场景 | 直接取历史实现值；不平衡价为日前价加历史溢价 |
| 激活 | 历史统计的激活概率乘固定系数 0.5 得到平均激活功率，不生成激活路径 |
| 依赖 | 风电场景保留预测误差的时间相关；未描述容量价、日前价与激活之间的依赖 |
| SOC 与风险 | 按德国规则保证已中标容量可持续提供 1.5 小时；目标函数加 CVaR，风险系数人为设定并扫描 |
| 评估 | 主要报告模型内期望收益；只把优化中的双边不平衡价换成历史单一不平衡价做回测 |

`paper-reported`：风险中性时组合运行比两资产各自运行日均总收益高 20,203 EUR，来自电池抵消风电不平衡；电池自身日均收益反而低 12,843 EUR；电池收益主要来自 aFRR 容量与激活。

截至 2026-09-11，DOI、会议与页码由 Crossref 核验；全文由用户经机构订阅提供并已核读。该全文非开放获取，采集器无法重取，PDF 不入库，故保存来源卡。
