# DRIFT-TSLM v2 数据

当前的金融数据入口为 `local/history-v3`。每日观测、未来监督以及线上使用的单边反事实分开存储。四任务的独立样本在 `tasks/`，所有特征统计与分类阈值在 `train_stats.json`。训练集 24,000 条，验证集 2,000 条，测试集 4,000 条，另有 1,000 道全市场随机配对题。

可匿名复现的公开版为 `10eit/drift-alpha158-flow-public@1.0.0`。官方构建器下载固定的 Kaggle v613 行情包，重算 157 个因子。98 个符合数据源核对要求的观测覆盖 30 只目标股和 11 个行业，共 392 题。训练、验证和测试分别为 39、17、42 个观测。联网与离线构建经 SDK 比对，678,160 个 float32 数值和全部任务逐项一致。命令与依赖见 `timenet_connectors/datasets/10eit/drift_alpha158_flow_public/README.md`。

原研究源 240 个观测使用独立 ID `10eit/drift-alpha158-flow@1.0.0`。其 TimeNet 代码位于 `timenet_connectors/datasets/10eit/drift_alpha158_flow`。在本机运行 `export_drift_v2_timef.py` 准备受限输入，再根据该 connector 的 README 运行官方构建。使用 `verify_drift_v2_timef.py --qlib-reference` 产生原 Qlib 参照矩阵，再用安装了 TimeNet 的环境运行同一脚本核对 SDK。此前发布的 v1 实验记录如下。

# 从 Qlib 因子到 TimeNet 数据集

建议以一个可重建的金融任务数据集提交 `<org>/marketflow`。本地训练可保留完整因子与结构张量。比赛 PR 包含 connector、card、依赖与测试；获准分发的数据工件另按版本和内容哈希引用。真实组织名、源许可、可供评审使用的下载地址仍待确定。

## 从哪些文件取数

|仓库来源|转换步骤|输出|
|---|---|---|
|`us_yahoo_sp500_pit_ohlcv_20260723/calendars/day.txt` 和 `instruments/sp500_pit.txt`|按交易日 t 和成份有效区间选股。保留有多段 membership 的同一 ticker|稳定的日期、股票标识、当日 PIT universe|
|`features/<symbol>/*.day.bin`|按 Qlib 日历偏移读取 open/high/low/close/volume/factor/change。`factor.day.bin` 是 provider 的单个字段；157 个因子由 `alpha158_us_ohlcv_features()` 返回的表达式和列名生成|带交易日和列名的因子矩阵；保留表达式 SHA、处理器和数据源版本|
|同源 close|取得截至 t 的 21 个连续交易日收盘，生成与 DRIFT 因子日期对齐的 20 个日收益。先保存原量纲，训练时按窗口归一化|每只股票一条数值序列和有效位|
|冻结 DRIFT 的当日输出|用 20×157 因子窗口、历史状态与完整当日可见截面算 K/Z/U/F；再选取五个 sender、同步置换|有方向的边、三个分量范数、结构张量、预测基线|
|独立 target 表|在同日 PIT 全市场有观测标签的股票上计算未来相对收益类别，将标签按 sample_id 关联|F/Q/M 答案及 split；未来收益仅在 F 的监督路径读取|

正式训练数据位于 `local/history-v2/`。一个观察日对应一个 NPZ，`manifest.json` 每日只存一次快照 SHA256，三个分区各用一份 NPY 保存监督。每条索引占 6 字节，依次为交易日编号、目标股票行号、任务编号、该任务的标签编号。三个索引合计 17,729,088 字节，全量发布共 2,982,121,574 字节。推理只用当天特征，无需安装任何一个监督索引。规模及独立校验见[数据审计](../../reports/data_audit.md)。

单日 Stage 0 仍可用 `code/TSLM_training/build_marketflow_samples.py` 生成 `local/samples-v1.jsonl`。新文件将快照 SHA256 集中存于首行，读取器也支持旧的逐行哈希。以 2022-12-28 的 1,449 条任务为例，索引从 925,428 字节减至 799,521 字节。

## 建议的 TimeF 表达

一个 `Record` 表示一个 `market|date|target` 观察。每只股票建一个有唯一归属的 `Source`。P0 有六个 Source，每个含一个 float32 历史收益 `Signal`。用 `OrdinalAxis()` 表示交易日序号，并保存对应 ISO 交易日列表，保留节假日间隔。ticker、target/sender 身份、顺序、时间范围和数据版本作为历史上下文保存。单个 observation 的多个任务共享同一个 Record。

如果提交的是**因子序列本身**，按同一 ticker/窗口建 Source，**每个命名因子一条沿交易日变化的 Signal**，例如 20×157 的美股输入映射为 157 条长度为 20 的 Signal。spec 标明因子名、原始表达式或其版本、数值单位、原始/已标准化、缺失处理；保留共同的时间轴与列顺序。可按股票及连续成份区间存一次长信号、用窗口引用减少重复。P0 的任务输入只引用所需收益信号；因子版若同时发布，宜独立建 Record 或父数据集，避免误将整段历史与非任务字段送入 LM。

本机用 AAPL 2022-11-30 至 2022-12-28 的原始 Qlib 因子构造了原生 TimeNet Record。一个 Source 下的 157 条 Signal 各有 20 个 float32 值。`record.signals` 枚举从 BETA10、BETA20 开始，Qlib 的原列序从 KMID、KLEN 开始。直接按遍历顺序拼接会错列。将信号按名称索引、再按保存的 factor_order 重建后，3,140 个值与原矩阵逐值相等。这是内存 API 验证，磁盘 build 与 SDK 回读仍待执行。六路收益同样必须按持久化的 ordered_ids 恢复 target/sender 顺序。

P1 的 Z/U/F/K 是 t 时点的特征向量。用 sample_id 引用固定 schema 的结构表（标明维度、dtype、checkpoint SHA、sender 顺序）；Q/M 对应的已知边权与范数可作为该任务可见事实。避免将向量维度当作时间轴。最终 DRIFT score 保存在 baseline/snapshot 表，不放入可直接序列化进 forecast prompt 的 Record 注释。所有 gold answer 只在各自 Task 的 `targets` 中。

下面是按固定版 SDK 的组装片段；其中的入参由上述转换产生，未提供示例行情或成绩。

```python
from timenet.dataset import OrdinalAxis, Record, Signal, Source
from timenet.types import Annotation, AnswerTask, Split, TimeSeriesSpec, ureg

spec = TimeSeriesSpec(spec_type="daily_return", name="Daily close-to-close return",
                      dtype="float32", unit_value=ureg.dimensionless)
sources = []
for ticker, returns in zip(ordered_ids, historical_returns, strict=True):
    signal = Signal.from_values(returns, spec=spec, name="daily_return",
                                time_axis=OrdinalAxis())
    sources.append(Source(ticker, signals=[signal]))
import json
record = Record(sources=sources, annotations=[
    Annotation("sample_id", value=sample_id),
    Annotation("ordered_ids", value=json.dumps(ordered_ids)),
    Annotation("trading_dates", value=json.dumps(trading_dates)),
])
dataset.add_record(record)
split = {"train": Split.TRAIN, "valid": Split.VALIDATION, "test": Split.TEST}[partition]
for question, answer in approved_tasks:
    dataset.add_task(AnswerTask(inputs=(record,), prompt=question, targets=answer, split=split))
```

## 准备 commit 与 PR

1. 逐源核对 `release_manifest.json` 中的 Yahoo 行情与日历、Wikipedia 成份、Qlib 离线包、Quandl/Kaggle 补数的获取权、衍生使用与再分发条件。记录各工件的源 URL、版本、哈希和署名。仓库已有数据、可下载以及经标准化都不足以确定发布许可。
2. 有可再分发授权时，发布版本化的合规 payload 并固定 SHA；本仓库可提交授权范围内的样本、schema、manifest 和转换代码。需要受控访问时，须有允许本地转换的上游条款、真实 access_url 和审查者可复现路径，card 填相应 access。缺少这些条件时用有明确许可的金融上游重建、改标 universe 并重评覆盖；不同数据源与股票池的成绩分开呈报。
3. 在官方仓库的 `packages/timenet-connectors/src/timenet_connectors/datasets/<org>/marketflow/` 提交 `__init__.py`（暴露 CONNECTOR）、`connector.py`、`dataset.yaml` 与测试。依赖写邻近的 `requirements.txt`。card 声明 id、版本、名称、描述、实际数据许可、source_url 与访问方式。数据集许可独立于代码许可。
4. `download(cache_dir)` 获取有授权且校验过的工件；`convert(raw_refs)` 依锁定日期创建 Record → Source → Signal 及 AnswerTask。预先导出有许可的真实派生数据时，connector 不需要 GPU。确保同行能从下载地址取得相同输入，移除对本机路径的依赖。
5. 在安装**含该 connector** 的 producer 环境中执行以下验收。命令使用固定版的 `--local-registry`；MarketFlow 正式 build 与 PR 均待完成。

```bash
export DATASET_ID='实际组织名/marketflow'
timenet-build build "$DATASET_ID" --local-registry .timenet-registry --no-isolation
python - <<'PY'
import os
from timenet import TimeNet
from timenet.types import Split
ds = TimeNet(sources=[".timenet-registry"]).load(os.environ["DATASET_ID"])
ds.describe()
for split in (Split.TRAIN, Split.VALIDATION, Split.TEST):
    assert sum(1 for task in ds.iter_tasks(split=split)) > 0
PY
```

再抽查每个 split 的 Signal 数值、交易日、ticker 顺序、target、任务数与本地样本逐项相同；在隔离环境从空缓存重建并记录版本。审核通过后记录 Dataset PR URL。测试文件入 Git，本机预览留在 `local/`。

官方依据：[Connector 规格](https://docs.timenet.ai/connectors.html)、[Build 与 SDK 回读](https://docs.timenet.ai/build.html)、[TSQA 源码](https://github.com/OpenTSLM/TimeNet/blob/979a0ae8894f225ecfd04a364678bea49f66bfc1/packages/timenet-connectors/src/timenet_connectors/datasets/chengsenwang/tsqa/connector.py)、[Hello World 源码](https://github.com/OpenTSLM/TimeNet/blob/979a0ae8894f225ecfd04a364678bea49f66bfc1/packages/timenet-connectors/src/timenet_connectors/datasets/timenet/hello_world/connector.py)。
