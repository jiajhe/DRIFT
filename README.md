# DRIFT Demo

DRIFT 模型代码、预训练权重和配套数据。

## 目录

```text
code/          模型实现、配置和加载脚本
checkpoints/   预训练权重
data/          A 股与美股 Qlib 数据
```

## 运行

使用 Python 3.12，在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt

./run.sh load --cell alpha158_csi300
./run.sh verify
```

`load` 加载指定模型；`verify` 检查权重、前向计算和数据读取。

## 模型

| 配置 | 数据 | 特征数 |
|---|---|---:|
| `alpha158_csi300` | CSI300 | 158 |
| `alpha360_csi300` | CSI300 | 360 |
| `alpha158_csi500` | CSI500 | 158 |
| `alpha360_csi500` | CSI500 | 360 |
| `alpha158us_sp500_pit` | S&P 500 | 157 |

回看窗口为 20 个交易日。训练、验证和测试区间分别为 2018–2022、2023、2024–2025。
各模型的配置与指标见 `code/configs/`。

## 论文主表的四 seed 集成

`checkpoints/paper_ensemble/manifest.json` 记录 Alpha158–CSI300、Alpha360–CSI300 和 Alpha158–S&P 500 各自的 seed `0、2、5、9`，以及按 2023 年验证集 IC 选出的 epoch、训练 run ID 和权重 SHA-256。上面的 `load` 命令仍用于加载单 seed demo。

在 CPU 上检查四个模型的加载、合成输入前向计算和分数聚合：

```bash
./run.sh ensemble --cell alpha158_csi300 --smoke-test
```

使用配套 Qlib 数据生成完整测试集的集成预测：

```bash
./run.sh ensemble --cell alpha158_csi300 --device cpu --output "$PWD/runs/alpha158_csi300_ensemble.parquet"
```

`--cell` 也支持 `alpha360_csi300` 和 `alpha158us_sp500_pit`。命令用验证期最后 20 个交易日预热各自的状态。每个交易日先把各 seed 的股票截面分数做 z-score（`ddof=0`），再等权平均。输出列为 `date`、`instrument`、`label` 和 `score`。
