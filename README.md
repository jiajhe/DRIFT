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
