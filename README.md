# DRIFT 精简运行目录

服务器：connect.nmb2.seetacloud.com:33691
目录：/root/autodl-tmp/drift-minimal

## GitHub 获取

仓库：https://github.com/jiajhe/DRIFT
本仓库直接包含代码、5 个精简权重和两套数据，不需要另行下载数据，也不依赖 Git LFS。

    git clone git@github.com:jiajhe/DRIFT.git
    cd DRIFT
    ./run.sh verify

Git clone 后的数据是普通文件。本服务器原整理目录使用硬链接节省空间。
新机器需要准备 Python 环境，可设置 DRIFT_PYTHON 指向对应解释器；
依赖版本见 code/ENVIRONMENT.json 与 code/requirements.txt。

## 三类内容

- code/：原项目运行时依赖闭包、训练入口、数据预处理入口、加载与核验脚本、5 组配置。
- data/：A 股与美股 Qlib provider；仅保留 calendars、instruments、features 和来源元数据。
- checkpoints/：5 组 A18 长预算实验中按验证集 IC 最高选出的单模型权重。

选择范围为每组 15 个种子；不是论文的四种子集成，也不是按 Test 选择。
对应时间切分：Train 2018--2022 / Valid 2023 / Test 2024--2025；lookback=20。
详细 seed、epoch、指标与原始路径见 MANIFEST.json。

## 直接使用

    cd /root/autodl-tmp/drift-minimal
    ./run.sh verify
    ./run.sh load --cell alpha158_csi300
    ./run.sh load --cell alpha360_csi300
    ./run.sh load --cell alpha158_csi500
    ./run.sh load --cell alpha360_csi500
    ./run.sh load --cell alpha158us_sp500_pit

Python 入口：

    from load_checkpoint import load_model
    model, config, checkpoint = load_model("alpha158_csi300", device="cpu")

model 输入：[股票数, 20, 特征数]，必须沿用 config 中的训练期拟合预处理。
时间序列预测需按日期保持模型状态；随机张量前向检查不是实际收益预测。
通过 ./run.sh python 启动 Python 会自动设置本目录 code/src 和 code 的导入路径。

已有原始训练入口 code/scripts/run_flow_fdg.py 与 compact 数据物化入口
code/scripts/materialize_compact_qlib.py；配置采用相对路径。
本次未启动训练或创建派生特征缓存。需要训练/完整复评时，先按 runtime.json
中的同一预处理配置物化 cache/compact，再运行相应入口。

## 权重与数据

权重是推理精简导出：保留 model_state、模型结构、数据配置、epoch 和选模信息。
优化器、随机数与恢复训练状态未复制，因此不能用于逐步完全一致地续训。
原始完整 checkpoint 的服务器路径和 SHA256 已写入 MANIFEST.json。
每个导出 state_dict 均逐张量核对，与原权重完全相同。

数据采用同一文件系统上的硬链接；不是依赖旧路径的符号链接。
删除旧目录不会令本目录的数据失效，但修改硬链接文件会修改同一底层数据。
请保持原始 provider 只读，将派生内容写到 cache/ 或新的输出目录。
下载源文件、压缩包、历史结果、日志、训练中间权重和 Python 环境均未复制。

运行脚本默认复用 /root/autodl-tmp/meta-fdg/.venv/bin/python。
更换环境时可设置 DRIFT_PYTHON；安装版本见 code/ENVIRONMENT.json 和 requirements.txt。
整个目录搬迁后仍使用相对数据/权重/配置路径；Python 环境需在目标机器另行准备。

VERIFY.json 记录本次哈希、严格加载、前向和真实原始行情/标签读取检查。
该检查不重新计算完整测试集指标。
