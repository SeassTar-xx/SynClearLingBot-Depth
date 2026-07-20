# SynClearLingBot-Depth

本项目在固定的 SynClearDepth 测试子集上，使用 StereoSGBM 从左右视图计算 raw depth，并将左视图与 raw depth 输入 LingBot-Depth v0.5，评测预测 depth 与 GT depth 的 RMSE 和 MAE。运行过程会按固定 sample ID 选择性获取数据、校验官方模型、缓存 raw depth，并支持中断后恢复。

## Quick Start

```bash
git clone https://github.com/SeassTar-xx/SynClearLingBot-Depth.git
cd SynClearLingBot-Depth

git clone --depth 1 https://github.com/Robbyant/lingbot-depth repos/lingbot-depth

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

bash scripts/run_depth_evaluation_pipeline.sh
```