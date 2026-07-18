# ClearDepth / LingBot-Depth 评测项目

本仓库提供一套基于清单文件的可复现流程，用于在不向模型输入 GT 深度的前提下，评测 **LingBot-Depth v0.5** 在 SynClearDepth 上的深度补全与细化能力。

正式流程为：

```text
校正后的左右 RGB 与相机标定
  → 独立双目基线
  → 对齐的原始/不完整度量深度
  → LingBot-Depth（左 RGB 与输入深度）
  → 预测的度量深度
  → 仅使用 GT 的评测
```