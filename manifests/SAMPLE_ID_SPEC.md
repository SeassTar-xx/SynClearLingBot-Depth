# 样本 ID 规范

`sample_id` 是由原始 ZIP 成员路径确定性生成的样本标识。当前格式为 `<scene>/<frame>`，其中 `scene` 与 `frame` 分别来自实际归档文件名 `ImageNNNN_L/R`、`depthNNNN_L/R` 和 `segmentationNNNN_L/R`。它不依赖本地下载顺序。
