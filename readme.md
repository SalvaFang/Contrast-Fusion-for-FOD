
run `stats.py`，并配置一下参数
```python
# 数据集根路径
dataset_root = "data/cdnet/dataset"
# 检测结果根路径（与数据集路径结构保持一致）
binary_root = "data/cdnet/results"
# 统计结果根路径
output_path = "results_stats"
# 数据集名称，CDNet数据集默认为 0，LASIESTA数据集为 1
dataset_name = 0
# 统计
```

图像融合结果的评估参见https://github.com/AiqingFang/Objective-evaluation-for-image-fusion
