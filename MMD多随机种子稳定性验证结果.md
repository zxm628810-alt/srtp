# MMD 多随机种子稳定性验证结果

## 实验目的

此前的单次滚动验证显示，MMD 领域自适应（`lambda=0.5`）可改善远期
Batch 10 的气体分类。为排除随机初始化造成的偶然性，本实验固定该权重，
在 3 个独立随机种子（11、23、37）下重复测试普通 DNN 和 MMD DNN。

规则保持严格：对每个测试 Batch k，只使用 Batch 1 至 k-1 训练；没有使用
测试 Batch 标签，也没有依据 Batch 9、10 调整 MMD 权重。

## 主要结果

| 指标（Batch 4--10 所有测试合计） | 普通 DNN | MMD DNN (`lambda=0.5`) |
|---|---:|---:|
| Accuracy（均值） | 81.49% | 81.39% |
| Macro-F1（均值） | 0.747 | 0.750 |
| 低浓度准确率（均值） | 70.21% | 68.60% |

总体准确率基本持平，说明 MMD 不是对所有时间批次都稳定提升的通用方法；但
它在最远期、漂移最强的 Batch 10 上有一致正向效果。

### Batch 10（三随机种子均值 ± 标准差）

| 指标 | 普通 DNN | MMD DNN | 变化 |
|---|---:|---:|---:|
| Accuracy | 68.79% ± 0.57% | **70.36% ± 0.56%** | **+1.57 个百分点** |
| Macro-F1 | 0.693 ± 0.005 | **0.710 ± 0.003** | **+0.018** |
| Ethylene Recall | 36.6% | **39.7%** | +3.1 个百分点 |
| Acetone Recall | 47.5% | **54.7%** | **+7.2 个百分点** |
| Toluene Recall | **81.8%** | 80.1% | -1.7 个百分点 |

三个随机种子的 Batch 10 Accuracy 差值（MMD - Baseline）分别为
`+1.61%`、`+1.56%`、`+1.56%`，方向一致。

## 结论

1. MMD 的主要价值是缓解**远期强漂移**，不是提升全部 Batch 的平均准确率。
2. 增益集中于此前边界分析中更易受漂移影响的 Ethylene 与 Acetone；这与
   PCA、混淆矩阵和决策边界覆盖分析相互印证。
3. Batch 4 的 MMD 表现较差，说明在漂移尚不明显时过强的分布对齐可能损失
   原始判别信息。因此，后续应将 MMD 作为强漂移候选方案，并继续测试与困难
   气体/低浓度样本加权结合后的效果。

## 可复核文件

- 脚本：`run_mmd_stability.py`
- 原始结果：`mmd_stability_results/mmd_stability_raw.csv`
- 分 Batch 汇总：`mmd_stability_results/mmd_stability_by_batch.csv`
- 总体汇总：`mmd_stability_results/mmd_stability_overall.csv`
- 成对比较：`mmd_stability_results/mmd_stability_paired_accuracy.csv`
- 图：`mmd_stability_results/mmd_stability_accuracy_errorbars.png`
