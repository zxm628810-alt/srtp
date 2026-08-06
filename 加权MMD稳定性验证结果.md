# 加权 MMD 多随机种子稳定性验证结果

## 实验目的

单种子（seed=42）实验显示困难气体加权 MMD 在 Batch 10 可将 Ethylene Recall 提升 11 个百分点，但同时 Acetone 和 Toluene 略有下降。为排除随机初始化造成的偶然性，本实验在 3 个独立随机种子（11、23、37）下重复对比 MMD 基线与困难气体加权 MMD，确认 Batch 10 增益的稳定性。

两个模型均固定 MMD λ=0.5，架构完全一致（128→256→128→64→6）。唯一区别：困难气体加权方案将 Ethylene、Acetone、Toluene 的交叉熵 loss 权重设为 1.5（基线为 1.0）。规则保持严格：对每个测试 Batch k，只使用 Batch 1~k-1 训练。

## 整体均值（3种子 × 7批次）

| 指标 | MMD 基线 | MMD 困难气体加权 | 差值 |
|------|:---:|:---:|:---:|
| Accuracy | 81.39% | 81.41% | +0.02% |
| Macro-F1 | 0.750 | 0.754 | +0.004 |
| Ethylene Recall | 73.2% | 73.8% | +0.6% |
| Acetone Recall | 87.7% | 86.9% | -0.8% |
| Toluene Recall | 30.7% | 32.1% | +1.4% |

整体指标几乎相同，加权方案在 7 批次平均上没有明显增益。

## Batch 10 配对差值

| 种子 | MMD 基线 | 困难气体加权 | 差值 |
|:---:|:---:|:---:|:---:|
| 11 | 70.11% | 70.94% | **+0.83%** |
| 23 | 71.00% | 71.75% | **+0.75%** |
| 37 | 69.97% | 68.72% | **-1.25%** |
| **均值** | | | **+0.11%** |

两个种子正向、一个种子负向。Batch 10 的改善方向不一致，均值接近零。

## 结论

1. **困难气体加权 MMD 的 Batch 10 改善在多随机种子下不稳定。**两个种子正向（+0.8%左右）、一个种子负向（-1.25%），与之前种子 42 的 +0.75% 单次结果相比，没有稳定的统计学趋势。

2. **与普通 MMD 稳定性形成对比。**普通 MMD（无加权）在三种子下 Batch 10 差值分别为 +1.61%、+1.56%、+1.56%，方向完全一致。说明普通 MMD 的远期增益是稳定的，加权变体的增益不稳定。

3. **困难气体统一为 1.5 倍权重过于粗糙。**不同种子下 Ethylene/Acetone/Toluene 的 Recall 波动模式不一致（种子 11 改善 Toluene，种子 23 改善 Ethylene，种子 37 全部下降），说明统一权重无法适应不同初始化下的优化路径。

4. **当前最佳方案仍然是普通 MMD（λ=0.5，无加权）。**它在 Batch 10 上有稳定改善（+1.5~1.6%），困难气体加权未能提供额外的稳定增益。后续如需继续探索加权方向，应只提高 Ethylene 权重或使用更细粒度的类别-浓度分档权重。

## 可复核文件

- 脚本：`run_mmd_weighted_stability.py`
- 原始结果：`mmd_weighted_stability_results/mmd_weighted_stability_raw.csv`
- 分 Batch 汇总：`mmd_weighted_stability_results/mmd_weighted_stability_by_batch.csv`
- 总体汇总：`mmd_weighted_stability_results/mmd_weighted_stability_overall.csv`
- 成对比较：`mmd_weighted_stability_results/mmd_weighted_stability_paired.csv`
- 图：`mmd_weighted_stability_results/figures/加权MMD稳定性误差条图.png`、`加权MMD稳定性Batch10成对差值.png`、`加权MMD稳定性Batch10困难气体.png`
