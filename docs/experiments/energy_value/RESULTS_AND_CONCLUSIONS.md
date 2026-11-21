# SCPRC 能源预测价值验证结果

完成日期：2026-11-05。主实验身份保持 Original PhaseFormer (MG) → SCPRC；Exact-PCGrad仅用于辨识关系适配器相对PCGrad-only的增量。

该组实验包含4个严格消融arm×18fold=72次独立训练，评价16个模型身份×18fold×2种后处理，共576行fold指标。全部城市、seeds和721起点均保留。

## 结果概述

主论文比较：Full相对Original PhaseFormer，raw的Peak、Ramp、24h Total和Timing误差分别降低21.67%、7.77%、41.74%和2.93%。这是18fold原始量值误差均值之比，不能解释成每个城市均改善相同比例；配对城市区间与逐城结果见下表及CSV。

相位条件关系适配器对峰值预测有额外价值；Full相对Exact在raw峰值MAE降低9.09%，在既有非负运营口径降低9.43%。Full相对Daily+完整衰减在raw峰值降低4.13%、非负峰值降低4.42%。这些是固定六城三seed的探索性、回顾性结果。

尺度边界：上述百分比是原始量值误差的等fold均值之比，大城市误差可主导绝对改变量。先在各fold归一再平均，Full对Exact的非负Peak收益为1.74%，Ramp为0.25%；Peak在4/6城市改善，Ramp仅1/6城市改善。移除SZH后Peak仍改善4.90%，但总幅度明显缩小。不能将9.43%写成典型城市均改善9.43%，也不能把聚合Ramp改善写成六城普遍改善。

完整衰减包不是所有能源指标上的最优组合。Scale主要体现总量与时刻误差的权衡；Peer改善固定容量proxy，却使峰值/时刻略差；Source-envelope增量很小，这些结果不足以证明三项衰减全部必要。

## 指标与后处理

Peak：每个起点未来24小时城市总量最大值的绝对误差。Ramp：23个相邻lead的一阶差分误差均值。Total：24小时累加总量的绝对误差。Timing：first-argmax峰值lead的非环形绝对时差（小时）。前三项保持原始量值单位。

raw是冻结点预测口径。nonnegative_clip先逐站点max(raw,0)，再求城市总量，沿用R12运营诊断。两张表属于不同estimand，不互相替换。

## 完整对比表：raw

| 模型                      |   Peak MAE |   Ramp MAE |   24h Total MAE |   Timing h |
|:--------------------------|-----------:|-----------:|----------------:|-----------:|
| Original PhaseFormer (MG) |  3278.4502 |   965.1712 |      40714.7605 |     5.4102 |
| Exact-PCGrad              |  2824.5944 |   913.6002 |      23312.4793 |     5.2779 |
| SCPRC                |  2567.9265 |   890.2114 |      23719.1915 |     5.2517 |
| Full-no-PCGrad            |  4133.4249 |  1001.1293 |      39648.7145 |     5.3223 |
| Daily relation            |  2861.6500 |   894.8852 |      22037.5452 |     5.2712 |
| Phase relation            |  2389.9605 |   908.9144 |      25282.6461 |     5.3565 |
| Scale-only (independent)  |  2526.7408 |   890.0112 |      23283.5287 |     5.1952 |
| Peer-only (independent)   |  2226.7966 |   854.1626 |      27082.3216 |     5.3185 |
| Source-only (independent) |  2189.4420 |   847.7545 |      24314.0125 |     5.3088 |
| Full (independent-switch) |  2568.8923 |   890.2609 |      23698.8682 |     5.2517 |
| Attention                 |  2421.1655 |   905.4685 |      24396.2498 |     5.2563 |
| DeepSets                  |  2530.1046 |   900.8974 |      24397.2138 |     5.2916 |
| Full-drop-scale           |  2226.4499 |   853.6306 |      27072.6336 |     5.3192 |
| Full-drop-peer            |  2526.1274 |   890.2233 |      23313.8715 |     5.1953 |
| Full-drop-source-envelope |  2570.2523 |   890.0431 |      23650.8181 |     5.2552 |
| Daily+full attenuation    |  2678.4751 |   911.0177 |      24651.8242 |     5.2692 |

## 完整对比表：nonnegative_clip

| 模型                      |   Peak MAE |   Ramp MAE |   24h Total MAE |   Timing h |
|:--------------------------|-----------:|-----------:|----------------:|-----------:|
| Original PhaseFormer (MG) |  2889.3216 |   963.2941 |      30978.1513 |     5.4508 |
| Exact-PCGrad              |  2807.4471 |   912.8331 |      22740.9672 |     5.2924 |
| SCPRC                |  2542.5692 |   889.1255 |      22919.8924 |     5.2813 |
| Full-no-PCGrad            |  3855.3054 |   998.8757 |      32512.7799 |     5.3797 |
| Daily relation            |  2844.2020 |   894.1286 |      21461.6608 |     5.2923 |
| Phase relation            |  2382.5574 |   908.1025 |      24999.1668 |     5.3704 |
| Scale-only (independent)  |  2499.2650 |   888.0679 |      22370.9794 |     5.2277 |
| Peer-only (independent)   |  2212.8494 |   853.7934 |      26589.4965 |     5.3323 |
| Source-only (independent) |  2182.8247 |   847.3979 |      24049.5173 |     5.3192 |
| Full (independent-switch) |  2543.5348 |   889.1750 |      22899.5716 |     5.2813 |
| Attention                 |  2407.3276 |   903.8950 |      23887.9110 |     5.2636 |
| DeepSets                  |  2511.0004 |   899.9511 |      23771.3008 |     5.2999 |
| Full-drop-scale           |  2212.5135 |   853.2618 |      26579.5711 |     5.3326 |
| Full-drop-peer            |  2498.6181 |   888.2809 |      22400.2049 |     5.2272 |
| Full-drop-source-envelope |  2544.9245 |   888.9586 |      22852.3306 |     5.2812 |
| Daily+full attenuation    |  2660.2241 |   910.2521 |      24051.8963 |     5.2829 |

## 按城市聚类的配对敏感性区间

差值=左侧模型误差−Full误差，正值有利于Full。每次重抽6个城市并保留其全部3seed，10,000次percentile bootstrap。只有6城，且LOCO模型共享源城市；这不是新城市/未来时期的保证。区间均为探索性pointwise区间，无多重比较校正，不据此写全指标显著占优。

| left_model                | metric                |   mean_difference |   ci95_low |   ci95_high |   right_better_cities |
|:--------------------------|:----------------------|------------------:|-----------:|------------:|----------------------:|
| original_phaseformer      | peak_magnitude_mae    |          346.7524 |   -26.5924 |    872.7240 |                     5 |
| original_phaseformer      | ramp_mae              |           74.1687 |    -3.7445 |    216.1802 |                     5 |
| original_phaseformer      | total_24h_mae         |         8058.2589 |  -596.6406 |  22988.5220 |                     4 |
| original_phaseformer      | peak_timing_mae_hours |            0.1695 |    -0.0452 |      0.4169 |                     3 |
| exact_pcgrad              | peak_magnitude_mae    |          264.8779 |     3.4811 |    752.4196 |                     4 |
| exact_pcgrad              | ramp_mae              |           23.7077 |    -0.9204 |     72.2748 |                     1 |
| exact_pcgrad              | total_24h_mae         |         -178.9252 | -1415.7530 |    896.2169 |                     3 |
| exact_pcgrad              | peak_timing_mae_hours |            0.0111 |    -0.0127 |      0.0349 |                     3 |
| full_drop_scale           | peak_magnitude_mae    |         -330.0557 | -1088.6670 |     87.7341 |                     3 |
| full_drop_scale           | ramp_mae              |          -35.8637 |  -109.9164 |      2.8314 |                     3 |
| full_drop_scale           | total_24h_mae         |         3659.6786 |  -111.1551 |   9949.3894 |                     4 |
| full_drop_scale           | peak_timing_mae_hours |            0.0513 |     0.0025 |      0.1021 |                     4 |
| full_drop_peer            | peak_magnitude_mae    |          -43.9511 |  -106.5927 |     -1.5158 |                     3 |
| full_drop_peer            | ramp_mae              |           -0.8446 |    -2.1256 |      0.2350 |                     3 |
| full_drop_peer            | total_24h_mae         |         -519.6875 | -1990.5648 |    309.8255 |                     4 |
| full_drop_peer            | peak_timing_mae_hours |           -0.0541 |    -0.0928 |     -0.0195 |                     1 |
| full_drop_source_envelope | peak_magnitude_mae    |            2.3552 |     0.0637 |      6.6564 |                     4 |
| full_drop_source_envelope | ramp_mae              |           -0.1669 |    -0.4926 |      0.0035 |                     3 |
| full_drop_source_envelope | total_24h_mae         |          -67.5618 |  -206.1616 |      3.1382 |                     4 |
| full_drop_source_envelope | peak_timing_mae_hours |           -0.0002 |    -0.0013 |      0.0012 |                     1 |
| daily_full_attenuation    | peak_magnitude_mae    |          117.6549 |     3.6169 |    307.9663 |                     4 |
| daily_full_attenuation    | ramp_mae              |           21.1266 |    -0.9761 |     64.6422 |                     1 |
| daily_full_attenuation    | total_24h_mae         |         1132.0039 |   -39.3407 |   2746.0315 |                     4 |
| daily_full_attenuation    | peak_timing_mae_hours |            0.0016 |    -0.0252 |      0.0293 |                     3 |

另保存18fold条件bootstrap，以及固定六城三seed的168-origin非环形moving-block bootstrap。后者在同一城市跨seed/模型/指标共用时间块，仅衡量已冻结模型的时间重采样敏感性，不能取代城市差异或重训练不确定性。

## 组件保留判断

- **Phase：有条件支持。** Daily+完整衰减与Full保持衰减、优化器和预算一致；Full峰值误差低4.42%，总量低4.71%，但总量城市区间跨0。不能从这里推出通用关系算子优越。
- **Scale：体现权衡。** 与Full-drop-scale相比，Full的24h Total MAE低13.77%、Timing低0.96%，但Peak高14.92%、Ramp高4.20%。主张应限于稳定总量/时刻与峰值响应之间的平衡。
- **Peer：精度必要性未建立。** Full-drop-peer峰值/时刻更好；Full在10%margin/penalty10的容量proxy低3.15%。这支持情景中的风险/预留权衡，不支持普遍精度增益或最佳复杂度。
- **Source-envelope：当前增量很弱。** 去掉后峰值MAE仅高约0.09%，Full的总量反而略差；gate均值0.99844。应降级为可选约束/边界分析，不能仅凭很窄区间宣称重要机制。

上述比率均以对照模型的等fold均值为分母；逐fold相对变化另在CSV中列出。

## 激活、周期性与条件关联

18fold冻结gate诊断与逐起点误差严格对齐，12,978个起点×两种后处理全部保留。Scale均值约0.8066、Peer约0.8303；Source约0.9984且15/18fold为常量。已计算输入168h内的lag24相关、24h/168h频率能量和24个issue-hour相位分组。逐fold Spearman方向不稳定，不能把gate激活或周期性相关当成因果收益。168h只有一个周周期，无法估计lag168自相关。

## 容量情景与负面结果

| model                                |   total_proxy_normalized |   overload_window_rate_pct |   overload_volume |   excess_reservation_pct |
|:-------------------------------------|-------------------------:|---------------------------:|------------------:|-------------------------:|
| attention_operator                   |                   1.8000 |                    29.3497 |          226.9797 |                   8.8458 |
| daily_full_attenuation               |                   1.9247 |                    34.1424 |          248.8458 |                   7.9066 |
| daily_relation                       |                   1.9244 |                    34.3967 |          256.1390 |                   7.8346 |
| deepsets_operator                    |                   1.8738 |                    32.0542 |          231.0199 |                   8.3078 |
| exact_pcgrad                         |                   1.9117 |                    34.2040 |          255.8489 |                   7.8717 |
| full_drop_peer                       |                   2.0216 |                    35.7682 |          234.3350 |                   7.3436 |
| full_drop_scale                      |                   1.9434 |                    33.8881 |          248.9382 |                   8.2761 |
| full_drop_source_envelope            |                   1.9582 |                    35.2905 |          234.7948 |                   7.5759 |
| full_independent_switch              |                   1.9580 |                    35.2674 |          234.4438 |                   7.5750 |
| full_no_pcgrad                       |                   2.8798 |                    42.5875 |          905.2828 |                   6.7033 |
| full_scpcr                           |                   1.9580 |                    35.2674 |          234.4437 |                   7.5754 |
| original_phaseformer                 |                   3.3391 |                    45.4307 |          294.9748 |                   5.4205 |
| peer_only_independent                |                   1.9444 |                    33.8958 |          249.0667 |                   8.2746 |
| phase_relation                       |                   1.9527 |                    34.3890 |          242.5429 |                   7.9848 |
| scale_only_independent               |                   2.0214 |                    35.7682 |          234.4058 |                   7.3459 |
| source_disagreement_only_independent |                   1.9441 |                    33.7263 |          239.2969 |                   8.1932 |

Full相对Original的proxy更低，但高于Exact与Seasonal-24h；Attention的proxy也更低。Full的超限量较Exact低约8.37%，超限窗口率却更高，说明风险量与发生频率不能互相替代。季节24/168与oracle保存在04_capacity/legacy_*表，未因对Full不利而删除。

16个固定margin/penalty组合中，Full对Original的proxy为16/16更低，对Full-drop-peer为15/16更低，但对Exact只有1/16更低。Peer因此有此类预留proxy的局部证据，这一结果仅支持相应预留情景下的局部收益，未显示Full在运营指标上普遍优于PCGrad-only。完整网格在04_capacity/proxy_grid_pairwise_tradeoffs.csv。

原始review/R12逐fold峰值、爬坡、总量、时刻和容量指标复核最大差约4.48e-7。没有真实电价、容量、电网或排队约束，因此不声称电费节约、调度可行性或电网安全。

## 实验条件与限制

- 四项新arm均独立seed重置并使用原2048源minibatch/410更新的PCGrad路径；不是修改Full checkpoint后做冻结gate开关。训练审计另在agent_strict_gap/audit。
- 新严格控制未同时重训Full，参照为既有canonical Full；预算、初始化约定和数据一致，但不能保证跨运行GPU轨迹逐位一致。checkpoint未保存逐步optimizer/RNG轨迹；因此不能将极小差异完全归因为组件移除。
- Full canonical与independent-switch Full分开保留；源代码/运行时间与CUDA归约可能引入微小差异。Full重复run敏感性CSV作为数值参照，不能当成正式训练噪声估计。Source的极小增量尤其需要谨慎。
- August/September已参与开发检查的事实未改变，结果仍属于回顾性评估；没有新盲测。
- 未基于测试结果改变模型选择、损失、训练预算、扰动掩码或评价集。没有追加训练去追逐有利结果。

## 分析代码与输入

重算能源指标需要对应检查点的逐起点预测、真实值、时间和站点索引。原始数组不随仓库提供；检查点及其文件身份见[权重索引](../../../manifests/CHECKPOINTS.csv)。

- [run_energy_evaluation.py](../../../experiments/energy_value/run_energy_evaluation.py)
- [analyze_mechanism.py](../../../experiments/energy_value/03_mechanism/analyze_mechanism.py)
- [reconcile_existing.py](../../../experiments/energy_value/04_capacity/reconcile_existing.py)
