# 矩阵博弈去中心化FQI - 理论验证实验

## 概述

这是去中心化拟合Q迭代(Decentralized Fitted Q-Iteration)算法在**矩阵博弈环境**下的理论验证实现。

与MPE环境不同,矩阵博弈具有以下优势:
- ✅ **解析可解的Nash均衡**: 可以精确计算真实Nash策略,便于验证算法收敛性
- ✅ **非平凡混合策略**: 设计的博弈需要混合策略Nash均衡(如剪刀石头布)
- ✅ **理论性能可量化**: 通过L2距离度量学习策略与真实Nash的差距
- ✅ **训练快速**: 单步博弈,无需长序列,训练时间短(约10-20分钟完成100次迭代)

## 文件结构

```
Matrix_FQI/
├── matrix_game_env.py           # 矩阵博弈环境(支持多种博弈类型)
├── config_matrix.py             # 配置文件
├── decentralized_fqi_matrix.py  # 简化版FQI算法(使用MLP替代RNN)
├── diging_optimizer.py          # DIGing优化器(复用自MPE_FQI)
├── train_matrix.py              # 训练脚本
└── README.md                    # 本文档
```

## 支持的博弈类型

### 1. Rock-Paper-Scissors (剪刀石头布)
- **动作空间**: 3 (Rock, Paper, Scissors)
- **Nash均衡**: (1/3, 1/3, 1/3) - 均匀混合策略
- **Nash值**: 0 (完全对称零和博弈)
- **Payoff矩阵**:
  ```
  [[  0, -1,  1],
   [  1,  0, -1],
   [ -1,  1,  0]]
  ```

### 2. Matching Pennies (配对硬币)
- **动作空间**: 2 (Head, Tail)
- **Nash均衡**: (0.5, 0.5)
- **Nash值**: 0
- **Payoff矩阵**:
  ```
  [[  1, -1],
   [ -1,  1]]
  ```

### 3. Coordination Game (协调博弈)
- **动作空间**: 3
- **Nash均衡**: 非对称混合策略(由LP求解器计算)
- **Payoff矩阵**:
  ```
  [[  2, -1,  0],
   [ -1,  1,  2],
   [  0,  2, -1]]
  ```

## 快速开始

### 训练

```bash
# 剪刀石头布博弈
python train_matrix.py --game_type rock_paper_scissors --seed 0

# 配对硬币博弈
python train_matrix.py --game_type matching_pennies --seed 0

# 协调博弈
python train_matrix.py --game_type coordination --seed 0
```

### 配置参数

关键超参数(在`config_matrix.py`中):

```python
MatrixFQIConfig = {
    'K_iterations': 100,           # FQI迭代次数
    'gamma': 0.0,                  # 折扣因子(单步博弈设为0)
    'hidden_dim': 64,              # MLP隐藏层维度
    'lr': 0.001,                   # 学习率
    'fit_epochs_per_iteration': 20, # 每次迭代训练轮数
    'batch_size': 64,              # Batch大小
    'num_collection_episodes': 1000, # 样本数(单步博弈=episode数)
    'consensus_matrix_type': 'metropolis', # 共识矩阵类型
}
```

## 理论验证指标

### 1. Nash距离 (主要指标)
$$
\text{Nash Distance} = \|\pi_{\text{learned}} - \pi_{\text{true}}\|_2
$$

- **目标**: < 0.01 (表示学习到的策略非常接近真实Nash均衡)
- **评估频率**: 每10次迭代

### 2. Nash值误差
$$
\text{Value Error} = |V_{\text{learned}} - V_{\text{true}}|
$$

- **目标**: < 0.001

### 3. 训练Loss
- **目标**: < 0.1 (MSE Loss)

## 算法简化说明

相比MPE_FQI版本,本实现做了以下简化:

1. **网络架构**: 使用简单MLP替代GRU
   - 原因: 矩阵博弈状态固定,不需要序列建模
   - 参数量: 从~20K降至~5K

2. **折扣因子**: γ=0
   - 原因: 单步博弈,无需考虑长期回报

3. **样本效率**: 更高的样本复用率
   - 1000样本 × 100迭代 × 20 epochs = 每样本被使用2000次

## 预期训练结果

### Rock-Paper-Scissors (理想情况)

```
Iteration 10:  Nash distance = 0.156, Loss = 0.324
Iteration 20:  Nash distance = 0.089, Loss = 0.143
Iteration 40:  Nash distance = 0.034, Loss = 0.067
Iteration 60:  Nash distance = 0.015, Loss = 0.032
Iteration 80:  Nash distance = 0.008, Loss = 0.018
Iteration 100: Nash distance = 0.005, Loss = 0.012

Final: Nash distance = 0.005 ✓ (目标 < 0.01)
```

### 收敛性标准

- ✅ Nash距离 < 0.01
- ✅ Loss稳定在 < 0.1
- ✅ 学习到的策略接近均匀分布 [0.33, 0.33, 0.33]

## 与MPE_FQI的对比

| 维度 | MPE_FQI | Matrix_FQI |
|------|---------|------------|
| 环境复杂度 | 高(连续状态,多步序列) | 低(单步博弈) |
| 真实Nash | 未知 | 可解析计算 |
| 训练时间 | 5-6小时(120迭代) | 10-20分钟(100迭代) |
| 验证方式 | 性能对比 | Nash距离+值误差 |
| 理论意义 | 实际应用验证 | **理论收敛性验证** |
| 网络架构 | GRU (复杂) | MLP (简单) |
| 样本数 | 5K-10K | 1K |

## 理论贡献验证

本实验设计用于验证以下理论结果:

1. **去中心化优化的收敛性**
   - DIGing算法能否在非平凡博弈中找到Nash均衡?
   - 共识误差是否收敛到0?

2. **样本复杂度**
   - 多少样本才能学习到精确的Nash策略?
   - 样本效率如何随智能体数量变化?

3. **网络容量**
   - 简单MLP是否足以表示Nash策略?
   - 需要多大的隐藏层维度?

## 故障排查

### 问题1: Nash距离不收敛

**可能原因**:
- 学习率过高/过低
- Epoch数不足
- 样本数太少

**解决方案**:
```python
# 增加训练轮数
'fit_epochs_per_iteration': 30  # 从20增至30

# 增加样本数
'num_collection_episodes': 2000  # 从1000增至2000

# 调整学习率
'lr': 0.0005  # 从0.001降至0.0005
```

### 问题2: Loss震荡

**可能原因**: Batch size过小

**解决方案**:
```python
'batch_size': 128  # 从64增至128
```

### 问题3: 收敛过慢

**可能原因**: 网络容量不足

**解决方案**:
```python
'hidden_dim': 128  # 从64增至128
```

## 输出文件

训练完成后,保存在`./matrix_models/train_YYYYMMDD_HHMMSS/`:

```
matrix_models/train_20251216_095408/
├── agent_0_iter_20.pdparams    # Agent 0模型(第20次迭代)
├── agent_0_iter_40.pdparams
├── ...
├── agent_0_final.pdparams      # 最终模型
├── agent_1_final.pdparams
├── ...
├── agent_5_final.pdparams
└── convergence_curves.png      # 收敛曲线图
```

## 可视化

`convergence_curves.png`包含4个子图:

1. **Training Loss**: MSE Loss随迭代变化
2. **TD Error**: TD误差随迭代变化  
3. **Nash Distance**: 学习策略与真实Nash的L2距离
4. **Smoothed Loss**: 5步移动平均Loss

## 引用

如果本实验对您的研究有帮助,请引用:

```bibtex
@misc{matrix_fqi_verification,
  title={Decentralized Fitted Q-Iteration: Matrix Game Verification},
  author={Your Name},
  year={2024},
  note={Theoretical verification experiment for Nash equilibrium convergence}
}
```

## 联系方式

如有问题,请通过issue或邮件联系。
