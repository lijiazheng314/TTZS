#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""  
矩阵博弈MDVI配置 - 理论验证实验
"""

MatrixMDVIConfig = {
    # 环境配置
    'game_type': 'shapley_two_stage',  # Shapley两阶段博弈 (2x2 Matching Pennies + 4个子博弈)
    'n_agents_team1': 3,  # Team 1智能体数量
    'n_agents_team2': 3,  # Team 2智能体数量  
    'n_actions': 2,  # 每个智能体的动作数(Shapley博弈是2x2)
    'cooperation_weight': 0.0,  # 团队协作权重 (0=完全独立, 1=完全团队奖励)
                                 # 先设为0,每对独立学习
    
    # 算法核心参数
    'K_iterations': 120,  # MDVI迭代次数
    'gamma': 0.9,  # 折扣因子(Shapley两阶段博弈需要>0)
    'double_q': False,
    
    # MDVI特有参数(优化策略:一次Planning样本,全程重用)
    'exploration_interval': 100,  # 禁用中途数据更换(设置很大的值)
    'Ns_samples': 0,  # 中途不补充数据
    'Tm_trajectory_length': 1,  # Planning轨迹长度(矩阵博弈为1)
    'Np_planning_samples': 20000,  # Planning阶段生成样本(与FQI相同,每次重新生成)
    'planning_resample_interval': 999,  # 禁用Planning样本重采样(设置很大的值,让它只在第0次迭代生成)
    'use_environment_model': False,  # 暂时关闭环境模型
    'cumulative_exploration': False,  # 关闭累积模式
    'use_sliding_window': False,  # 关闭滑动窗口
    
    # 网络架构参数(矩阵博弈用简单MLP即可,不需要RNN)
    'hidden_dim': 64,  # 隐藏层维度
    'obs_shape': 7,  # 观测维度(Shapley: 局部state[5] + 队友1负奖励[1] + 队友2负奖励[1])
    'state_shape': 5,  # 状态维度(Shapley: [stage(1), a1_onehot(2), a2_onehot(2)] = 5)
    'n_agents': 6,  # 总智能体数(team1+team2)
    'n_actions_team2': 2,  # Team 2动作数
    
    # 优化器参数(优化以减少震荡)
    'lr': 0.003,  # 初始学习率
    'lr_decay': 0.98,  # 学习率衰减(更温和,0.9太激进)
    'min_lr': 0.0003,  # 最小学习率(提高,避免后期学习太慢)
    'optimizer_type': 'Adam',
    'clip_grad_norm': 1.0,  # 梯度裁剪(适度防止梯度爆炸)
    
    # 训练参数(优化以减少震荡)  
    'fit_epochs_per_iteration': 1,  # 每次iteration只训练1个epoch，避免Target过时
    'batch_size': 128,  # 增大batch,减小梯度方差
    'train_batch_per_epoch': 10,
    
    # 数据收集参数(增加数据量和多样性)
    'trajectory_buffer_size': 10000,  # 样本容量(两阶段博弈会翻倍)
    'num_collection_episodes': 5000,  # 收集episode数(将产生20000个样本)
    'warmup_episodes': 100,
    
    # 共识矩阵类型
    'consensus_matrix_type': 'metropolis',  # 'metropolis', 'uniform'
    
    # 评估参数
    'eval_interval': 10,  # 评估间隔
    'eval_episodes': 100,  # 评估episode数
    
    # 日志和保存
    'log_dir': './log_mdvi',
    'save_dir': './mdvi_models',
    'save_interval': 20,
    
    # 其他参数
    'seed': 0,
    'use_cuda': False,
}


def get_mdvi_config():
    """获取矩阵博弈MDVI配置副本"""
    from copy import deepcopy
    return deepcopy(MatrixMDVIConfig)
