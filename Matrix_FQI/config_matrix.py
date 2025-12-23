#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
矩阵博弈FQI配置 - 理论验证实验
"""

MatrixFQIConfig = {
    # 环境配置
    'game_type': 'shapley',  # Shapley两阶段博弈
    'n_agents_team1': 3,  # Team 1智能体数量
    'n_agents_team2': 3,  # Team 2智能体数量  
    'n_actions': 2,  # 每个智能体的动作数(Shapley博弈是2x2)
    'cooperation_weight': 0.0,  # 团队协作权重 (0=完全独立, 1=完全团队奖励)
    
    # 算法核心参数
    'K_iterations': 20,  # FQI迭代次数(增加以充分收敛)
    'gamma': 0.5,  # 折扣因子(Shapley两阶段博弈需要>0)
    'double_q': False,
    
    # 网络架构参数(矩阵博弈用简单MLP即可,不需要RNN)
    'hidden_dim': 64,  # 隐藏层维度
    'obs_shape': 7,  # 观测维度(Shapley: 局部state[5] + 队友1负奖励[1] + 队友2负奖励[1])
    'state_shape': 5,  # 状态维度(Shapley: [stage(1), a1_onehot(2), a2_onehot(2)] = 5)
    'n_agents': 6,  # 总智能体数(team1+team2)
    'n_actions_team2': 2,  # 【修改】Team 2动作数
    
    # 优化器参数(优化以减少震荡)
    'lr': 0.003,  # 初始学习率
    'lr_decay': 0.98,  # 学习率衰减（与MDVI一致）
    'min_lr': 0.0003,  # 最小学习率
    'optimizer_type': 'Adam',
    'clip_grad_norm': 1.0,  # 【关键】梯度裁剪（与MDVI一致，减少震荡）
    
    # 训练参数(优化以减少震荡)  
    'fit_epochs_per_iteration': 1,  # 【关键】每次iteration只训练1个epoch，避免Target过时
    'batch_size': 128,  # 增大batch,减小梯度方差
    'train_batch_per_epoch': 10,
    
    # 数据收集参数
    'trajectory_buffer_size': 100000,  # 样本容量
    'num_collection_episodes': 10000,  # 收集episode数(增加数据多样性)
    'warmup_episodes': 100,
    
    # 共识矩阵类型
    'consensus_matrix_type': 'metropolis',  # 'metropolis', 'uniform'
    
    # 评估参数
    'eval_interval': 10,  # 评估间隔
    'eval_episodes': 100,  # 评估episode数
    
    # 日志和保存
    'log_dir': './log_matrix',
    'save_dir': './matrix_models',
    'save_interval': 20,
    
    # 其他参数
    'seed': 0,
    'use_cuda': False,
}


def get_matrix_config():
    """获取矩阵博弈配置副本"""
    from copy import deepcopy
    return deepcopy(MatrixFQIConfig)
