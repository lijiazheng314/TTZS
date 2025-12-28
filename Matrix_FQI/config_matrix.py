#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
矩阵博弈FQI配置 - 理论验证实验
"""

MatrixFQIConfig = {
    'game_type': 'shapley_two_stage',
    'n_agents_team1': 3,
    'n_agents_team2': 3,
    'n_actions': 2,
    'cooperation_weight': 0.0,
    'K_iterations': 120,
    'gamma': 0.9,
    'double_q': False,
    'hidden_dim': 64,
    'obs_shape': 7,
    'state_shape': 5,
    'n_agents': 6,
    'n_actions_team2': 2,
    'lr': 0.003,
    'lr_decay': 0.995,
    'min_lr': 0.0008,
    'optimizer_type': 'Adam',
    'clip_grad_norm': 0.2,
    'fit_epochs_per_iteration': 1,
    'batch_size': 128,
    'train_batch_per_epoch': 10,
    'trajectory_buffer_size': 100000,
    'num_collection_episodes': 10000,
    'consensus_matrix_type': 'metropolis',
    'eval_interval': 10,
    'eval_episodes': 100,
    'log_dir': './log_matrix',
    'save_dir': './matrix_models',
    'save_interval': 20,
    'seed': 0,
    'use_cuda': False,
}


def get_matrix_config():
    from copy import deepcopy
    return deepcopy(MatrixFQIConfig)
