#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""  
矩阵博弈MDVI配置 - 理论验证实验
"""

MatrixMDVIConfig = {
    'game_type': 'shapley_two_stage',
    'n_agents_team1': 3,
    'n_agents_team2': 3,
    'n_actions': 2,
    'cooperation_weight': 0.0,
    'K_iterations': 120,
    'gamma': 0.9,
    'double_q': False,
    'Tm_trajectory_length': 1,
    'Np_planning_samples': 40000,
    'planning_resample_interval': 999,
    'use_environment_model': True,
    'hidden_dim': 64,
    'obs_shape': 7,
    'state_shape': 5,
    'n_agents': 6,
    'n_actions_team2': 2,
    'lr': 0.003,
    'lr_decay': 0.998,
    'min_lr': 0.0015,
    'optimizer_type': 'Adam',
    'clip_grad_norm': 0.2,
    'fit_epochs_per_iteration': 1,
    'batch_size': 256,
    'train_batch_per_epoch': 20,
    'trajectory_buffer_size': 20000,
    'num_collection_episodes': 5000,
    'consensus_matrix_type': 'metropolis',
    'eval_interval': 10,
    'eval_episodes': 100,
    'log_dir': './log_mdvi',
    'save_dir': './mdvi_models',
    'save_interval': 20,
    'seed': 0,
    'use_cuda': False,
}


def get_mdvi_config():
    from copy import deepcopy
    return deepcopy(MatrixMDVIConfig)
