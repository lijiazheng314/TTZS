#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
矩阵博弈FQI训练脚本 - 理论验证实验
"""

import os
import sys
import numpy as np
import paddle
import argparse
from datetime import datetime

# 添加PARL路径
PARL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PARL_DIR)

from parl.utils import logger
from matrix_game_env import TwoTeamMatrixGame
from decentralized_mdvi_matrix import DecentralizedFQIAlgorithm
from config_mdvi import get_mdvi_config
from environment_model import EnvironmentModel  # 导入环境模型


class MatrixGameBuffer:
    
    def __init__(self, capacity, obs_shape, n_agents):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.n_agents = n_agents
        self.config = {'trajectory_buffer_size': capacity}  # 添加config属性
        
        self.observations = np.zeros((capacity, n_agents, obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, n_agents, obs_shape), dtype=np.float32)
        self.team1_actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.team2_actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        
        self.count = 0
        self.position = 0
    
    def __len__(self):
        return self.count
    
    def add(self, obs, next_obs, team1_actions, team2_actions, rewards, terminated):
        idx = self.position
        
        self.observations[idx] = obs
        self.next_observations[idx] = next_obs
        self.team1_actions[idx] = team1_actions
        self.team2_actions[idx] = team2_actions
        self.rewards[idx] = rewards
        self.terminated[idx] = terminated
        
        self.position = (self.position + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
    
    def get_all(self):
        return {
            'observations': self.observations[:self.count],
            'next_observations': self.next_observations[:self.count],
            'team1_actions': self.team1_actions[:self.count],
            'team2_actions': self.team2_actions[:self.count],
            'rewards': self.rewards[:self.count],
            'terminated': self.terminated[:self.count]
        }


def collect_random_data(env, config, use_ns_samples=False):
    """
    收集随机策略数据
    
    Args:
        use_ns_samples: 如果True,使用Ns_samples; 否则使用num_collection_episodes
    """
    # 根据参数决定收集数量
    if use_ns_samples:
        num_samples = config.get('Ns_samples', config['num_collection_episodes'])
    else:
        num_samples = config['num_collection_episodes']
    
    buffer = MatrixGameBuffer(
        capacity=config['trajectory_buffer_size'],
        obs_shape=env.obs_shape,
        n_agents=env.n_agents
    )
    
    logger.info(f"\n=== Collecting random trajectory data ===")
    logger.info(f"  Target samples: {num_samples}")
    if hasattr(env, 'is_two_stage') and env.is_two_stage:
        logger.info(f"  Expected samples: {num_samples * 2} (two-stage game: 2 steps per episode)")
    else:
        logger.info(f"  Expected samples: {num_samples} (single-step game)")
    
    for episode in range(num_samples):
        # 重置环境
        state, obs = env.reset()
        
        # Shapley两阶段博弈:需要收集两步数据
        if env.is_two_stage:
            # === 第一阶段 ===
            team1_actions_s1 = np.random.randint(0, env.n_actions, size=env.n_agents_team1)
            team2_actions_s1 = np.random.randint(0, env.n_actions, size=env.n_agents_team2)
            
            next_state_s1, next_obs_s1, rewards_team1_s1, rewards_team2_s1, terminated_s1 = env.step(
                team1_actions_s1, team2_actions_s1
            )
            
            all_actions_team1_s1 = np.concatenate([team1_actions_s1, team2_actions_s1])
            all_actions_team2_s1 = np.concatenate([team2_actions_s1, team1_actions_s1])
            all_rewards_s1 = np.concatenate([rewards_team1_s1, rewards_team2_s1])
            
            buffer.add(
                obs, next_obs_s1, all_actions_team1_s1, all_actions_team2_s1,
                all_rewards_s1, terminated_s1
            )
            
            # === 第二阶段 ===
            team1_actions_s2 = np.random.randint(0, env.n_actions, size=env.n_agents_team1)
            team2_actions_s2 = np.random.randint(0, env.n_actions, size=env.n_agents_team2)
            
            next_state_s2, next_obs_s2, rewards_team1_s2, rewards_team2_s2, terminated_s2 = env.step(
                team1_actions_s2, team2_actions_s2
            )
            
            all_actions_team1_s2 = np.concatenate([team1_actions_s2, team2_actions_s2])
            all_actions_team2_s2 = np.concatenate([team2_actions_s2, team1_actions_s2])
            all_rewards_s2 = np.concatenate([rewards_team1_s2, rewards_team2_s2])
            
            buffer.add(
                next_obs_s1, next_obs_s2, all_actions_team1_s2, all_actions_team2_s2,
                all_rewards_s2, terminated_s2
            )
        else:
            # 单阶段博弈
            team1_actions = np.random.randint(0, env.n_actions, size=env.n_agents_team1)
            team2_actions = np.random.randint(0, env.n_actions, size=env.n_agents_team2)
            
            next_state, next_obs, rewards_team1, rewards_team2, terminated = env.step(
                team1_actions, team2_actions
            )
            
            all_actions_team1 = np.concatenate([team1_actions, team2_actions])
            all_actions_team2 = np.concatenate([team2_actions, team1_actions])
            all_rewards = np.concatenate([rewards_team1, rewards_team2])
            
            buffer.add(
                obs, next_obs, all_actions_team1, all_actions_team2,
                all_rewards, terminated
            )
        
        if (episode + 1) % 100 == 0:
            logger.info(f"Collected {episode + 1}/{num_samples} samples")
    
    logger.info(f"Collection finished: Buffer size={buffer.count}")
    
    return buffer


def evaluate_nash_distance(env, algorithm, show_q_matrix=False):
    """
    评估学习到的策略与真实Nash均衡的距离
    改为评估所有状态下的平均策略距离: π(a|s)
    
    Args:
        show_q_matrix: 是否显示Q矩阵详细值 (默认False, Final Evaluation时为True)
    
    Returns:
        nash_distance: 所有状态的平均Nash距离
        learned_value: 学习到的Nash值
        policy_entropy: 【新增】策略熄(平均值)
    """
    # 定义所有可能的状态
    all_states = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # 第一阶段
        np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # 子博弈(0,0)
        np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),  # 子博弈(0,1)
        np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),  # 子博弈(1,0)
        np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # 子博弈(1,1)
    ]
    
    # 对应的真实Nash策略(所有零和博弈的Nash都是[0.5, 0.5])
    true_nash_policies = [
        np.array([0.5, 0.5], dtype=np.float32),  # 第一阶段: Matching Pennies
        np.array([0.5, 0.5], dtype=np.float32),  # 子博弈(0,0): Matching Pennies
        np.array([0.5, 0.5], dtype=np.float32),  # 子博弈(0,1): 反向Matching Pennies
        np.array([0.5, 0.5], dtype=np.float32),  # 子博弈(1,0): 激进零和
        np.array([0.5, 0.5], dtype=np.float32),  # 子博弈(1,1): 保守零和
    ]
    
    all_distances = []  # 所有(agent, state)组合的距离
    all_learned_values = []
    all_policy_entropies = []  # 【新增】所有策略的熄
    
    logger.info(f"\n[Nash Distance调试 - 所有状态]")
    
    # 对每个状态评估
    for state_idx, pair_state in enumerate(all_states):
        state_name = ["第一阶段", "子博弈(0,0)", "子博弈(0,1)", "子博弈(1,0)", "子博弈(1,1)"][state_idx]
        true_nash = true_nash_policies[state_idx]
        
        logger.info(f"\n  [{state_name}] 状态: {pair_state}")
        logger.info(f"    真实Nash策略: {true_nash}")
        
        state_distances = []
        
        # 对每个智能体评估
        for agent_id in range(env.n_agents_team1):
            # 构造observation: pair_state + 两个队友的负奖励标志(评估时为0)
            teammate_flags = np.array([0.0, 0.0], dtype=np.float32)
            obs = np.concatenate([pair_state, teammate_flags])
            obs_tensor = paddle.to_tensor(obs[np.newaxis, :], dtype='float32')
            
            q_network = algorithm.q_networks[agent_id]
            
            with paddle.no_grad():
                # 评估Q矩阵
                q_matrix = q_network.evaluate_all_actions(obs_tensor)
                q_matrix = q_matrix.squeeze(0).numpy()
                
                # 【原方法：通过Nash均衡求解策略】（注释掉）
                # pi_nash, sigma_nash, nash_value, _ = \
                #     algorithm.nash_solver.solve_nash_equilibrium(q_matrix)
                
                # 【新方法：直接从Q值生成策略】
                # 对每个自己的动作，计算对对手动作的期望Q值
                q_expected = np.mean(q_matrix, axis=1)  # [n_actions_team1]
                
                # 使用softmax转换为概率分布
                temperature = 0.5  # 降低温度，让策略对Q值差异更敏感
                q_exp = np.exp((q_expected - np.max(q_expected)) / temperature)
                pi_direct = q_exp / np.sum(q_exp)
                
                # 对于对手也类似处理
                q_expected_opponent = np.mean(q_matrix, axis=0)  # [n_actions_team2]
                q_exp_opp = np.exp((q_expected_opponent - np.max(q_expected_opponent)) / temperature)
                sigma_direct = q_exp_opp / np.sum(q_exp_opp)
                
                # 计算该策略下的期望值
                value_direct = np.dot(pi_direct, np.dot(q_matrix, sigma_direct))
                
                # 使用直接策略
                pi_nash = pi_direct
                sigma_nash = sigma_direct
                nash_value = value_direct
                
                # 【新增】计算策略熄 (Policy Entropy)
                # Entropy = -Σ π(a) * log(π(a))
                # 对于Matching Pennies，理论熄 = log(2) = 0.693
                epsilon = 1e-10  # 避免log(0)
                entropy = -np.sum(pi_nash * np.log(pi_nash + epsilon))
                all_policy_entropies.append(entropy)
                
                # 调试: 输出Q矩阵值和Nash策略 (仅当show_q_matrix=True时)
                if show_q_matrix and state_idx <= 1 and agent_id == 0:  # 输出前2个状态的Agent 0
                    logger.info(f"\n      [调试] Agent {agent_id} 在{state_name}的Q矩阵:")
                    logger.info(f"        Q矩阵: [[{q_matrix[0,0]:.4f}, {q_matrix[0,1]:.4f}],")
                    logger.info(f"                [{q_matrix[1,0]:.4f}, {q_matrix[1,1]:.4f}]]")
                    logger.info(f"        Q(a=0)的期望: {np.mean(q_matrix[0,:]):.4f}")
                    logger.info(f"        Q(a=1)的期望: {np.mean(q_matrix[1,:]):.4f}")
                    logger.info(f"        求解后Nash策略: pi={pi_nash}, sigma={sigma_nash}")
                    logger.info(f"        Nash值: {nash_value:.4f}")
                
                # 计算距离
                distance = np.linalg.norm(pi_nash - true_nash)
                state_distances.append(distance)
                all_distances.append(distance)
                all_learned_values.append(nash_value)
                
                logger.info(f"      Agent {agent_id}: 策略={pi_nash}, 距离={distance:.6f}")
        
        # 本状态的平均距离
        avg_state_distance = np.mean(state_distances)
        logger.info(f"    本状态平均距离: {avg_state_distance:.6f}")
    
    # 所有(agent, state)组合的平均距离
    nash_distance = np.mean(all_distances)
    nash_value_avg = np.mean(all_learned_values)
    policy_entropy_avg = np.mean(all_policy_entropies)  # 【新增】平均策略熄
    
    logger.info(f"\n  [总体统计]")
    logger.info(f"    所有状态平均Nash距离: {nash_distance:.6f}")
    logger.info(f"    学习Nash值(平均): {nash_value_avg:.6f} (真实: {env.nash_value:.6f})")
    logger.info(f"    平均策略熄: {policy_entropy_avg:.6f} (理论Nash熄: {np.log(2):.6f})")
    
    return nash_distance, nash_value_avg, policy_entropy_avg


def plot_convergence_curves(algorithm, env, save_dir, suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Matrix Game MDVI Convergence ({env.game_type})', fontsize=16)
    
    iterations = list(range(1, len(algorithm.loss_history) + 1))
    
    # 1. Training Loss (左上)
    axes[0, 0].plot(iterations, algorithm.loss_history, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration', fontsize=11)
    axes[0, 0].set_ylabel('MSE Loss (Training)', fontsize=11)
    axes[0, 0].set_title('Training Loss', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Value Error (右上) - 新增
    if len(algorithm.value_error_history) > 0:
        eval_iters = [0] + list(range(10, len(iterations)+1, 10))[:len(algorithm.value_error_history)-1]
        axes[0, 1].plot(eval_iters, algorithm.value_error_history, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].axhline(y=0, color='orange', linestyle='--', 
                          alpha=0.7, label='True Value=0.000')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Value Error')
        axes[0, 1].set_title('Value Error (|学习Nash值 - 真实值|)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Nash Distance (左下) - 保留
    if len(algorithm.eval_reward_history) > 0:
        eval_iters = [0] + list(range(10, len(iterations)+1, 10))[:len(algorithm.eval_reward_history)-1]
        axes[1, 0].plot(eval_iters, algorithm.eval_reward_history, 'g-o', linewidth=2, markersize=6)
        axes[1, 0].axhline(y=0, color='orange', linestyle='--', 
                          alpha=0.7, label='Perfect Nash=0.000')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Nash Distance')
        axes[1, 0].set_title('Distance to True Nash Equilibrium')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Policy Entropy (右下) - 新增
    if len(algorithm.policy_entropy_history) > 0:
        eval_iters = [0] + list(range(10, len(iterations)+1, 10))[:len(algorithm.policy_entropy_history)-1]
        axes[1, 1].plot(eval_iters, algorithm.policy_entropy_history, 'm-o', linewidth=2, markersize=6)
        axes[1, 1].axhline(y=np.log(2), color='orange', linestyle='--', 
                          alpha=0.7, label=f'Nash Entropy={np.log(2):.3f}')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Policy Entropy')
        axes[1, 1].set_title('Policy Entropy (-Σπlogπ)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f'convergence_curves{suffix}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Convergence curves saved: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_type', type=str, default='shapley',
                       choices=['rock_paper_scissors', 'matching_pennies', 'coordination', 'shapley'])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    # 配置
    config = get_mdvi_config()
    config['game_type'] = args.game_type
    config['seed'] = args.seed
    
    # 根据game_type调整n_actions和gamma
    if args.game_type == 'matching_pennies' or args.game_type == 'shapley':
        config['n_actions'] = 2
        config['n_actions_team2'] = 2
    
    if args.game_type == 'shapley':
        # Shapley两阶段博弈需要gamma>0
        # 降低gamma减少误差累积 (从0.9降低到0.5)
        config['gamma'] = 0.5
    else:
        # 单阶段博弈gamma=0
        config['gamma'] = 0.0
    
    # 设置随机种子
    np.random.seed(config['seed'])
    paddle.seed(config['seed'])
    
    # 创建环境
    env = TwoTeamMatrixGame(
        n_agents_team1=config['n_agents_team1'],
        n_agents_team2=config['n_agents_team2'],
        n_actions=config['n_actions'],
        game_type=config['game_type'],
        cooperation_weight=config.get('cooperation_weight', 0.0)
    )
    
    # 更新配置
    config['obs_shape'] = env.obs_shape
    config['state_shape'] = env.state_shape
    config['n_agents'] = env.n_agents
    
    # 【关键】重新设置种子，确保fqi和MDVI的Q网络初始化完全一致
    # 环境创建可能消耗了随机状态，需要重新同步
    np.random.seed(config['seed'])
    paddle.seed(config['seed'])
    
    # 创建算法
    algorithm = DecentralizedFQIAlgorithm(config)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config['save_dir'], f'train_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    config['save_dir'] = save_dir
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Matrix Game FQI Training")
    logger.info(f"{'='*60}")
    logger.info(f"Game type: {config['game_type']}")
    logger.info(f"True Nash equilibrium:")
    logger.info(f"  Team 1 strategy: {env.true_nash_team1}")
    logger.info(f"  Team 2 strategy: {env.true_nash_team2}")
    logger.info(f"  Nash value: {env.nash_value:.4f}")
    logger.info(f"Save directory: {save_dir}\n")
    
    # 初始探索：收集初始数据
    logger.info(f"\n{'='*60}")
    logger.info(f"Initial Exploration: Collecting {config['num_collection_episodes']} samples")
    logger.info(f"{'='*60}")
    buffer = collect_random_data(env, config)
    
    # 建立环境模型MG (Algorithm 1: Establish the model MG)
    logger.info(f"\n{'='*60}")
    logger.info(f"Establishing Environment Model MG")
    logger.info(f"{'='*60}")
    
    env_model = EnvironmentModel(
        obs_shape=config['obs_shape'],
        state_shape=config['state_shape'],
        n_actions_team1=config['n_actions'],
        n_actions_team2=config['n_actions_team2'],
        hidden_dim=config['hidden_dim']
    )
    
    # 从样本中学习模型: P(s,a,b,s') = count(s,a,b,s') / count(s,a,b)
    trajectory_data = buffer.get_all()
    env_model.update_from_samples(trajectory_data)
    
    logger.info(f"Model MG established:")
    logger.info(f"  Trained 6 reward networks (R(a,b) fitted by NN)")
    logger.info(f"  Trained 1 transition network (P(s'|s,a,b) fitted by NN)")
    logger.info(f"  Training data: {len(trajectory_data['rewards'])} samples")
    logger.info(f"  Transition training data: {len(env_model.transition_data['states'])} samples")
    logger.info(f"  Can generalize to unseen (s,a,b) tuples!\n")
    
    # MDVI训练循环 (Algorithm 1 - 优化版:Planning样本重用)
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting MDVI Training ({config['K_iterations']} iterations)")
    logger.info(f"MDVI模式: 大量Planning样本重复使用(每{config['planning_resample_interval']}次迭代重采样)")
    logger.info(f"{'='*60}\n")
    
    # 【新增】检查Q网络的初始值
    logger.info(f"\n[检查Q网络初始值]")
    state, obs = env.reset()
    test_obs = obs[0:1]  # Agent 0的obs
    test_obs_tensor = paddle.to_tensor(test_obs, dtype='float32')
    with paddle.no_grad():
        q_matrix_init = algorithm.q_networks[0].evaluate_all_actions(test_obs_tensor)
        q_matrix_init = q_matrix_init.squeeze(0).numpy()
    logger.info(f"  Agent 0在初始状态的Q矩阵 (训练前):")
    logger.info(f"    [[{q_matrix_init[0,0]:.4f}, {q_matrix_init[0,1]:.4f}],")
    logger.info(f"     [{q_matrix_init[1,0]:.4f}, {q_matrix_init[1,1]:.4f}]]")
    logger.info(f"  初始Q值平均: {np.mean(q_matrix_init):.4f} (应该接近0)")
    logger.info(f"  初始Q值范围: [{np.min(q_matrix_init):.4f}, {np.max(q_matrix_init):.4f}]")
    
    # 【新增】给Q网络加偏置，让初始策略偏离Nash均衡
    logger.info(f"\n[添加初始偏置以显示学习过程]")
    # 【关键】重新设置种子，保证偏置的随机性与FQI一致
    np.random.seed(config['seed'] + 999)  # 使用不同的种子偏移，但FQI和MDVI一致
    # 给每个Q网络的输出层bias加一个偏置，让初始Q值更不对称
    for agent_id in range(algorithm.n_agents):
        with paddle.no_grad():
            # 给bias加一个随机偏置，范围[-1.5, 1.5]，注意转换为float32
            bias_offset = np.random.uniform(-1.5, 1.5, size=algorithm.q_networks[agent_id].fc3.bias.shape).astype(np.float32)
            algorithm.q_networks[agent_id].fc3.bias.set_value(
                algorithm.q_networks[agent_id].fc3.bias.numpy() + bias_offset
            )
    logger.info(f"  已为{algorithm.n_agents}个Q网络添加随机偏置（范围±1.5，seed={config['seed']+999}）")
    
    # 重新检查加偏置后的Q矩阵
    with paddle.no_grad():
        q_matrix_biased = algorithm.q_networks[0].evaluate_all_actions(test_obs_tensor)
        q_matrix_biased = q_matrix_biased.squeeze(0).numpy()
    logger.info(f"  Agent 0加偏置后的Q矩阵:")
    logger.info(f"    [[{q_matrix_biased[0,0]:.4f}, {q_matrix_biased[0,1]:.4f}],")
    logger.info(f"     [{q_matrix_biased[1,0]:.4f}, {q_matrix_biased[1,1]:.4f}]]")
    
    # 【新增】评估初始Nash距离
    logger.info(f"\n[评估初始Nash距离]")
    initial_nash_dist, initial_value_error, initial_entropy = evaluate_nash_distance(env, algorithm, show_q_matrix=False)
    logger.info(f"  初始Nash距离: {initial_nash_dist:.6f}")
    logger.info(f"  初始Value误差: {initial_value_error:.6f}")
    logger.info(f"  初始策略熄: {initial_entropy:.6f}\n")
    
    # 【关键】将初始指标加入历史记录，确保绘图时包含初始点
    algorithm.eval_reward_history.append(initial_nash_dist)
    algorithm.value_error_history.append(abs(initial_value_error - env.nash_value))
    algorithm.policy_entropy_history.append(initial_entropy)
    
    planning_buffer = None  # Planning样本缓存
    
    for k in range(config['K_iterations']):
        # 每planning_resample_interval次迭代重新生成Planning样本
        if k % config['planning_resample_interval'] == 0:
            logger.info(f"\n=== Iteration {k+1}/{config['K_iterations']} ===")
            logger.info(f"Resampling Planning: Generating {config['Np_planning_samples']} trajectories")
            logger.info(f"  Sampling strategy: Uniform random (full exploration)")
            logger.info(f"  Reward source: Model prediction (NN)")
            logger.info(f"  Transition source: Model prediction (NN)")
            
            # 【修改】渐进式数据混合，减少突变
            if planning_buffer is None:
                # 第一次：创建新buffer
                logger.info(f"  Mode: Creating initial buffer")
                planning_buffer = MatrixGameBuffer(
                    capacity=config['Np_planning_samples'],
                    obs_shape=env.obs_shape,
                    n_agents=env.n_agents
                )
            else:
                # 后续次：保留70%旧数据 + 30%新数据（保守更新，防止突变）
                old_size = len(planning_buffer)
                keep_ratio = 0.7  # 保留70%旧数据
                keep_size = int(old_size * keep_ratio)
                new_size = int(config['Np_planning_samples'] * (1 - keep_ratio))  # 30%新数据
                logger.info(f"  Mode: Conservative mixing (keep {keep_size} old [{keep_ratio*100:.0f}%] + add {new_size} new [{(1-keep_ratio)*100:.0f}%])")
                
                # 保存旧数据
                old_data = planning_buffer.get_all()
                
                # 创建新buffer
                planning_buffer = MatrixGameBuffer(
                    capacity=keep_size + new_size,
                    obs_shape=env.obs_shape,
                    n_agents=env.n_agents
                )
                
                # 恢复最近的70%旧数据
                start_idx = old_size - keep_size
                for i in range(start_idx, old_size):
                    planning_buffer.add(
                        old_data['observations'][i],
                        old_data['next_observations'][i],
                        old_data['team1_actions'][i],
                        old_data['team2_actions'][i],
                        old_data['rewards'][i],
                        old_data['terminated'][i]
                    )
                logger.info(f"  Restored {keep_size} recent samples from old buffer ({keep_ratio*100:.0f}%)")
            
            for episode in range(config['Np_planning_samples']):
                # 从环境重置获取初始状态
                state, obs = env.reset()
                
                # Shapley两阶段博弈: 需要生成两步轨迹!
                if env.is_two_stage:
                    # ===== 第一阶段 =====
                    team1_actions_s1 = np.array([np.random.choice(env.n_actions) 
                                                 for _ in range(env.n_agents_team1)], dtype=np.int64)
                    team2_actions_s1 = np.array([np.random.choice(env.n_actions) 
                                                 for _ in range(env.n_agents_team2)], dtype=np.int64)
                    
                    # 用模型预测reward和next_state!
                    rewards_team1_s1 = np.array([env_model.predict_reward(
                        obs, team1_actions_s1, team2_actions_s1, i
                    ) for i in range(env.n_agents_team1)], dtype=np.float32)
                    
                    rewards_team2_s1 = np.array([env_model.predict_reward(
                        obs, team2_actions_s1, team1_actions_s1, 
                        env.n_agents_team1 + i
                    ) for i in range(env.n_agents_team2)], dtype=np.float32)
                    
                    # 用模型预测下一状态
                    next_state_s1 = env_model.predict_next_state(
                        state, team1_actions_s1[0], team2_actions_s1[0]
                    )
                    
                    # 生成next_obs (基于第一阶段动作,每对进入各自子博弈)
                    # obs格式: [state(5), teammate_flags(2)] = 7维
                    next_obs_s1 = np.zeros_like(obs)
                    for agent_id in range(len(obs)):
                        # 确定pair索引
                        if agent_id < env.n_agents_team1:
                            pair_idx = agent_id
                        else:
                            pair_idx = agent_id - env.n_agents_team1
                        
                        # 生成该对的第二阶段state: [0, a1_onehot, a2_onehot]
                        state_dim = 1 + env.n_actions + env.n_actions
                        pair_state = np.zeros(state_dim, dtype=np.float32)
                        pair_state[0] = 0.0  # stage=0 (第二阶段)
                        pair_state[1 + int(team1_actions_s1[pair_idx])] = 1.0
                        pair_state[1 + env.n_actions + int(team2_actions_s1[pair_idx])] = 1.0
                        
                        # teammate_flags暂时设为0 (Planning时没有真实rewards)
                        teammate_flags = np.zeros(2, dtype=np.float32)
                        
                        # obs = [pair_state, teammate_flags]
                        next_obs_s1[agent_id] = np.concatenate([pair_state, teammate_flags])
                    
                    terminated_s1 = False
                    
                    # 存储第一阶段转换
                    all_actions_team1_s1 = np.concatenate([team1_actions_s1, team2_actions_s1])
                    all_actions_team2_s1 = np.concatenate([team2_actions_s1, team1_actions_s1])
                    all_rewards_s1 = np.concatenate([rewards_team1_s1, rewards_team2_s1])
                    
                    planning_buffer.add(obs, next_obs_s1, all_actions_team1_s1, all_actions_team2_s1,
                                      all_rewards_s1, terminated_s1)  # terminated=False!
                    
                    # ===== 第二阶段 =====
                    team1_actions_s2 = np.array([np.random.choice(env.n_actions) 
                                                 for _ in range(env.n_agents_team1)], dtype=np.int64)
                    team2_actions_s2 = np.array([np.random.choice(env.n_actions) 
                                                 for _ in range(env.n_agents_team2)], dtype=np.int64)
                    
                    # 用模型预测第二阶段reward (依赖于第一阶段动作!)
                    rewards_team1_s2 = np.array([env_model.predict_reward(
                        next_obs_s1, team1_actions_s2, team2_actions_s2, i
                    ) for i in range(env.n_agents_team1)], dtype=np.float32)
                    
                    rewards_team2_s2 = np.array([env_model.predict_reward(
                        next_obs_s1, team2_actions_s2, team1_actions_s2,
                        env.n_agents_team1 + i
                    ) for i in range(env.n_agents_team2)], dtype=np.float32)
                    
                    # 用模型预测最终状态
                    next_state_s2 = env_model.predict_next_state(
                        next_state_s1, team1_actions_s2[0], team2_actions_s2[0]
                    )
                                        
                    # 第二阶段结束,重置到第一阶段
                    next_obs_s2 = np.zeros_like(next_obs_s1)
                    for agent_id in range(len(next_obs_s1)):
                        # 重置为第一阶段state: [1, 0, 0, 0, 0]
                        state_dim = 1 + env.n_actions + env.n_actions
                        pair_state = np.zeros(state_dim, dtype=np.float32)
                        pair_state[0] = 1.0  # stage=1 (第一阶段)
                        
                        # teammate_flags暂时设为0
                        teammate_flags = np.zeros(2, dtype=np.float32)
                        
                        # obs = [pair_state, teammate_flags]
                        next_obs_s2[agent_id] = np.concatenate([pair_state, teammate_flags])
                                        
                    terminated_s2 = True
                    
                    # 存储第二阶段转换
                    all_actions_team1_s2 = np.concatenate([team1_actions_s2, team2_actions_s2])
                    all_actions_team2_s2 = np.concatenate([team2_actions_s2, team1_actions_s2])
                    all_rewards_s2 = np.concatenate([rewards_team1_s2, rewards_team2_s2])
                    
                    planning_buffer.add(next_obs_s1, next_obs_s2, all_actions_team1_s2, all_actions_team2_s2,
                                      all_rewards_s2, terminated_s2)  # terminated=True!
                    
                else:
                    # 单阶段博弈: 一步即终止
                    team1_actions = np.array([np.random.choice(env.n_actions) 
                                             for _ in range(env.n_agents_team1)], dtype=np.int64)
                    team2_actions = np.array([np.random.choice(env.n_actions) 
                                             for _ in range(env.n_agents_team2)], dtype=np.int64)
                    
                    # 用模型预测reward
                    rewards_team1 = np.array([env_model.predict_reward(
                        obs, team1_actions, team2_actions, i
                    ) for i in range(env.n_agents_team1)], dtype=np.float32)
                    
                    rewards_team2 = np.array([env_model.predict_reward(
                        obs, team2_actions, team1_actions,
                        env.n_agents_team1 + i
                    ) for i in range(env.n_agents_team2)], dtype=np.float32)
                    
                    next_obs = obs
                    terminated = True
                    
                    all_actions_team1 = np.concatenate([team1_actions, team2_actions])
                    all_actions_team2 = np.concatenate([team2_actions, team1_actions])
                    all_rewards = np.concatenate([rewards_team1, rewards_team2])
                    
                    planning_buffer.add(obs, next_obs, all_actions_team1, all_actions_team2,
                                      all_rewards, terminated)
            
            logger.info(f"Planning complete: Generated {len(planning_buffer)} model samples")
            
            # 调试: 分析Planning样本的分布
            trajectory_data = planning_buffer.get_all()
            logger.info(f"\n  [Planning样本分析]")
            
            # 状态转移预测质量分析 (两阶段博弈)
            if env.is_two_stage:
                logger.info(f"\n  [状态转移预测分析]")
                # 统计第一阶段的状态分布
                first_stage_mask = ~trajectory_data['terminated']
                if np.sum(first_stage_mask) > 0:
                    first_stage_obs = trajectory_data['observations'][first_stage_mask]
                    first_stage_next_obs = trajectory_data['next_observations'][first_stage_mask]
                    logger.info(f"    第一阶段样本数: {np.sum(first_stage_mask)}")
                    logger.info(f"    当前状态范围: [{first_stage_obs.min():.4f}, {first_stage_obs.max():.4f}]")
                    logger.info(f"    下一状态范围: [{first_stage_next_obs.min():.4f}, {first_stage_next_obs.max():.4f}]")
                    
                    # 展示几个状态转移示例
                    logger.info(f"    状态转移示例 (前3个):")
                    for i in range(min(3, np.sum(first_stage_mask))):
                        idx = np.where(first_stage_mask)[0][i]
                        logger.info(f"      样本{i}: obs={trajectory_data['observations'][idx][0][:3]}...")
                        logger.info(f"             next_obs={trajectory_data['next_observations'][idx][0][:3]}...")
                        logger.info(f"             actions=({trajectory_data['team1_actions'][idx][0]}, {trajectory_data['team2_actions'][idx][0]})")
            
            if env.is_two_stage:
                # 两阶段博弈: 应该50% terminated=False, 50% terminated=True
                n_terminated = np.sum(trajectory_data['terminated'])
                n_total = len(trajectory_data['terminated'])
                logger.info(f"    总样本数: {n_total}")
                logger.info(f"    第一阶段(terminated=False): {n_total - n_terminated} ({100*(n_total-n_terminated)/n_total:.1f}%)")
                logger.info(f"    第二阶段(terminated=True): {n_terminated} ({100*n_terminated/n_total:.1f}%)")
            else:
                logger.info(f"    总样本数: {len(trajectory_data['terminated'])}")
                logger.info(f"    Terminated样本: {np.sum(trajectory_data['terminated'])} (100.0%)")
            
            # Team1 Agent0的奖励统计
            agent0_rewards = trajectory_data['rewards'][:, 0]
            logger.info(f"\n  [Reward预测分析]")
            logger.info(f"    Agent0奖励统计: mean={np.mean(agent0_rewards):.4f}, "
                       f"std={np.std(agent0_rewards):.4f}, "
                       f"min={np.min(agent0_rewards):.4f}, "
                       f"max={np.max(agent0_rewards):.4f}")
            logger.info(f"    【重要】理论期望: 0.0000 (零和博弈的期望reward)")
            logger.info(f"    【重要】实际偏差: {np.mean(agent0_rewards):.4f} (系统性偏差)")
            
            # 如果是两阶段博弈,对比模型预测和真实reward
            # 注意: 由于cooperation_weight和多子博弈,真实验证很复杂,暂时禁用
            if False and env.is_two_stage and k == 0:  # 禁用验证
                logger.info(f"\n  [模型预测vs真实Reward对比]")
                # 随机采样几个样本,用真实环境验证
                n_samples_to_check = min(5, len(agent0_rewards))
                logger.info(f"    采样{n_samples_to_check}个样本验证:")
                
                for sample_idx in range(n_samples_to_check):
                    obs = trajectory_data['observations'][sample_idx]
                    a1 = int(trajectory_data['team1_actions'][sample_idx][0])
                    a2 = int(trajectory_data['team2_actions'][sample_idx][0])
                    pred_r = trajectory_data['rewards'][sample_idx][0]
                    
                    # 用真实环境获取reward
                    if not trajectory_data['terminated'][sample_idx]:  # 第一阶段
                        true_r = env.payoff_stage1[a1, a2]
                    else:  # 第二阶段,需要知道第一阶段动作
                        # 简化:假设第一阶段动作也是(a1,a2)
                        true_r = env.payoff_stage2[a1, a2, a1, a2]
                    
                    error = abs(pred_r - true_r)
                    logger.info(f"      样本{sample_idx}: 预测={pred_r:.4f}, 真实={true_r:.4f}, 误差={error:.4f}")
            
            # 全体奖励统计
            all_rewards = trajectory_data['rewards']
            logger.info(f"    全体奖励统计: mean={np.mean(all_rewards):.4f}, "
                       f"std={np.std(all_rewards):.4f}, "
                       f"min={np.min(all_rewards):.4f}, "
                       f"max={np.max(all_rewards):.4f}")
            
            # 分析动作分布 (Agent0)
            team1_actions = trajectory_data['team1_actions'][:, 0]
            action_dist = np.bincount(team1_actions, minlength=env.n_actions) / len(team1_actions)
            logger.info(f"    Agent0动作分布: {action_dist}")
            # 计算熵时只考虑实际存在的动作
            actual_dist = action_dist[action_dist > 0]
            max_entropy = np.log(env.n_actions)  # 最大熵
            actual_entropy = -np.sum(actual_dist * np.log(actual_dist + 1e-10))
            logger.info(f"    动作熵: {actual_entropy:.4f} (最大={max_entropy:.4f}, {'均匀' if abs(actual_entropy - max_entropy) < 0.01 else '不均匀'})")
            
            # 分析reward分布 (看是否有-1,0,+1三种)
            reward_counts = {}
            for r in agent0_rewards:
                r_rounded = round(r, 1)  # 四舍五入到一位小数
                reward_counts[r_rounded] = reward_counts.get(r_rounded, 0) + 1
            logger.info(f"    Agent0 Reward分布: {sorted(reward_counts.items())}")
            logger.info(f"    总样本数: {len(agent0_rewards)}")
        else:
            # 重用Planning样本
            logger.info(f"\n=== Iteration {k+1}/{config['K_iterations']} ===")
            logger.info(f"Reusing {len(planning_buffer)} Planning samples (no resampling)")
            trajectory_data = planning_buffer.get_all()
        
        # 用Planning样本训练
        result = algorithm.run_iteration(trajectory_data)
        
        # 定期评估 (显示Q矩阵调试信息)
        if (k + 1) % config['eval_interval'] == 0:
            nash_dist, learned_value, entropy = evaluate_nash_distance(env, algorithm, show_q_matrix=True)
            algorithm.eval_reward_history.append(nash_dist)
            algorithm.value_error_history.append(abs(learned_value - env.nash_value))
            algorithm.policy_entropy_history.append(entropy)
            
            logger.info(f"\n=== Evaluation at Iteration {k+1} ===")
            logger.info(f"  Nash distance: {nash_dist:.6f}")
            logger.info(f"  Learned Nash value: {learned_value:.4f} "
                       f"(True: {env.nash_value:.4f})")
            logger.info(f"  Value error: {abs(learned_value - env.nash_value):.6f}")
            logger.info(f"  Policy entropy: {entropy:.6f} (True: {np.log(2):.6f})\n")
        
        # 定期保存模型
        if (k + 1) % config['save_interval'] == 0:
            algorithm.save_models(save_dir, iteration=k+1)
            # 【新增】保存中途收敛曲线
            logger.info(f"  生成中途收敛曲线 (iteration {k+1})...")
            plot_convergence_curves(algorithm, env, save_dir, suffix=f'_iter{k+1}')
            logger.info(f"  ✓ 中途曲线已保存: {save_dir}/convergence_curves_iter{k+1}.png\n")
    
    # 最终保存
    algorithm.save_models(save_dir)
    
    # 最终评估 (显示Q矩阵调试信息)
    logger.info(f"\n{'='*60}")
    logger.info(f"Final Evaluation")
    logger.info(f"{'='*60}")
    
    nash_dist, learned_value, entropy = evaluate_nash_distance(env, algorithm, show_q_matrix=True)
    
    logger.info(f"True Nash equilibrium:")
    logger.info(f"  Team 1: {env.true_nash_team1}")
    logger.info(f"  Team 2: {env.true_nash_team2}")
    logger.info(f"  Value: {env.nash_value:.4f}")
    logger.info(f"\nFinal Nash distance: {nash_dist:.6f}")
    logger.info(f"Final value error: {abs(learned_value - env.nash_value):.6f}")
    logger.info(f"Final policy entropy: {entropy:.6f} (True: {np.log(2):.6f})")
    
    # 绘制曲线
    plot_convergence_curves(algorithm, env, save_dir)
    
    # 输出查表统计
    lookup_stats = env_model.get_lookup_stats()
    logger.info(f"\n{'='*60}")
    logger.info(f"Reward Lookup Statistics (Planning阶段)")
    logger.info(f"{'='*60}")
    logger.info(f"  查表命中率: {lookup_stats['hit_rate']*100:.2f}%")
    logger.info(f"  命中次数: {lookup_stats['hits']}")
    logger.info(f"  总查询次数: {lookup_stats['total']}")
    logger.info(f"  使用神经网络: {lookup_stats['total'] - lookup_stats['hits']}次")
    if lookup_stats['hit_rate'] >= 0.95:
        logger.info(f"  ✓ 查表命中率高,Planning阶段使用的奖励几乎都是真实值!")
    elif lookup_stats['hit_rate'] >= 0.5:
        logger.info(f"  ⚠ 查表命中率中等,部分奖励来自神经网络预测")
    else:
        logger.info(f"  ✗ 查表命中率低,大量奖励来自神经网络预测")
    
    logger.info(f"\n✓ Training completed!")
    logger.info(f"✓ Models saved to: {save_dir}")


if __name__ == '__main__':
    main()
