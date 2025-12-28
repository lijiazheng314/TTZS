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
from decentralized_fqi_matrix import DecentralizedFQIAlgorithm
from config_matrix import get_matrix_config


class MatrixGameBuffer:
    
    def __init__(self, capacity, obs_shape, n_agents):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.n_agents = n_agents
        self.config = {'trajectory_buffer_size': capacity}  # 添加config属性以保持接口一致
        
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
    
    if show_q_matrix:
        logger.info(f"\n[Nash Distance调试 - 所有状态]")
    
    # 对每个状态评估
    for state_idx, pair_state in enumerate(all_states):
        state_name = ["第一阶段", "子博弈(0,0)", "子博弈(0,1)", "子博弈(1,0)", "子博弈(1,1)"][state_idx]
        true_nash = true_nash_policies[state_idx]
        
        if show_q_matrix:
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
                
                # 【原方法：求解Nash均衡策略】（注释掉）
                # pi_nash, sigma_nash, nash_value, _ = \
                #     algorithm.nash_solver.solve_nash_equilibrium(q_matrix)
                
                # 【新方法：直接从Softmax策略计算】
                # 对每个自己的动作，计算对对手动作的期望Q值
                q_expected_team1 = np.mean(q_matrix, axis=1)  # [n_actions]
                q_expected_team2 = np.mean(q_matrix, axis=0)  # [n_actions]
                
                # 使用softmax转换为概率分布
                temperature = 0.5  # 降低温度，让策略对Q值差异更敏感
                q_exp = np.exp((q_expected_team1 - np.max(q_expected_team1)) / temperature)
                pi_direct = q_exp / np.sum(q_exp)
                
                q_exp2 = np.exp((q_expected_team2 - np.max(q_expected_team2)) / temperature)
                sigma_direct = q_exp2 / np.sum(q_exp2)
                
                # 计算Nash值
                value_direct = float(np.dot(pi_direct, np.dot(q_matrix, sigma_direct)))
                
                # 使用直接策略
                pi_nash = pi_direct
                sigma_nash = sigma_direct
                nash_value = value_direct
                
                # 【新增】计算策略熵 (Policy Entropy)
                # Entropy = -Σ π(a) * log(π(a))
                # 对于Matching Pennies，理论熵 = log(2) = 0.693
                epsilon = 1e-10  # 避免log(0)
                entropy = -np.sum(pi_nash * np.log(pi_nash + epsilon))
                all_policy_entropies.append(entropy)
                
                # 调试: 输出Q矩阵值和Nash策略 (仅当show_q_matrix=True时)
                if show_q_matrix and state_idx <= 1 and agent_id == 0:  # 输出前2个状态的Agent 0
                    logger.info(f"\n      [调试] Agent {agent_id} 在{state_name}的Q矩阵:")
                    logger.info(f"        Q矩阵: [[{q_matrix[0,0]:.4f}, {q_matrix[0,1]:.4f}],")
                    logger.info(f"                [{q_matrix[1,0]:.4f}, {q_matrix[1,1]:.4f}]]")
                    logger.info(f"        求解后Nash策略: pi={pi_nash}, sigma={sigma_nash}")
                    logger.info(f"        Nash值: {nash_value:.4f}")
                
                # 计算距离
                distance = np.linalg.norm(pi_nash - true_nash)
                state_distances.append(distance)
                all_distances.append(distance)
                all_learned_values.append(nash_value)
                
                if show_q_matrix:
                    logger.info(f"      Agent {agent_id}: 策略={pi_nash}, 距离={distance:.6f}")
        
        # 本状态的平均距离
        if show_q_matrix:
            avg_state_distance = np.mean(state_distances)
            logger.info(f"    本状态平均距离: {avg_state_distance:.6f}")
    
    # 所有(agent, state)组合的平均距离
    nash_distance = np.mean(all_distances)
    nash_value_avg = np.mean(all_learned_values)
    policy_entropy_avg = np.mean(all_policy_entropies)  # 【新增】平均策略熄
    
    if show_q_matrix:
        logger.info(f"\n  [总体统计]")
        logger.info(f"    所有状态平均Nash距离: {nash_distance:.6f}")
        logger.info(f"    学习Nash值(平均): {nash_value_avg:.6f} (真实: {env.nash_value:.6f})")
        logger.info(f"    平均策略熄: {policy_entropy_avg:.6f} (理论Nash熄: {np.log(2):.6f})")
    
    return nash_distance, nash_value_avg, policy_entropy_avg


def plot_convergence_curves(algorithm, env, save_dir, suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Matrix Game FQI Convergence ({env.game_type})', fontsize=16)
    
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
        axes[1, 1].axhline(y=np.log(env.n_actions), color='orange', linestyle='--', 
                          alpha=0.7, label=f'Nash Entropy={np.log(env.n_actions):.3f}')
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
    config = get_matrix_config()
    config['game_type'] = args.game_type
    config['seed'] = args.seed
    
    # 根据game_type调整n_actions和gamma
    if args.game_type == 'matching_pennies' or args.game_type == 'shapley':
        config['n_actions'] = 2
        config['n_actions_team2'] = 2
    
    if args.game_type == 'shapley':
        # Shapley两阶段博弈需要gamma>0
        config['gamma'] = 0.5
    
    # 设置随机种子
    np.random.seed(config['seed'])
    paddle.seed(config['seed'])
    
    # 创建环境
    env = TwoTeamMatrixGame(
        n_agents_team1=config['n_agents_team1'],
        n_agents_team2=config['n_agents_team2'],
        n_actions=config['n_actions'],
        game_type=config['game_type'],
        cooperation_weight=config.get('cooperation_weight', 0.0)  # 【新增】
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
    
    # 收集随机数据
    buffer = collect_random_data(env, config)
    trajectory_data = buffer.get_all()
    
    # 【新增】检查Q网络初始值
    logger.info(f"\n[检查Q网络初始值]")
    state, obs = env.reset()
    q_network = algorithm.q_networks[0]
    obs_tensor = paddle.to_tensor(obs[0][np.newaxis, :], dtype='float32')
    with paddle.no_grad():
        q_matrix_initial = q_network.evaluate_all_actions(obs_tensor)
        q_matrix_initial = q_matrix_initial.squeeze(0).numpy()
    logger.info(f"  Agent 0在初始状态的Q矩阵 (训练前):")
    logger.info(f"    {q_matrix_initial}")
    logger.info(f"  初始Q值平均: {np.mean(q_matrix_initial):.4f} (应该接近0)")
    logger.info(f"  初始Q值范围: [{np.min(q_matrix_initial):.4f}, {np.max(q_matrix_initial):.4f}]")
    
    # 【新增】给Q网络加偏置，让初始策略偏离Nash均衡（与MDVI一致）
    logger.info(f"\n[添加初始偏置以显示学习过程]")
    # 【关键】重新设置种子，保证偏置的随机性与MDVI一致
    np.random.seed(config['seed'] + 999)  # 使用不同的种子偏移，但FQI和MDVI一致
    for agent_id in range(algorithm.n_agents):
        with paddle.no_grad():
            # 给bias加一个随机偏置，范围[-1.5, 1.5]，与MDVI一致
            bias_offset = np.random.uniform(-1.5, 1.5, size=algorithm.q_networks[agent_id].fc3.bias.shape).astype(np.float32)
            algorithm.q_networks[agent_id].fc3.bias.set_value(
                algorithm.q_networks[agent_id].fc3.bias.numpy() + bias_offset
            )
    logger.info(f"  已为{algorithm.n_agents}个Q网络添加随机偏置（范围±1.5，seed={config['seed']+999}）")
    
    # 重新检查加偏置后的Q矩阵
    with paddle.no_grad():
        q_matrix_biased = algorithm.q_networks[0].evaluate_all_actions(obs_tensor)
        q_matrix_biased = q_matrix_biased.squeeze(0).numpy()
    logger.info(f"  Agent 0加偏置后的Q矩阵:")
    logger.info(f"    [[{q_matrix_biased[0,0]:.4f}, {q_matrix_biased[0,1]:.4f}],")
    logger.info(f"     [{q_matrix_biased[1,0]:.4f}, {q_matrix_biased[1,1]:.4f}]]")
    logger.info(f" ")
    
    # 【新增】评估初始Nash距离
    logger.info(f"\n[评估初始Nash距离]")
    initial_nash_dist, initial_value_error, initial_entropy = evaluate_nash_distance(env, algorithm, show_q_matrix=False)
    logger.info(f"  初始Nash距离: {initial_nash_dist:.6f}")
    logger.info(f"  初始Value误差: {abs(initial_value_error - env.nash_value):.6f}")
    logger.info(f"  初始策略熄: {initial_entropy:.6f}\n")
    
    # 【关键】将初始指标加入历史记录，确保绘图时包含初始点
    algorithm.eval_reward_history.append(initial_nash_dist)
    algorithm.value_error_history.append(abs(initial_value_error - env.nash_value))
    algorithm.policy_entropy_history.append(initial_entropy)
    
    # FQI训练循环
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting FQI Training ({config['K_iterations']} iterations)")
    logger.info(f"{'='*60}\n")
    
    for k in range(config['K_iterations']):
        # 执行一次迭代
        result = algorithm.run_iteration(trajectory_data)
        
        # 定期评估
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
            logger.info(f"  Policy entropy: {entropy:.6f} (True: {np.log(env.n_actions):.6f})\n")
        
        # 定期保存模型
        if (k + 1) % config['save_interval'] == 0:
            algorithm.save_models(save_dir, iteration=k+1)
            # 【新增】保存中途收敛曲线
            logger.info(f"  生成中途收敛曲线 (iteration {k+1})...")
            plot_convergence_curves(algorithm, env, save_dir, suffix=f'_iter{k+1}')
            logger.info(f"  ✓ 中途曲线已保存: {save_dir}/convergence_curves_iter{k+1}.png\n")
    
    # 最终保存
    algorithm.save_models(save_dir)
    
    # 最终评估
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
    logger.info(f"Final policy entropy: {entropy:.6f} (True: {np.log(env.n_actions):.6f})")
    
    # 绘制曲线
    plot_convergence_curves(algorithm, env, save_dir)
    
    logger.info(f"\n✓ Training completed!")
    logger.info(f"✓ Models saved to: {save_dir}")


if __name__ == '__main__':
    main()
