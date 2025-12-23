#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Decentralized Fitted Q-Iteration Algorithm - Matrix Game Version
基于Algorithm 1：矩阵博弈理论验证版本
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import time
import os
from parl.core.paddle import Model
from parl.utils import logger
from scipy.optimize import linprog

from diging_optimizer import ConsensusMatrix, DIGingWrapper


# ============================================================================
# 1. Q网络定义 - 简化版(矩阵博弈不需要RNN)
# ============================================================================

class SimpleQNetwork(Model):
    """
    简化Q网络 - 适用于矩阵博弈
    Q^{i,k}(s, a, b) 对于智能体 i ∈ Team 1
    
    输入: 状态(obs) + Team1动作(one-hot) + Team2动作(one-hot)
    输出: Q值(标量)
    """
    
    def __init__(self, obs_shape, n_actions_team1, n_actions_team2, hidden_dim=64):
        super(SimpleQNetwork, self).__init__()
        
        self.obs_shape = obs_shape
        self.n_actions_team1 = n_actions_team1
        self.n_actions_team2 = n_actions_team2
        self.hidden_dim = hidden_dim
        
        # 输入 = obs + team1_action(one-hot) + team2_action(one-hot)
        self.input_shape = obs_shape + n_actions_team1 + n_actions_team2
        
        self.fc1 = nn.Linear(self.input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 【关键】初始化：适度随机范围，让收敛更平稳
        for layer in [self.fc1, self.fc2]:
            nn.initializer.XavierUniform()(layer.weight)
            nn.initializer.Constant(0.0)(layer.bias)
        
        # 最后一层使用较小范围的随机初始化（缩小以让收敛更平滑）
        nn.initializer.Uniform(low=-0.2, high=0.2)(self.fc3.weight)  # 从±0.5改为±0.2
        nn.initializer.Uniform(low=-0.05, high=0.05)(self.fc3.bias)  # 从±0.1改为±0.05
    
    def forward(self, obs, action_team1, action_team2):
        """
        前向传播
        
        Args:
            obs: [batch, obs_shape]
            action_team1: [batch] 动作索引
            action_team2: [batch] 动作索引
            
        Returns:
            q_value: [batch, 1]
        """
        # 检查输入维度
        if obs.shape[-1] != self.obs_shape:
            raise ValueError(f"obs维度不匹配! 得到{obs.shape[-1]}, 期望{self.obs_shape}")
        
        # One-hot编码
        action_team1_onehot = F.one_hot(action_team1.astype('int64'), 
                                       num_classes=self.n_actions_team1).astype('float32')
        action_team2_onehot = F.one_hot(action_team2.astype('int64'), 
                                       num_classes=self.n_actions_team2).astype('float32')
        
        # 拼接输入
        inputs = paddle.concat([obs, action_team1_onehot, action_team2_onehot], axis=-1)
        
        # MLP前向传播
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        # 检查NaN
        if paddle.isnan(q_value).any():
            print(f"\n[WARNING] Q网络输出NaN!")
            print(f"  obs: {obs[0] if len(obs) > 0 else 'empty'}")
            print(f"  inputs: {inputs[0] if len(inputs) > 0 else 'empty'}")
            print(f"  fc1.weight范围: [{float(self.fc1.weight.min()):.6f}, {float(self.fc1.weight.max()):.6f}]")
            print(f"  fc1输出: min={float(paddle.min(x)):.6f}, max={float(paddle.max(x)):.6f}")
        
        return q_value
    
    def evaluate_all_actions(self, obs, n_actions_team2=None):
        """
        评估所有动作组合的Q矩阵: Q^{i,k}(s, a, b)
        
        Args:
            obs: [batch, obs_shape]
            n_actions_team2: Team 2动作数(默认为self.n_actions_team2)
            
        Returns:
            q_matrix: [batch, n_actions_team1, n_actions_team2]
        """
        if n_actions_team2 is None:
            n_actions_team2 = self.n_actions_team2
        
        batch_size = obs.shape[0]
        q_matrix = paddle.zeros((batch_size, self.n_actions_team1, n_actions_team2), 
                               dtype='float32')
        
        for a in range(self.n_actions_team1):
            for b in range(n_actions_team2):
                action_team1 = paddle.full((batch_size,), a, dtype='int64')
                action_team2 = paddle.full((batch_size,), b, dtype='int64')
                q_value = self.forward(obs, action_team1, action_team2)
                q_matrix[:, a, b] = q_value.squeeze(-1)
        
        return q_matrix


# ============================================================================
# 2. 矩阵博弈求解器 (纳什均衡)
# ============================================================================

class MatrixGameNashSolver:
    """
    求解矩阵博弈的纳什均衡
    max_{π'∈P(A)} min_{σ'∈P(B)} E_{π',σ'}[Q^{i,k}_k(s_{t+1}, a, b)]
    
    优化:
    1. 使用'highs-ipm'更快的求解器
    2. 缓存重复Q矩阵的Nash解
    """
    
    def __init__(self, method='linprog', cache_size=10000):
        self.method = method
        # Nash解缓存 (key: Q矩阵hash, value: (pi, sigma, value))
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _hash_q_matrix(self, q_matrix, precision=4):
        """将Q矩阵转为hash键 (保疙51e-4精度)"""
        # 四舍五入到指定精度后转为tuple
        rounded = np.round(q_matrix, decimals=precision)
        return tuple(rounded.flatten())
    
    def solve_nash_equilibrium(self, q_matrix):
        """
        求解纳什均衡策略 (带缓存)
        
        Args:
            q_matrix: [n_actions_A, n_actions_B] Q矩阵
        
        Returns:
            pi_nash: [n_actions_A] Team 1的均衡策略 π'_k
            sigma_nash: [n_actions_B] Team 2的均衡策略 σ'_k  
            nash_value: float 均衡值
            solve_time: float 求解时间
        """
        start_time = time.time()
        
        # 尝试从缓存中获取
        cache_key = self._hash_q_matrix(q_matrix)
        if cache_key in self.cache:
            pi_nash, sigma_nash, nash_value = self.cache[cache_key]
            self.cache_hits += 1
            solve_time = time.time() - start_time  # 缓存命中时间极短
            return pi_nash.copy(), sigma_nash.copy(), nash_value, solve_time
        
        self.cache_misses += 1
        
        # 实际求解
        pi_nash, nash_value = self._solve_maximin_lp(q_matrix)
        sigma_nash, _ = self._solve_minimax_lp(q_matrix)
        
        # 存入缓存 (限制缓存大小)
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = (pi_nash.copy(), sigma_nash.copy(), nash_value)
        
        solve_time = time.time() - start_time
        return pi_nash, sigma_nash, nash_value, solve_time
    
    def _solve_maximin_lp(self, q_matrix):
        """线性规划求解Team 1的maximin策略 (优化: 使用highs-ipm求解器)"""
        n_actions_a, n_actions_b = q_matrix.shape
        
        # 目标函数: min -v
        c = np.zeros(n_actions_a + 1)
        c[-1] = -1
        
        # 不等式约束: -Q^T * π + v * 1 <= 0
        A_ub = np.zeros((n_actions_b, n_actions_a + 1))
        A_ub[:, :n_actions_a] = -q_matrix.T
        A_ub[:, -1] = np.ones(n_actions_b)
        b_ub = np.zeros(n_actions_b)
        
        # 等式约束: Σ π(a) = 1
        A_eq = np.zeros((1, n_actions_a + 1))
        A_eq[0, :n_actions_a] = np.ones(n_actions_a)
        b_eq = np.array([1.0])
        
        # 变量边界: π(a) >= 0
        bounds = [(0, None) for _ in range(n_actions_a)] + [(None, None)]
        
        try:
            # 使用highs-ipm内点法(比highs更快)
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs-ipm')
        except:
            # 备用: highs
            try:
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                               bounds=bounds, method='highs')
            except:
                # 备用: 默认
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                               bounds=bounds)
        
        if not result.success:
            pi_nash = np.ones(n_actions_a) / n_actions_a
            nash_value = self.compute_expected_value(
                q_matrix, pi_nash, np.ones(n_actions_b) / n_actions_b
            )
        else:
            pi_nash = result.x[:n_actions_a]
            nash_value = -result.fun
            pi_nash = np.maximum(pi_nash, 0)
            pi_nash = pi_nash / (pi_nash.sum() + 1e-10)
        
        return pi_nash, nash_value
    
    def _solve_minimax_lp(self, q_matrix):
        """线性规划求解Team 2的minimax策略 (优化: 使用highs-ipm求解器)"""
        n_actions_a, n_actions_b = q_matrix.shape
        
        c = np.zeros(n_actions_b + 1)
        c[-1] = 1
        
        A_ub = np.zeros((n_actions_a, n_actions_b + 1))
        A_ub[:, :n_actions_b] = q_matrix
        A_ub[:, -1] = -np.ones(n_actions_a)
        b_ub = np.zeros(n_actions_a)
        
        A_eq = np.zeros((1, n_actions_b + 1))
        A_eq[0, :n_actions_b] = np.ones(n_actions_b)
        b_eq = np.array([1.0])
        
        bounds = [(0, None) for _ in range(n_actions_b)] + [(None, None)]
        
        try:
            # 使用highs-ipm内点法(比highs更快)
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs-ipm')
        except:
            # 备用: highs
            try:
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                               bounds=bounds, method='highs')
            except:
                # 备用: 默认
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                               bounds=bounds)
        
        if not result.success:
            sigma_nash = np.ones(n_actions_b) / n_actions_b
            minimax_value = 0
        else:
            sigma_nash = result.x[:n_actions_b]
            minimax_value = result.fun
            sigma_nash = np.maximum(sigma_nash, 0)
            sigma_nash = sigma_nash / (sigma_nash.sum() + 1e-10)
        
        return sigma_nash, minimax_value
    
    def compute_expected_value(self, q_matrix, pi, sigma):
        """计算期望Q值: E[Q] = π^T * Q * σ"""
        return float(np.dot(pi, np.dot(q_matrix, sigma)))
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


# ============================================================================
# 3. 主算法: Decentralized Fitted Q-Iteration
# ============================================================================

class DecentralizedFQIAlgorithm:
    """
    去中心化拟合Q迭代算法
    
    Algorithm 1: Decentralized Fitted Q-Iteration Algorithm
    
    Input:
        - Function class H
        - Trajectory data D = {(s_t, {a^i_t}, {b^j_t}, s_{t+1})}
        - Number of iterations K
        - Initial estimator vectors Q^{i,0}_0
    
    Output:
        - Vector of estimates Q^1_K = [Q^{i,1}_K]_{i∈N}
        - Joint equilibrium policy π_K = E^1(Q^1_K)
    """
    
    def __init__(self, config):
        self.config = config
        self.n_agents = config['n_agents']
        self.n_actions = config['n_actions']
        self.n_actions_team2 = config['n_actions_team2']
        self.obs_shape = config['obs_shape']
        self.K_iterations = config['K_iterations']
        self.gamma = config['gamma']
        self.initial_lr = config['lr']  # 保存初始学习率
        self.lr = config['lr']  # 当前学习率
        self.lr_decay = config.get('lr_decay', 1.0)  # 学习率衰减系数
        self.min_lr = config.get('min_lr', 1e-5)  # 最小学习率
        self.batch_size = config['batch_size']
        self.fit_epochs = config['fit_epochs_per_iteration']
        self.clip_grad_norm = config.get('clip_grad_norm', None)  # 梯度裁剪
        
        # 初始化Q网络 (每个智能体一个,矩阵博弈用简化MLP)
        self.q_networks = []
        for i in range(self.n_agents):
            q_net = SimpleQNetwork(
                obs_shape=self.obs_shape,
                n_actions_team1=self.n_actions,
                n_actions_team2=self.n_actions_team2,
                hidden_dim=config['hidden_dim']
            )
            self.q_networks.append(q_net)
        
        # 创建共识矩阵
        self.consensus_matrix = ConsensusMatrix(
            n_agents=self.n_agents,
            matrix_type=config.get('consensus_matrix_type', 'metropolis')
        )
        
        logger.info(f"Consensus matrix is doubly stochastic: "
                   f"{self.consensus_matrix.is_doubly_stochastic()}")
        
        # 创建DIGing优化器
        parameters_list = [q_net.parameters() for q_net in self.q_networks]
        self.diging_optimizer = DIGingWrapper(
            parameters_list=parameters_list,
            consensus_matrix=self.consensus_matrix,
            learning_rate=self.lr
        )
        
        # 纳什均衡求解器
        self.nash_solver = MatrixGameNashSolver(method='linprog')
        
        # 迭代计数
        self.iteration_count = 0
        
        # 收敛性监控
        self.loss_history = []
        self.td_error_history = []  # 保留以保持兼容性，但不再绘图
        self.eval_reward_history = []  # Nash Distance历史
        self.value_error_history = []  # 【新增】Value Error历史
        self.policy_entropy_history = []  # 【新增】Policy Entropy历史
    
    def run_iteration(self, trajectory_data):
        """
        执行一次拟合Q迭代 (Algorithm 1的主循环内容)
        
        for k = 0, 1, 2, ..., K-1 do:
            for agent i ∈ N in Team 1 do:
                1. Solve matrix game to get (π'_k, σ'_k)
                2. Sample r^{1,i}_t ~ R^{1,i}(·| s_t, a_t, b_t)
                3. Compute local target Y^i_t
            end for
            4. Solve (4) by decentralized optimization (DIGing)
            5. Update estimate Q^{1,i}_{k+1}
        end for
        """
        logger.info(f"\n=== Fitted Q-Iteration {self.iteration_count + 1}/{self.K_iterations} ===")
        
        iteration_losses = []
        iteration_td_errors = []
        nash_solve_times = []
        
        # 步骤1-3: 对每个智能体计算目标
        all_targets = []
        
        for agent_id in range(self.n_agents):
            logger.info(f"Computing targets for agent {agent_id + 1}/{self.n_agents}...")
            
            # 计算该智能体的TD目标
            targets, nash_time = self._compute_targets_for_agent(
                trajectory_data, agent_id
            )
            
            all_targets.append(targets)
            nash_solve_times.append(nash_time)
        
        logger.info(f"Mean Nash solving time: {np.mean(nash_solve_times):.4f}s")
        
        # 显示缓存统计
        cache_stats = self.nash_solver.get_cache_stats()
        logger.info(f"Nash cache: {cache_stats['hit_rate']*100:.1f}% hit rate "
                   f"({cache_stats['hits']} hits, {cache_stats['misses']} misses, "
                   f"{cache_stats['cache_size']} cached)")
        
        # 步骤4: 使用DIGing分布式优化求解 (4)
        logger.info("Solving (4) by DIGing decentralized optimization...")
        
        loss, td_error = self._optimize_by_diging(
            trajectory_data, all_targets
        )
        
        iteration_losses.append(loss)
        iteration_td_errors.append(td_error)
        
        # 步骤5: 更新Q估计
        self.iteration_count += 1
        
        # 学习率衰减
        if self.lr_decay < 1.0:
            old_lr = self.lr
            self.lr = max(self.lr * self.lr_decay, self.min_lr)
            if self.iteration_count % 10 == 0:
                logger.info(f"Learning rate decayed: {old_lr:.6f} -> {self.lr:.6f}")
            # 更新DIGing优化器中的学习率
            self.diging_optimizer.learning_rate = self.lr
            self.diging_optimizer.diging.alpha = self.lr
        
        # 记录历史用于收敛性分析
        self.loss_history.append(loss)
        self.td_error_history.append(td_error)
        
        # 计算共识误差
        consensus_error = self.diging_optimizer.get_consensus_error()
        
        logger.info(f"Iteration {self.iteration_count} completed: "
                   f"loss={loss:.4f}, td_error={td_error:.4f}, "
                   f"consensus_error={consensus_error:.6f}")
        
        # 收敛性检测
        if self.iteration_count >= 5:
            self._check_convergence()
        
        return {
            'iteration': self.iteration_count,
            'mean_loss': np.mean(iteration_losses),
            'mean_td_error': np.mean(iteration_td_errors),
            'consensus_error': consensus_error,
            'nash_solve_time': np.mean(nash_solve_times)
        }
    
    def _compute_targets_for_agent(self, trajectory_data, agent_id):
        """
        计算智能体i的TD目标
        
        Y^i_t = r^{1,i}_t + γ · E_{π'_k,σ'_k}[Q^{i,k}_k(s_{t+1}, a, b)]
        """
        n_transitions = len(trajectory_data['rewards'])
        targets = np.zeros(n_transitions, dtype=np.float32)
        
        q_network = self.q_networks[agent_id]
        
        total_nash_time = 0
        
        for t in range(n_transitions):
            reward = trajectory_data['rewards'][t, agent_id]
            terminated = trajectory_data['terminated'][t]
            
            if terminated:
                targets[t] = reward
                continue
            
            # 获取下一状态
            next_obs = trajectory_data['next_observations'][t, agent_id]
            next_obs_tensor = paddle.to_tensor(next_obs[np.newaxis, :], dtype='float32')
            
            with paddle.no_grad():
                # 评估Q矩阵: Q^{i,k}_k(s_{t+1}, a, b) (矩阵博弈不需要hidden_state)
                q_matrix_next = q_network.evaluate_all_actions(next_obs_tensor)
                q_matrix_next = q_matrix_next.squeeze(0).numpy()
                
                # 【原方法：求解纳什均衡策略】（注释掉）
                # pi_nash, sigma_nash, nash_value, solve_time = \
                #     self.nash_solver.solve_nash_equilibrium(q_matrix_next)
                # total_nash_time += solve_time
                # expected_q = self.nash_solver.compute_expected_value(
                #     q_matrix_next, pi_nash, sigma_nash
                # )
                
                # 【新方法：使用Softmax策略计算期望Q值】
                # 对每个自己的动作，计算对对手动作的期望Q值
                q_expected_team1 = np.mean(q_matrix_next, axis=1)  # [n_actions_team1]
                q_expected_team2 = np.mean(q_matrix_next, axis=0)  # [n_actions_team2]
                
                # 使用softmax转换为概率分布
                temperature = 0.5  # 降低温度，让策略对Q值差异更敏感
                q_exp_t1 = np.exp((q_expected_team1 - np.max(q_expected_team1)) / temperature)
                pi_softmax = q_exp_t1 / np.sum(q_exp_t1)
                
                q_exp_t2 = np.exp((q_expected_team2 - np.max(q_expected_team2)) / temperature)
                sigma_softmax = q_exp_t2 / np.sum(q_exp_t2)
                
                # 计算期望: E_{π,σ}[Q^{i,k}_k(s_{t+1}, a, b)]
                expected_q = float(np.dot(pi_softmax, np.dot(q_matrix_next, sigma_softmax)))
                
                # TD目标: Y^i_t = r^{1,i}_t + γ · E[...]
                targets[t] = float(reward + self.gamma * expected_q)
        
        avg_nash_time = total_nash_time / max(n_transitions, 1)
        
        # 【新增】验证Target期望是否接近0（零和博弈理论要求）
        target_mean = np.mean(targets)
        target_std = np.std(targets)
        
        # 调试信息(仅Agent 0)
        if agent_id == 0:
            logger.info(f"  [Agent {agent_id} Target计算调试]")
            logger.info(f"    【重要】Target期望验证: mean={target_mean:.6f} (理论应接近0)")
            logger.info(f"    Target标准差: {target_std:.6f}")
            logger.info(f"    总样本数: {n_transitions}")
            logger.info(f"    Terminated样本: {np.sum(trajectory_data['terminated'])} ({100*np.mean(trajectory_data['terminated']):.1f}%)")
            logger.info(f"    Target统计: mean={np.mean(targets):.4f}, "
                       f"std={np.std(targets):.4f}, "
                       f"min={np.min(targets):.4f}, max={np.max(targets):.4f}")
            
            # 【新增】分析Target的组成
            terminated_targets = targets[trajectory_data['terminated']]
            non_terminated_targets = targets[~trajectory_data['terminated']]
            logger.info(f"    Terminated样本Target: mean={np.mean(terminated_targets):.4f} (应该=reward)")
            logger.info(f"    非Terminated样本Target: mean={np.mean(non_terminated_targets):.4f} (=reward+γ*Q_next)")
            
            # 【新增】理论期望检查
            logger.info(f"    【重要】理论期望: Target应该接近0 (零和博弈)")
            logger.info(f"    【重要】实际偏差: {np.mean(targets):.4f}")
            
            # 显示前3个样本的详细信息
            logger.info(f"    前3个样本详情:")
            for i in range(min(3, len(targets))):
                logger.info(f"      样本{i}: reward={trajectory_data['rewards'][i, agent_id]:.2f}, "
                           f"target={targets[i]:.4f}, terminated={trajectory_data['terminated'][i]}")
            
            # 【新增】按状态-动作对分组，看每个(s,a,b)的平均Target
            logger.info(f"    【状态-动作对的平均Target】(Agent 0)")
            obs_data = trajectory_data['observations'][:, agent_id]  # [N, obs_dim]
            actions_team1 = trajectory_data['team1_actions'][:, agent_id]  # [N]
            actions_team2 = trajectory_data['team2_actions'][:, agent_id]  # [N]
            
            # 对于第一阶段state=[1,0,0,0,0]的样本
            first_stage_mask = (obs_data[:, 0] == 1.0)
            if np.sum(first_stage_mask) > 0:
                logger.info(f"      第一阶段[1,0,0,0,0]:")
                for a in range(2):
                    for b in range(2):
                        mask = first_stage_mask & (actions_team1 == a) & (actions_team2 == b)
                        if np.sum(mask) > 0:
                            avg_target = np.mean(targets[mask])
                            avg_reward = np.mean(trajectory_data['rewards'][mask, agent_id])
                            count = np.sum(mask)
                            logger.info(f"        动作({a},{b}): {count}个样本, 平均reward={avg_reward:.4f}, 平均Target={avg_target:.4f}")
            
            # 统计Target≠Reward的比例
            diff_ratio = np.sum(np.abs(targets - trajectory_data['rewards'][:, agent_id]) > 0.01) / len(targets)
            logger.info(f"    Target≠Reward的样本: {100*diff_ratio:.1f}% (这些样本有γ*next_Q项)")
        
        # 【Target Normalization已注释】
        # target_mean = np.mean(targets)
        # if abs(target_mean) > 0.001:
        #     logger.info(f"    【Target Normalization】检测到偏差{target_mean:.4f}, 进行中心化")
        #     targets = targets - target_mean
        #     logger.info(f"    中心化后Target均值: {np.mean(targets):.6f}")
        
        return targets, avg_nash_time
    
    def _optimize_by_diging(self, trajectory_data, all_targets):
        """
        使用DIGing算法求解优化问题 (4)
        
        Solve (4) for agents in Team 1, by decentralized optimization algorithms
        """
        n_transitions = len(trajectory_data['rewards'])
        indices = list(range(n_transitions))
        
        epoch_losses = []
        epoch_td_errors = []
        
        # 【调试】统计样本池信息
        logger.info(f"  [样本池统计]")
        logger.info(f"    总样本数: {n_transitions}")
        logger.info(f"    Batch大小: {self.batch_size}")
        logger.info(f"    每Epoch batch数: {(n_transitions + self.batch_size - 1) // self.batch_size}")
        logger.info(f"    训练Epochs: {self.fit_epochs}")
        
        # 统计样本分布
        rewards_all = trajectory_data['rewards'][:, 0]  # Agent 0的reward
        logger.info(f"    Agent0 Reward: mean={np.mean(rewards_all):.4f}, "
                   f"std={np.std(rewards_all):.4f}, "
                   f"min={np.min(rewards_all):.4f}, max={np.max(rewards_all):.4f}")
        
        # 统计terminated分布
        n_terminated = np.sum(trajectory_data['terminated'])
        logger.info(f"    Terminated样本: {n_terminated}/{n_transitions} ({100*n_terminated/n_transitions:.1f}%)")
        
        for epoch in range(self.fit_epochs):
            np.random.shuffle(indices)
            epoch_batch_losses = []  # 记录本epoch每个batch的loss
            
            # 【调试】每个epoch开始时,统计前10个采样索引
            if epoch == 0:
                logger.info(f"\n  [Epoch {epoch+1} 采样]")
                logger.info(f"    前10个采样索引: {indices[:10]}")
                logger.info(f"    对应rewards: {[f'{rewards_all[i]:.2f}' for i in indices[:10]]}")
            
            for start_idx in range(0, n_transitions, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_transitions)
                batch_indices = indices[start_idx:end_idx]
                
                # 清除梯度
                self.diging_optimizer.clear_grad()
                
                # 计算每个智能体的损失并反向传播
                batch_losses = []
                batch_td_errors = []
                
                for agent_id in range(self.n_agents):
                    loss, td_error = self._compute_agent_loss(
                        trajectory_data, all_targets[agent_id], 
                        agent_id, batch_indices
                    )
                    
                    # 反向传播计算梯度
                    loss.backward()
                    
                    batch_losses.append(float(loss))
                    batch_td_errors.append(float(td_error))
                
                # 关键:梯度裁剪防止NaN!
                if hasattr(self, 'clip_grad_norm') and self.clip_grad_norm is not None:
                    # 检查是否有NaN/Inf梯度
                    has_nan = False
                    for agent_id in range(self.n_agents):
                        for param in self.q_networks[agent_id].parameters():
                            if param.grad is not None:
                                if paddle.isnan(param.grad).any() or paddle.isinf(param.grad).any():
                                    has_nan = True
                                    logger.error(f"Agent {agent_id} 梯度出现NaN/Inf!")
                                    break
                        if has_nan:
                            break
                    
                    if has_nan:
                        # 跳过本次更新
                        logger.error(f"\n[CRITICAL] 梯度NaN/Inf, 跳过本次更新!")
                        self.diging_optimizer.clear_grad()
                        continue
                    
                    # 正常裁剪
                    for agent_id in range(self.n_agents):
                        paddle.nn.utils.clip_grad_norm_(
                            self.q_networks[agent_id].parameters(),
                            max_norm=self.clip_grad_norm
                        )
                
                # DIGing更新步骤
                self.diging_optimizer.step()
                
                batch_loss_mean = np.mean(batch_losses)
                epoch_losses.append(batch_loss_mean)
                epoch_td_errors.append(np.mean(batch_td_errors))
                epoch_batch_losses.append(batch_loss_mean)
            
            # 【调试】每个epoch结束统计
            if epoch == 0 or epoch == self.fit_epochs - 1:  # 第一个和最后一个epoch
                logger.info(f"  [Epoch {epoch+1} 完成]")
                logger.info(f"    本Epoch Loss: mean={np.mean(epoch_batch_losses):.6f}, "
                           f"std={np.std(epoch_batch_losses):.6f}, "
                           f"min={np.min(epoch_batch_losses):.6f}, "
                           f"max={np.max(epoch_batch_losses):.6f}")
        
        # 调试信息
        final_loss = np.mean(epoch_losses)
        final_td_error = np.mean(epoch_td_errors)
        logger.info(f"  [优化结果]")
        logger.info(f"    训练Epochs: {self.fit_epochs}")
        logger.info(f"    总更新次数: {len(epoch_losses)}")
        logger.info(f"    最终Loss: {final_loss:.6f} (std={np.std(epoch_losses):.6f})")
        logger.info(f"    最终TD_Error: {final_td_error:.6f} (std={np.std(epoch_td_errors):.6f})")
        
        return final_loss, final_td_error
    
    def _compute_agent_loss(self, trajectory_data, targets, agent_id, batch_indices):
        """计算单个智能体的损失"""
        q_network = self.q_networks[agent_id]
        
        obs_batch = trajectory_data['observations'][batch_indices, agent_id]
        
        # 根据数据存储逻辑:
        # all_actions_team1 = [Team1动作(3), Team2动作(3)]
        # all_actions_team2 = [Team2动作(3), Team1动作(3)]
        # 对Team1: team1_actions[自己], team2_actions[0] (对手)
        # 对Team2: team1_actions[自己], team2_actions[自己] (对应Team1对手)
        team1_actions_batch = trajectory_data['team1_actions'][batch_indices, agent_id]
        team2_actions_batch = trajectory_data['team2_actions'][batch_indices, agent_id]  # 都用agent_id!
        
        targets_batch = targets[batch_indices]
        
        obs_tensor = paddle.to_tensor(obs_batch, dtype='float32')
        team1_actions_tensor = paddle.to_tensor(team1_actions_batch, dtype='int64')
        team2_actions_tensor = paddle.to_tensor(team2_actions_batch, dtype='int64')
        targets_tensor = paddle.to_tensor(targets_batch, dtype='float32')
        
        # 矩阵博弈不需要hidden_state
        q_pred = q_network.forward(
            obs_tensor, team1_actions_tensor, team2_actions_tensor
        )
        
        q_pred = q_pred.squeeze(-1)
        td_error = targets_tensor - q_pred
        loss = paddle.mean(td_error ** 2)
        
        # 【调试】显示Agent 0的前3个样本的Q预测和Target
        if agent_id == 0 and len(batch_indices) <= 5:  # 只在小batch时输出
            logger.info(f"    [Agent 0 Batch详情] batch_size={len(batch_indices)}")
            for i in range(min(3, len(batch_indices))):
                sample_idx = batch_indices[i]
                obs = obs_batch[i]
                a1 = int(team1_actions_batch[i])
                a2 = int(team2_actions_batch[i])
                q_val = float(q_pred[i])
                target_val = float(targets_tensor[i])
                td_val = float(td_error[i])
                reward = trajectory_data['rewards'][sample_idx, agent_id]
                terminated = trajectory_data['terminated'][sample_idx]
                
                logger.info(f"      样本{sample_idx}: obs_state={obs[3:].tolist()[:3]}..., "
                           f"action=({a1},{a2}), "
                           f"reward={reward:.2f}, terminated={terminated}")
                logger.info(f"        Q_pred={q_val:.4f}, Target={target_val:.4f}, TD_error={td_val:.4f}")
        
        return loss, paddle.mean(paddle.abs(td_error))
    
    def _check_convergence(self):
        """检测训练是否在有效收敛"""
        k = self.iteration_count
        current_loss = self.loss_history[-1]
        current_td = self.td_error_history[-1]
        
        # 检查最近5次的趋势
        if k >= 5:
            recent_losses = self.loss_history[-5:]
            loss_decrease = recent_losses[0] - recent_losses[-1]
            loss_std = np.std(recent_losses)
            
            # 调试信息 - 已注释
            # logger.info(f"  [收敛检测] Loss趋势: {loss_decrease:+.4f}, 波动std: {loss_std:.4f}")
            
            # 判断收敛状态 - 已注释
            # if current_loss < 0.5 and current_td < 0.2:
            #     logger.info(f"  ✓ 收敛良好: Loss={current_loss:.4f}, TD={current_td:.4f}")
            # elif current_loss > 5.0:
            #     logger.warning(f"  ✗ Loss过高({current_loss:.2f}), 可能需要更多样本或调整超参")
            # elif loss_std < 0.01 and k >= 30:
            #     logger.info(f"  ✓ Loss已稳定(std={loss_std:.4f}), 可能已收敛")
            # elif loss_decrease < 0 and k >= 30:
            #     logger.warning(f"  ⚠ Loss未下降, 检查学习率或数据质量")
    
    def extract_policy(self, obs, agent_id, available_actions=None):
        """
        提取策略 (使用纳什均衡)
        Joint equilibrium policy π_K = E^1(Q^1_K)
        """
        q_network = self.q_networks[agent_id]
        
        obs_tensor = paddle.to_tensor(obs[np.newaxis, :], dtype='float32')
        
        with paddle.no_grad():
            # 矩阵博弈不需要hidden_state
            q_matrix = q_network.evaluate_all_actions(obs_tensor)
            q_matrix = q_matrix.squeeze(0).numpy()
            
            # 应用动作mask
            if available_actions is not None:
                for a in range(len(available_actions)):
                    if available_actions[a] == 0:
                        q_matrix[a, :] = -1e10
            
            # 【原方法：通过Nash均衡求解策略】（注释掉）
            # pi_nash, sigma_nash, nash_value, _ = \
            #     self.nash_solver.solve_nash_equilibrium(q_matrix)
            
            # 【新方法：直接从Q值生成策略】
            q_expected = np.mean(q_matrix, axis=1)  # [n_actions]
            temperature = 0.5  # 降低温度，让策略对Q值差异更敏感
            q_exp = np.exp((q_expected - np.max(q_expected)) / temperature)
            pi_nash = q_exp / np.sum(q_exp)
            
            # 从均衡策略中采样动作
            action = int(np.argmax(pi_nash))
        
        return action
    
    def save_models(self, save_dir, iteration=None):
        """保存模型和训练历史"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存Q网络
        for i, q_network in enumerate(self.q_networks):
            if iteration is not None:
                model_path = os.path.join(save_dir, f'agent_{i}_iter_{iteration}.pdparams')
            else:
                model_path = os.path.join(save_dir, f'agent_{i}_final.pdparams')
            paddle.save(q_network.state_dict(), model_path)
        
        # 保存训练历史
        import pickle
        history = {
            'loss_history': self.loss_history,
            'td_error_history': self.td_error_history,
            'eval_reward_history': self.eval_reward_history,
            'iteration_count': self.iteration_count
        }
        history_path = os.path.join(save_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir, iteration=None):
        """加载模型"""
        for i, q_network in enumerate(self.q_networks):
            if iteration is not None:
                model_path = os.path.join(save_dir, f'agent_{i}_iter_{iteration}.pdparams')
            else:
                model_path = os.path.join(save_dir, f'agent_{i}_final.pdparams')
            
            if os.path.exists(model_path):
                q_network.set_state_dict(paddle.load(model_path))
                logger.info(f"Loaded model for agent {i} from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
