#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
环境模型 MG: P(s'|s,a,b), R(s,a,b,s')
用于MDVI算法的Planning阶段
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class TransitionNetwork(nn.Layer):
    """用神经网络拟合状态转移 P(s'|s,a,b)
    
    对于Shapley两阶段博弈:
    - 第一阶段: (s0, a1, b1) -> s1 (s1包含a1,b1的编码)
    - 第二阶段: (s1, a2, b2) -> s0 (重置)
    
    注意: 这里的state是全局状态,不是agent的局部观测obs!
    """
    
    def __init__(self, state_shape, n_actions_team1, n_actions_team2, hidden_dim=64):
        super().__init__()
        
        self.state_shape = state_shape
        self.n_actions_team1 = n_actions_team1
        self.n_actions_team2 = n_actions_team2
        
        # 输入: state + a1_onehot + b1_onehot
        input_dim = state_shape + n_actions_team1 + n_actions_team2
        
        # 输出: next_state
        output_dim = state_shape
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, state, action_team1, action_team2):
        """
        Args:
            state: [batch, state_dim]
            action_team1: [batch, n_actions_team1] one-hot
            action_team2: [batch, n_actions_team2] one-hot
        Returns:
            next_state: [batch, state_dim]
        """
        x = paddle.concat([state, action_team1, action_team2], axis=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state = self.fc3(x)  # 不用激活,直接输出state向量
        return next_state


class RewardNetwork(nn.Layer):
    """用神经网络拟合奖励函数 R(all_states, all_current_actions_team1, all_current_actions_team2)
    
    关键设计:
    - all_states: 所有对的状态(拼接)
    - all_actions: 当前两队所有动作
    - 支持cooperation_weight混合奖励
    - 网络内部不区分own_state,直接处理all_states
    """
    
    def __init__(self, state_shape, n_actions_team1, n_actions_team2, n_agents_team1=3, n_agents_team2=3, hidden_dim=256):
        super().__init__()
        
        self.n_agents_team1 = n_agents_team1
        self.n_agents_team2 = n_agents_team2
        
        # 输入: all_states + all_current_actions_onehot
        input_dim = (state_shape * 6) + (n_actions_team1 * n_agents_team1) + (n_actions_team2 * n_agents_team2)
        
        # 3层网络,增加拟合能力
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # 输出标量奖励
        
        # 使用Xavier初始化,改善训练
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.initializer.XavierUniform()(layer.weight)
            nn.initializer.Constant(0.0)(layer.bias)
        
    def forward(self, all_states, all_actions_team1, all_actions_team2):
        """
        Args:
            all_states: [batch, state_dim*6] 所有对的状态拼接
            all_actions_team1: [batch, n_agents_team1 * n_actions_team1] 当前阶段team1所有动作
            all_actions_team2: [batch, n_agents_team2 * n_actions_team2] 当前阶段team2所有动作
        Returns:
            reward: [batch, 1]
        """
        x = paddle.concat([all_states, all_actions_team1, all_actions_team2], axis=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        reward = self.fc3(x)
        return reward


class EnvironmentModel(nn.Layer):
    """
    环境模型: MG = (S, A, B, P, R1, R2, γ)
    
    包括:
    1. 状态转移模型: P(s'|s,a,b) - 用神经网络拟合
    2. 奖励模型: R(s,a,b) - 用神经网络拟合
    """
    
    def __init__(self, obs_shape, state_shape, n_actions_team1, n_actions_team2, hidden_dim=64):
        super().__init__()
        
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.n_actions_team1 = n_actions_team1
        self.n_actions_team2 = n_actions_team2
        
        # 1. 状态转移模型 (新增!)
        self.transition_network = TransitionNetwork(
            state_shape, n_actions_team1, n_actions_team2, hidden_dim=hidden_dim
        )
        
        # 2. 奖励模型 (每个智能体一个)
        # 关键: 每个网络输入所有智能体的联合动作
        self.reward_networks = []
        n_agents_team1 = 3  # 硬编码,与环境配置一致
        n_agents_team2 = 3
        for i in range(6):  # 6个智能体
            self.reward_networks.append(
                RewardNetwork(state_shape, n_actions_team1, n_actions_team2, 
                            n_agents_team1=n_agents_team1, n_agents_team2=n_agents_team2,
                            hidden_dim=256)
            )
        
        # 收集训练数据 - 关键改进: 存储所有智能体的状态信息
        self.training_data = {i: {'all_states': [],       # 所有智能体的states(拼接)
                                  'all_actions_team1': [], 
                                  'all_actions_team2': [], 
                                  'rewards': []} 
                             for i in range(6)}
        self.transition_data = {'states': [], 'actions_team1': [], 'actions_team2': [], 'next_states': []}
        
        # 查表统计
        self.lookup_hits = 0  # 查表命中次数
        self.lookup_total = 0  # 总查询次数
        
    def update_from_samples(self, trajectory_data):
        """
        从样本中收集数据并训练奖励网络和状态转移网络
        
        Args:
            trajectory_data: dict with keys:
                - observations: [N, n_agents, obs_dim]
                - next_observations: [N, n_agents, obs_dim]
                - team1_actions: [N, n_agents]
                - team2_actions: [N, n_agents]
                - rewards: [N, n_agents]
        """
        n_samples = len(trajectory_data['rewards'])
        
        # 收集状态转移数据
        # 对于Shapley两阶段博弈: state = [stage, a1_s1_onehot, a2_s1_onehot]
        # 需要从terminated和actions推断状态
        for i in range(n_samples):
            if trajectory_data['terminated'][i]:
                # 第二阶段样本: state包含第一阶段的动作编码
                # 但我们无法从buffer中知道第一阶段动作,只能用当前动作近似
                state = np.zeros(self.state_shape, dtype=np.float32)
                state[0] = 0.0  # stage=0 (第二阶段)
                # 用当前动作编码子博弈(这是近似)
                a1 = int(trajectory_data['team1_actions'][i, 0])
                a2 = int(trajectory_data['team2_actions'][i, 0])
                if self.state_shape > 1:
                    state[1 + a1] = 1.0
                    state[1 + self.n_actions_team1 + a2] = 1.0
                
                # Next state: 终止,返回初始状态
                next_state = np.zeros(self.state_shape, dtype=np.float32)
                next_state[0] = 1.0  # 回到第一阶段(为下一个episode准备)
            else:
                # 第一阶段样本: state = [1, 0, 0, 0, 0]
                state = np.zeros(self.state_shape, dtype=np.float32)
                state[0] = 1.0  # stage=1
                
                # Next state: 转移到第二阶段,编码第一阶段的动作
                next_state = np.zeros(self.state_shape, dtype=np.float32)
                next_state[0] = 0.0  # stage=0
                a1 = int(trajectory_data['team1_actions'][i, 0])
                a2 = int(trajectory_data['team2_actions'][i, 0])
                if self.state_shape > 1:
                    next_state[1 + a1] = 1.0
                    next_state[1 + self.n_actions_team1 + a2] = 1.0
            
            self.transition_data['states'].append(state)
            self.transition_data['actions_team1'].append(int(trajectory_data['team1_actions'][i, 0]))
            self.transition_data['actions_team2'].append(int(trajectory_data['team2_actions'][i, 0]))
            self.transition_data['next_states'].append(next_state)
        
        # 收集奖励数据 - 每个智能体记录自己的state + 所有对的states
        print(f"\n[数据收集] 多子博弈设计(支持团队奖励):")
        print(f"  每个智能体对有自己的子博弈状态")
        print(f"  输入: 自己的state + 所有对的states + 当前两队所有动作 → reward")
        print(f"  这样可以支持cooperation_weight混合奖励\n")
        
        for i in range(n_samples):
            # 提取所有智能体的state(从各自的observation)
            # obs = [state(5), teammate_flags(2)] = 7维，只提取前5维state
            all_states = []
            for agent_id in range(6):
                agent_obs = trajectory_data['observations'][i, agent_id]
                # 只取state部分，去掉teammate_flags
                state = agent_obs[:self.state_shape]  # 取前5维: [stage, a1_onehot(2), a2_onehot(2)]
                all_states.append(state)
            
            # 构造所有智能体的联合动作 (当前阶段)
            all_actions_team1_indices = trajectory_data['team1_actions'][i, :3]
            all_actions_team2_indices = trajectory_data['team2_actions'][i, :3]
            
            # 转换为one-hot并拼接
            all_actions_team1_onehot = []
            for agent_idx in range(3):
                onehot = np.zeros(self.n_actions_team1, dtype=np.float32)
                onehot[int(all_actions_team1_indices[agent_idx])] = 1.0
                all_actions_team1_onehot.append(onehot)
            all_actions_team1 = np.concatenate(all_actions_team1_onehot)
            
            all_actions_team2_onehot = []
            for agent_idx in range(3):
                onehot = np.zeros(self.n_actions_team2, dtype=np.float32)
                onehot[int(all_actions_team2_indices[agent_idx])] = 1.0
                all_actions_team2_onehot.append(onehot)
            all_actions_team2 = np.concatenate(all_actions_team2_onehot)
            
            # 存储每个智能体的训练数据
            for agent_id in range(6):
                reward = float(trajectory_data['rewards'][i, agent_id])
                
                self.training_data[agent_id]['all_states'].append(np.concatenate(all_states))  # 所有state拼接
                self.training_data[agent_id]['all_actions_team1'].append(all_actions_team1)
                self.training_data[agent_id]['all_actions_team2'].append(all_actions_team2)
                self.training_data[agent_id]['rewards'].append(reward)
        
        # 训练状态转移网络
        print("\n[训练状态转移模型]")
        self._train_transition_network(epochs=50, lr=0.01)
        
        # 训练每个智能体的奖励网络
        print("\n[训练奖励模型]")
        for agent_id in range(6):
            self._train_reward_network(agent_id, epochs=100, lr=0.001)  # 降低学习率,增加epochs
    
    
    def _train_transition_network(self, epochs=50, lr=0.01):
        """训练状态转移网络"""
        data = self.transition_data
        if len(data['states']) == 0:
            print("  警告: 没有状态转移数据!")
            return
        
        network = self.transition_network
        optimizer = paddle.optimizer.Adam(parameters=network.parameters(), learning_rate=lr)
        
        # 准备数据
        states = np.array(data['states'], dtype=np.float32)
        actions_team1 = np.array(data['actions_team1'], dtype=np.int64)
        actions_team2 = np.array(data['actions_team2'], dtype=np.int64)
        next_states = np.array(data['next_states'], dtype=np.float32)
        
        # One-hot编码
        actions_team1_onehot = np.eye(self.n_actions_team1)[actions_team1]
        actions_team2_onehot = np.eye(self.n_actions_team2)[actions_team2]
        
        dataset_size = len(states)
        batch_size = min(64, dataset_size)
        
        for epoch in range(epochs):
            indices = np.random.permutation(dataset_size)
            epoch_loss = 0
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                s_batch = paddle.to_tensor(states[batch_indices], dtype='float32')
                a1_batch = paddle.to_tensor(actions_team1_onehot[batch_indices], dtype='float32')
                a2_batch = paddle.to_tensor(actions_team2_onehot[batch_indices], dtype='float32')
                ns_batch = paddle.to_tensor(next_states[batch_indices], dtype='float32')
                
                pred_next_state = network(s_batch, a1_batch, a2_batch)
                loss = F.mse_loss(pred_next_state, ns_batch)
                
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                
                epoch_loss += float(loss)
            
            # 每10个epoch打印一次
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / (dataset_size / batch_size)
                print(f"  Transition Epoch {epoch+1}: Loss={avg_loss:.6f}")
        
        # 训练完成后打印最终效果
        with paddle.no_grad():
            s_all = paddle.to_tensor(states, dtype='float32')
            a1_all = paddle.to_tensor(actions_team1_onehot, dtype='float32')
            a2_all = paddle.to_tensor(actions_team2_onehot, dtype='float32')
            pred_all = network(s_all, a1_all, a2_all).numpy()
        
        mse = np.mean((pred_all - next_states)**2)
        print(f"\n  === Transition模型训练完成 ===")
        print(f"    总体MSE: {mse:.6f}")
        print(f"    预测范围: [{pred_all.min():.4f}, {pred_all.max():.4f}]")
        print(f"    真实范围: [{next_states.min():.4f}, {next_states.max():.4f}]")
        
        # 详细分析:检查每个维度的预测
        if self.state_shape <= 5:  # 只在state维度较小时打印详细信息
            print(f"    各维度MSE:")
            for dim in range(self.state_shape):
                dim_mse = np.mean((pred_all[:, dim] - next_states[:, dim])**2)
                print(f"      维度{dim}: MSE={dim_mse:.6f}, 预测范围=[{pred_all[:, dim].min():.4f}, {pred_all[:, dim].max():.4f}]")
        
        # 采样展示几个预测示例
        print(f"    预测示例 (前5个):")
        for i in range(min(5, len(states))):
            stage_name = "第一阶段" if states[i][0] > 0.5 else "第二阶段"
            next_stage_name = "第二阶段" if next_states[i][0] < 0.5 else "第一阶段(重置)"
            print(f"      样本{i} ({stage_name} -> {next_stage_name}):")
            print(f"        s={states[i]}, a1={actions_team1[i]}, a2={actions_team2[i]}")
            print(f"        预测s'={pred_all[i]}, 真实s'={next_states[i]}")
    
    def _train_reward_network(self, agent_id, epochs=100, lr=0.001):
        """训练单个智能体的奖励网络 R(all_states, all_current_actions)"""
        data = self.training_data[agent_id]
        if len(data['rewards']) == 0:
            return
        
        network = self.reward_networks[agent_id]
        optimizer = paddle.optimizer.Adam(parameters=network.parameters(), learning_rate=lr)
        
        # 准备数据 - 只需要all_states
        all_states = np.array(data['all_states'], dtype=np.float32)  # 所有对的states
        all_actions_team1 = np.array(data['all_actions_team1'], dtype=np.float32)
        all_actions_team2 = np.array(data['all_actions_team2'], dtype=np.float32)
        rewards = np.array(data['rewards'], dtype=np.float32)
        
        dataset_size = len(rewards)
        batch_size = min(64, dataset_size)
        
        # 【关键改进】分层采样:将样本按奖励值分类
        extreme_indices = np.where((rewards < -1.5) | (rewards > 1.5))[0]
        normal_indices = np.where((rewards >= -1.5) & (rewards <= 1.5))[0]
        
        print(f"  Agent {agent_id} 数据分布: 极值样本{len(extreme_indices)}, 普通样本{len(normal_indices)}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # 【平衡采样】确保每个batch都包含极值和普通样本
            num_batches_total = max(len(extreme_indices), len(normal_indices)) // (batch_size // 2)
            
            # 打乱两类样本
            np.random.shuffle(extreme_indices)
            np.random.shuffle(normal_indices)
            
            for batch_idx in range(num_batches_total):
                # 每个batch: 一半极值样本 + 一半普通样本
                start_e = (batch_idx * (batch_size // 2)) % len(extreme_indices)
                end_e = min(start_e + batch_size // 2, len(extreme_indices))
                batch_extreme = extreme_indices[start_e:end_e]
                
                start_n = (batch_idx * (batch_size // 2)) % len(normal_indices)
                end_n = min(start_n + batch_size // 2, len(normal_indices))
                batch_normal = normal_indices[start_n:end_n]
                
                batch_indices = np.concatenate([batch_extreme, batch_normal])
                np.random.shuffle(batch_indices)  # 打乱batch内顺序
                
                if len(batch_indices) == 0:
                    continue
                
                all_s_batch = paddle.to_tensor(all_states[batch_indices], dtype='float32')
                a1_batch = paddle.to_tensor(all_actions_team1[batch_indices], dtype='float32')
                a2_batch = paddle.to_tensor(all_actions_team2[batch_indices], dtype='float32')
                r_batch = paddle.to_tensor(rewards[batch_indices].reshape(-1, 1), dtype='float32')
                
                pred_reward = network(all_s_batch, a1_batch, a2_batch)
                
                # 加权Loss: 给极值样本5倍权重
                sample_weights = paddle.ones_like(r_batch)
                extreme_mask = (paddle.abs(r_batch) > 1.5)
                sample_weights = paddle.where(extreme_mask, 
                                            paddle.to_tensor(5.0, dtype='float32'), 
                                            sample_weights)
                
                # 加权MSE - 修复: 应该是sum(权重*误差)/sum(权重)
                squared_error = (pred_reward - r_batch) ** 2
                weighted_squared_error = squared_error * sample_weights
                loss = paddle.sum(weighted_squared_error) / paddle.sum(sample_weights)
                
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                
                epoch_loss += float(loss)
                num_batches += 1
            
            # 每10个epoch打印一次
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"  Agent {agent_id} Epoch {epoch+1}: Loss={avg_loss:.6f}")
        
        # 训练完成后打印最终效果
        with paddle.no_grad():
            all_s_all = paddle.to_tensor(all_states, dtype='float32')
            a1_all = paddle.to_tensor(all_actions_team1, dtype='float32')
            a2_all = paddle.to_tensor(all_actions_team2, dtype='float32')
            pred_all = network(all_s_all, a1_all, a2_all).squeeze().numpy()
            
        print(f"  Agent {agent_id} 训练完成:")
        print(f"    预测范围: [{pred_all.min():.4f}, {pred_all.max():.4f}]")
        print(f"    真实范围: [{rewards.min():.4f}, {rewards.max():.4f}]")
        print(f"    MSE: {np.mean((pred_all - rewards)**2):.6f}")
        
        # 检查数据分布
        print(f"    数据统计: 样本数={len(rewards)}, unique值={len(np.unique(rewards))}")
        unique_rewards = np.unique(rewards)
        print(f"    出现的奖励值: {sorted(unique_rewards)[:10]}")
        
        # 统计极值出现频率
        n_extreme = np.sum((rewards < -1.5) | (rewards > 1.5))
        n_middle = np.sum((rewards >= -0.5) & (rewards <= 0.5))
        print(f"    极值样本(|r|>1.5): {n_extreme}/{len(rewards)} = {100*n_extreme/len(rewards):.1f}%")
        print(f"    中等样本(|r|<0.5): {n_middle}/{len(rewards)} = {100*n_middle/len(rewards):.1f}%")
        
        # 详细分析极值预测
        extreme_mask = (rewards < -1.5) | (rewards > 1.5)
        if np.sum(extreme_mask) > 0:
            extreme_true = rewards[extreme_mask]
            extreme_pred = pred_all[extreme_mask]
            extreme_mse = np.mean((extreme_pred - extreme_true)**2)
            print(f"    极值样本MSE: {extreme_mse:.6f}")
            print(f"    极值预测范围: [{extreme_pred.min():.4f}, {extreme_pred.max():.4f}]")
            print(f"    极值真实范围: [{extreme_true.min():.4f}, {extreme_true.max():.4f}]")
        
        # 分析普通样本
        normal_mask = (rewards >= -1.5) & (rewards <= 1.5)
        if np.sum(normal_mask) > 0:
            normal_true = rewards[normal_mask]
            normal_pred = pred_all[normal_mask]
            normal_mse = np.mean((normal_pred - normal_true)**2)
            print(f"    普通样本MSE: {normal_mse:.6f}")
            print(f"    普通预测范围: [{normal_pred.min():.4f}, {normal_pred.max():.4f}]")
            print(f"    普通真实范围: [{normal_true.min():.4f}, {normal_true.max():.4f}]")
            
            # 找出误差最大的样本,分析原因
            errors = np.abs(normal_pred - normal_true)
            worst_indices = np.argsort(errors)[-10:]  # 误差最大的10个
            
            print(f"    普通样本误差最大的10个:")
            for idx in worst_indices[:5]:
                global_idx = np.where(normal_mask)[0][idx]
                all_s = all_states[global_idx]  # all_states而不是states
                a1_joint = all_actions_team1[global_idx]
                a2_joint = all_actions_team2[global_idx]
                print(f"      all_s[...15]={all_s[:15]}, all_a1={a1_joint}, all_a2={a2_joint} → pred={normal_pred[idx]:.3f}, true={normal_true[idx]:.3f}, error={errors[idx]:.3f}")
            
            # 检查相同(all_states,all_actions)是否有不同reward
            print(f"    检查数据一致性(相同all_states,all_actions是否总有相同reward):")
            conflict_found = False
            for i in range(min(200, len(all_states))):  # 增加检查范围
                for j in range(i+1, min(len(all_states), i+200)):  # 检查更多样本对
                    if (np.allclose(all_states[i], all_states[j], atol=1e-5) and 
                        np.allclose(all_actions_team1[i], all_actions_team1[j], atol=1e-5) and 
                        np.allclose(all_actions_team2[i], all_actions_team2[j], atol=1e-5)):
                        if abs(rewards[i] - rewards[j]) > 1e-3:
                            print(f"      ✗ 发现冲突! all_states[...15]={all_states[i][:15]}")
                            print(f"        all_a1={all_actions_team1[i]}, all_a2={all_actions_team2[i]}")
                            print(f"        样本{i}: reward={rewards[i]:.3f}")
                            print(f"        样本{j}: reward={rewards[j]:.3f}")
                            print(f"        差值: {abs(rewards[i] - rewards[j]):.3f}")
                            conflict_found = True
                            break
                if conflict_found:
                    break
            
            if not conflict_found:
                print(f"      ✓ 没有发现数据冲突! 相同(all_states,all_actions)总有相同reward")
    
    def predict_reward(self, all_obs, all_team1_actions, all_team2_actions, agent_id):
        """
        获取奖励 R(all_states, all_current_actions)
        
        Args:
            all_obs: [n_agents, obs_dim] 所有智能体的观测 (obs = [state(5), teammate_flags(2)])
            all_team1_actions: [n_agents_team1] 当前阶段team1所有动作
            all_team2_actions: [n_agents_team2] 当前阶段team2所有动作
            agent_id: int
        Returns:
            reward: float
        """
        # 提取所有对的states (obs = [state(5), teammate_flags(2)], 只取前5维)
        all_states_list = []
        for i in range(len(all_obs)):
            state_i = all_obs[i, :self.state_shape]  # 取前5维state部分
            all_states_list.append(state_i)
        all_states = np.concatenate(all_states_list)
        
        self.lookup_total += 1
        
        # 构造联合动作one-hot
        all_actions_team1_onehot = []
        for a in all_team1_actions:
            onehot = np.zeros(self.n_actions_team1, dtype=np.float32)
            onehot[int(a)] = 1.0
            all_actions_team1_onehot.append(onehot)
        all_a1 = np.concatenate(all_actions_team1_onehot)
        
        all_actions_team2_onehot = []
        for a in all_team2_actions:
            onehot = np.zeros(self.n_actions_team2, dtype=np.float32)
            onehot[int(a)] = 1.0
            all_actions_team2_onehot.append(onehot)
        all_a2 = np.concatenate(all_actions_team2_onehot)
        
        # 使用网络预测
        network = self.reward_networks[agent_id]
        with paddle.no_grad():
            all_states_tensor = paddle.to_tensor(all_states.reshape(1, -1), dtype='float32')
            a1_tensor = paddle.to_tensor(all_a1.reshape(1, -1), dtype='float32')
            a2_tensor = paddle.to_tensor(all_a2.reshape(1, -1), dtype='float32')
            pred_reward = network(all_states_tensor, a1_tensor, a2_tensor)
        
        return float(pred_reward.squeeze())
    
    def get_lookup_stats(self):
        """获取查表统计信息"""
        if self.lookup_total == 0:
            return {"hit_rate": 0.0, "hits": 0, "total": 0}
        return {
            "hit_rate": self.lookup_hits / self.lookup_total,
            "hits": self.lookup_hits,
            "total": self.lookup_total
        }
    
    def predict_next_state(self, state, action_team1, action_team2):
        """
        预测下一状态
        
        对于Shapley两阶段博弈,状态转移规则:
        - 第一阶段(state[0]=1): s' = [0, a1_onehot, a2_onehot] (转移到对应子博弈)
        - 第二阶段(state[0]=0): s' = [1, 0, 0, 0, 0] (终止,重置到初始状态)
        
        Args:
            state: [state_dim] 当前状态
            action_team1: int Team1的代表动作
            action_team2: int Team2的代表动作
        Returns:
            next_state: [state_dim] 预测的下一状态
        """
        next_state = np.zeros(self.state_shape, dtype=np.float32)
        
        if state[0] > 0.5:  # 第一阶段 (stage=1)
            # 转移到第二阶段,编码第一阶段的动作(决定子博弈)
            next_state[0] = 0.0  # stage=0
            if self.state_shape > 1:
                next_state[1 + int(action_team1)] = 1.0
                next_state[1 + self.n_actions_team1 + int(action_team2)] = 1.0
        else:  # 第二阶段 (stage=0)
            # 终止,重置到初始状态
            next_state[0] = 1.0  # 回到stage=1
        
        return next_state
    
    def generate_trajectory(self, env, initial_obs, Tm, agent_policies):
        """
        从模型生成轨迹 (Planning阶段)
        
        Args:
            env: 真实环境(用于采样动作)
            initial_obs: [n_agents, obs_dim] 初始观测
            Tm: int 轨迹长度
            agent_policies: List[callable] 每个智能体的策略函数
        
        Returns:
            trajectory: dict with same structure as real trajectory
        """
        # 矩阵博弈是单步,Tm=1
        assert Tm == 1, "矩阵博弈只支持单步轨迹"
        
        n_agents = len(initial_obs)
        
        # 用策略采样动作
        actions_team1 = []
        actions_team2 = []
        
        for agent_id in range(n_agents):
            if agent_id < n_agents // 2:  # Team 1
                action = agent_policies[agent_id](initial_obs[agent_id])
                actions_team1.append(action)
            else:  # Team 2
                action = agent_policies[agent_id](initial_obs[agent_id])
                actions_team2.append(action)
        
        actions_team1 = np.array(actions_team1, dtype=np.int64)
        actions_team2 = np.array(actions_team2, dtype=np.int64)
        
        # 预测奖励 - 关键: 使用所有智能体的联合动作
        rewards = np.zeros(n_agents, dtype=np.float32)
        for agent_id in range(n_agents):
            # 所有智能体都使用相同的联合动作
            rewards[agent_id] = self.predict_reward(
                initial_obs,  # 传入所有智能体的obs
                actions_team1,  # 所有team1智能体的动作
                actions_team2,  # 所有team2智能体的动作
                agent_id
            )
        
        # 下一状态(终止)
        next_obs = initial_obs  # 矩阵博弈中next_obs无意义
        terminated = True
        
        # 构造轨迹
        trajectory = {
            'observations': initial_obs[np.newaxis, :],  # [1, n_agents, obs_dim]
            'team1_actions': np.concatenate([actions_team1, actions_team2])[np.newaxis, :],
            'team2_actions': np.concatenate([actions_team2, actions_team1])[np.newaxis, :],
            'rewards': rewards[np.newaxis, :],
            'next_observations': next_obs[np.newaxis, :],
            'terminated': np.array([terminated])
        }
        
        return trajectory
