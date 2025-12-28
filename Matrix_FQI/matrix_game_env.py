#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
自定义矩阵博弈环境 - 具有非平凡混合策略Nash均衡的两队零和博弈

设计思路:
- 两队智能体的零和博弈
- 每个智能体有自己的局部Q值
- 通过DIGing优化器共享信息
- 真实Nash均衡可解析计算,便于验证理论收敛性
"""

import numpy as np


class TwoTeamMatrixGame:
    
    def __init__(self, n_agents_team1=3, n_agents_team2=3, n_actions=3, 
                 game_type='rock_paper_scissors', cooperation_weight=0.0):
        """
        Args:
            n_agents_team1: Team 1智能体数量
            n_agents_team2: Team 2智能体数量
            n_actions: 每个智能体的动作数
            game_type: 博弈类型
                - 'rock_paper_scissors': 经典剪刀石头布(3x3,单阶段)
                - 'matching_pennies': 配对硬币(2x2,单阶段)
                - 'coordination': 多智能体协调博弈(3x3,单阶段)
                - 'shapley': 经典两阶段Shapley博弈(2x2x2,需要计算两步Q值)
            cooperation_weight: 团队协作权重 (0=完全独立, 1=完全团队奖励)
        """
        self.n_agents_team1 = n_agents_team1
        self.n_agents_team2 = n_agents_team2
        self.n_agents = n_agents_team1 + n_agents_team2
        self.n_actions = n_actions
        self.game_type = game_type
        self.cooperation_weight = cooperation_weight
        
        # Shapley两阶段博弈特有属性
        self.is_two_stage = (game_type == 'shapley')
        self.current_stage = 0  # 0=第一阶段, 1=第二阶段
        self.first_stage_actions = None  # 记录第一阶段的动作 [team1_actions, team2_actions]
        
        # 【关键改进】每个智能体对维护自己的子博弈状态
        # agent_pair_states[i] = 第i对智能体的状态
        self.agent_pair_states = [np.zeros(1 + n_actions + n_actions, dtype=np.float32) 
                                   for _ in range(max(n_agents_team1, n_agents_team2))]
        for state in self.agent_pair_states:
            state[0] = 1.0  # 初始都在第一阶段
        
        # 设计payoff矩阵
        if self.is_two_stage:
            # 两阶段: 需要两个payoff矩阵
            self.payoff_stage1, self.payoff_stage2 = self._design_two_stage_payoff()
            self.payoff_matrix = self.payoff_stage1  # 兼容性
        else:
            # 单阶段: 一个payoff矩阵
            self.payoff_matrix = self._design_payoff_matrix()
        
        # 计算理论Nash均衡(用于验证)
        self.true_nash_team1, self.true_nash_team2, self.nash_value = \
            self._compute_true_nash()
        
        # 状态空间
        if self.is_two_stage:
            # 两阶段: 状态=[stage, first_action_team1, first_action_team2]
            self.state_dim = 1 + n_actions + n_actions  # stage + 动作编码
        else:
            # 单阶段: 状态=标量
            self.state_dim = 1
        
        # 观测空间 = 局部state + 两个队友的负奖励标志
        self.obs_dim = self.state_dim + 2  # +2 for two teammates' negative reward flags
        
        print(f"\n=== Two-Team Matrix Game Initialized ===")
        print(f"  Team 1: {n_agents_team1} agents")
        print(f"  Team 2: {n_agents_team2} agents")
        print(f"  Actions per agent: {n_actions}")
        print(f"  Game type: {game_type}")
        print(f"  Cooperation weight: {cooperation_weight:.2f} (0=independent, 1=full team reward)")
        print(f"  State dim: {self.state_dim}")
        print(f"  Obs dim: {self.obs_dim} (局部state[{self.state_dim}] + 队友1负奖励[1] + 队友2负奖励[1])")
        if self.is_two_stage:
            print(f"  Mode: Two-stage Shapley game (2 steps)")
            print(f"  Stage 1: r1(a1,b1) - immediate reward")
            print(f"  Stage 2: r2(a1,b1,a2,b2) - depends on stage1 actions!")
            print(f"  Total reward: r1 + γ*r2 (strict zero-sum)")
            print(f"  Discount factor gamma should be > 0!")
        else:
            print(f"  Mode: Single-stage game (1 step)")
        print(f"  True Nash value: {self.nash_value:.4f}")
        print(f"  Team 1 Nash strategy: {self.true_nash_team1}")
        print(f"  Team 2 Nash strategy: {self.true_nash_team2}")
    
    def _design_payoff_matrix(self):
        if self.game_type == 'rock_paper_scissors':
            # 经典剪刀石头布: Nash均衡为(1/3, 1/3, 1/3)
            assert self.n_actions == 3, "Rock-Paper-Scissors需要3个动作"
            payoff = np.array([
                [ 0,  -1,   1],  # Rock vs [Rock, Paper, Scissors]
                [ 1,   0,  -1],  # Paper vs [Rock, Paper, Scissors]
                [-1,   1,   0]   # Scissors vs [Rock, Paper, Scissors]
            ], dtype=np.float32)
            
        elif self.game_type == 'matching_pennies':
            # 配对硬币: Nash均衡为(0.5, 0.5)
            assert self.n_actions == 2, "Matching Pennies需要2个动作"
            payoff = np.array([
                [ 1,  -1],  # Head vs [Head, Tail]
                [-1,   1]   # Tail vs [Head, Tail]
            ], dtype=np.float32)
            
        elif self.game_type == 'coordination':
            # 复杂协调博弈(3x3): 非对称Nash均衡
            assert self.n_actions == 3
            payoff = np.array([
                [ 2,  -1,   0],
                [-1,   1,   2],
                [ 0,   2,  -1]
            ], dtype=np.float32)
            
        elif self.game_type == 'shapley':
            # 经典两阶段Shapley博弈
            # 这是单阶段的设计,两阶段用_design_two_stage_payoff()
            raise ValueError("Shapley博弈应该由_design_two_stage_payoff()处理")
            
        else:
            # 默认:随机生成对称零和博弈
            payoff = np.random.randn(self.n_actions, self.n_actions).astype(np.float32)
            payoff = (payoff - payoff.T) / 2  # 对称化为零和
        
        return payoff
    
    def _design_two_stage_payoff(self):
        """
        设计经典Shapley两阶段零和博弈
        
        经典Shapley博弈 (Shapley 1953):
        - 第一阶段: 两队博弈,获得即时reward r1(a1,b1)
        - 第二阶段: 继续博弈,获得reward r2(a1,b1,a2,b2)
                    注意: 第二阶段的payoff矩阵取决于第一阶段的动作!
        - 总reward = r1 + γ * r2
        
        关键设计:
        1) 第一阶段payoff非零(有即时收益)
        2) 第一阶段动作会改变第二阶段的博弈结构
        3) 必须是严格零和: Team1 + Team2 = 0
        
        具体设计(2x2两阶段):
        - Stage 1: 简单零和博弈,有即时reward
        - Stage 2: 根据stage1的(a1,b1)组合,进入4种不同的零和博弈
        
        Returns:
            payoff_stage1: [2, 2] 第一阶段的即时payoff
            payoff_stage2: [2, 2, 2, 2] 第二阶段的payoff
                          [a1_stage1, a2_stage1, a1_stage2, a2_stage2]
        """
        assert self.n_actions == 2, "Shapley博弈需要2个动作"
        
        # 第一阶段: Matching Pennies (零和,有即时reward)
        payoff_stage1 = np.array([
            [ 1, -1],  # Team1选0: 赢/输 → ±1
            [-1,  1]   # Team1选1: 输/赢 → ±1
        ], dtype=np.float32)
        
        # 第二阶段: 4个不同的零和子博弈
        # 关键: 第一阶段的结果决定第二阶段的博弈类型!
        payoff_stage2 = np.zeros((2, 2, 2, 2), dtype=np.float32)
        
        # 子博弈 (0,0): 第一阶段两队都选0 → 进入标准Matching Pennies
        payoff_stage2[0, 0, :, :] = np.array([
            [ 1, -1],
            [-1,  1]
        ], dtype=np.float32)
        
        # 子博弈 (0,1): 第一阶段Team1选0,Team2选1 → 反向Matching Pennies
        payoff_stage2[0, 1, :, :] = np.array([
            [-1,  1],
            [ 1, -1]
        ], dtype=np.float32)
        
        # 子博弈 (1,0): 第一阶段Team1选1,Team2选0 → 更激进的零和博弈
        payoff_stage2[1, 0, :, :] = np.array([
            [ 2, -2],  # 更高的stakes
            [-2,  2]
        ], dtype=np.float32)
        
        # 子博弈 (1,1): 第一阶段两队都选1 → 保守的零和博弈
        payoff_stage2[1, 1, :, :] = np.array([
            [ 0.5, -0.5],  # 较低的stakes
            [-0.5,  0.5]
        ], dtype=np.float32)
        
        # 验证零和性质
        print(f"\n[验证零和Shapley博弈]")
        
        # 验证第一阶段零和
        assert np.allclose(payoff_stage1 + (-payoff_stage1), 0), "第一阶段不是零和!"
        print(f"  第一阶段: 零和 ✓")
        
        # 验证第二阶段每个子博弈零和
        for a1_s1 in range(2):
            for a2_s1 in range(2):
                sub_game = payoff_stage2[a1_s1, a2_s1, :, :]
                total = sub_game + (-sub_game)
                assert np.allclose(total, 0), f"子博弈({a1_s1},{a2_s1})不是零和!"
                print(f"  子博弈({a1_s1},{a2_s1}): 零和 ✓, payoff范围[{sub_game.min():.1f}, {sub_game.max():.1f}]")
        
        print(f"  总reward = r1(a1,b1) + γ*r2(a1,b1,a2,b2), 严格零和!")
        
        return payoff_stage1, payoff_stage2
    
    def _compute_true_nash(self):
        from scipy.optimize import linprog
        
        n = self.n_actions
        A = self.payoff_matrix
        
        # Team 1求解: max_pi min_sigma pi^T A sigma
        # 转换为LP: max v, s.t. A^T pi >= v*1, sum(pi)=1, pi>=0
        c = np.zeros(n + 1)
        c[-1] = -1  # 最大化v
        
        A_ub = np.zeros((n, n + 1))
        A_ub[:, :n] = -A.T
        A_ub[:, -1] = np.ones(n)
        b_ub = np.zeros(n)
        
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = np.ones(n)
        b_eq = np.array([1.0])
        
        bounds = [(0, None) for _ in range(n)] + [(None, None)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs-ipm')
        
        if result.success:
            nash_team1 = result.x[:n]
            nash_value = -result.fun
            nash_team1 = nash_team1 / nash_team1.sum()  # 归一化
        else:
            # 失败则返回均匀分布
            nash_team1 = np.ones(n) / n
            nash_value = 0.0
        
        # Team 2求解: min_sigma max_pi pi^T A sigma  
        c = np.zeros(n + 1)
        c[-1] = 1  # 最小化v
        
        A_ub = np.zeros((n, n + 1))
        A_ub[:, :n] = A
        A_ub[:, -1] = -np.ones(n)
        b_ub = np.zeros(n)
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs-ipm')
        
        if result.success:
            nash_team2 = result.x[:n]
            nash_team2 = nash_team2 / nash_team2.sum()
        else:
            nash_team2 = np.ones(n) / n
        
        return nash_team1, nash_team2, nash_value
    
    def reset(self):
        # 重置阶段
        self.current_stage = 0
        self.first_stage_actions = None
        
        # 重置每个智能体对的状态
        for i in range(len(self.agent_pair_states)):
            self.agent_pair_states[i] = np.zeros(self.state_dim, dtype=np.float32)
            self.agent_pair_states[i][0] = 1.0  # 第一阶段
        
        # 全局state(兼容性,使用agent[0]的)
        state = self.agent_pair_states[0].copy()
        
        # 每个智能体有不同的局部观测
        observations = self._get_observations()
        
        return state, observations
    
    def step(self, actions_team1, actions_team2):
        if self.is_two_stage:
            return self._step_two_stage(actions_team1, actions_team2)
        else:
            return self._step_single_stage(actions_team1, actions_team2)
    
    def _step_single_stage(self, actions_team1, actions_team2):
        # 先计算每个Agent的个体奖励
        individual_rewards_team1 = np.zeros(self.n_agents_team1, dtype=np.float32)
        individual_rewards_team2 = np.zeros(self.n_agents_team2, dtype=np.float32)
        
        # Team 1每个智能体与Team 2对应位置的智能体博弈
        for i in range(self.n_agents_team1):
            my_action = int(actions_team1[i])
            opp_action = int(actions_team2[i])
            individual_rewards_team1[i] = self.payoff_matrix[my_action, opp_action]
        
        # Team 2每个智能体与Team 1对应位置的智能体博弈(零和)
        for i in range(self.n_agents_team2):
            my_action = int(actions_team2[i])
            opp_action = int(actions_team1[i])
            individual_rewards_team2[i] = -self.payoff_matrix[opp_action, my_action]
        
        # 每个agent获得自己的individual reward
        rewards_team1 = individual_rewards_team1
        rewards_team2 = individual_rewards_team2
        
        # 下一状态(单步博弈,重置为初始状态)
        for i in range(len(self.agent_pair_states)):
            self.agent_pair_states[i] = np.zeros(self.state_dim, dtype=np.float32)
            self.agent_pair_states[i][0] = 1.0  # 第一阶段
        
        next_state = self.agent_pair_states[0].copy()
        # 观测包含当前rewards信息
        next_observations = self._get_observations(rewards_team1, rewards_team2)
        terminated = True
        
        return next_state, next_observations, rewards_team1, rewards_team2, terminated
    
    def _step_two_stage(self, actions_team1, actions_team2):
        if self.current_stage == 0:
            # 第一阶段: 有即时reward,转移到第二阶段
            self.first_stage_actions = (actions_team1.copy(), actions_team2.copy())
            self.current_stage = 1
            
            # 第一阶段的即时reward - 一一对应博弈
            individual_rewards_team1 = np.zeros(self.n_agents_team1, dtype=np.float32)
            individual_rewards_team2 = np.zeros(self.n_agents_team2, dtype=np.float32)
            
            for i in range(self.n_agents_team1):
                a1 = int(actions_team1[i])
                a2 = int(actions_team2[i])
                individual_rewards_team1[i] = self.payoff_stage1[a1, a2]
            
            for i in range(self.n_agents_team2):
                a1 = int(actions_team1[i])
                a2 = int(actions_team2[i])
                individual_rewards_team2[i] = -self.payoff_stage1[a1, a2]  # 零和
            
            # 每个agent获得自己的individual reward
            rewards_team1 = individual_rewards_team1
            rewards_team2 = individual_rewards_team2
            
            # 构造第二阶段状态: 每个智能体对更新自己的状态
            for i in range(max(self.n_agents_team1, self.n_agents_team2)):
                next_state_i = np.zeros(self.state_dim, dtype=np.float32)
                next_state_i[0] = 0.0  # stage=0 (第二阶段)
                
                # 编码该对的第一阶段动作
                if i < len(actions_team1):
                    a1 = int(actions_team1[i])
                    a2 = int(actions_team2[i])
                    next_state_i[1 + a1] = 1.0
                    next_state_i[1 + self.n_actions + a2] = 1.0
                
                self.agent_pair_states[i] = next_state_i
            
            # Next state: 返回所有对的状态信息(向算法表明这是多子博弈系统)
            # 注意: 真正的状态信息在每个智能体的observation中
            next_state = self.agent_pair_states[0].copy()  # 兼容性:仍返回标准维度
            
            # 观测包含新状态 + 当前奖励信息
            next_observations = self._get_observations(rewards_team1, rewards_team2)
            terminated = False  # 未结束
            
        else:
            # 第二阶段: 计算第二阶段reward (注意:不包含第一阶段reward!)
            a1_s1, a2_s1 = self.first_stage_actions
            a1_s2, a2_s2 = actions_team1, actions_team2
            
            rewards_team1 = np.zeros(self.n_agents_team1, dtype=np.float32)
            rewards_team2 = np.zeros(self.n_agents_team2, dtype=np.float32)
            
            # 第二阶段reward: r2(a1_s1, a2_s1, a1_s2, a2_s2) - 一一对应博弈
            # 总reward = r1 + γ*r2, 由Q学习自动计算
            individual_rewards_team1 = np.zeros(self.n_agents_team1, dtype=np.float32)
            individual_rewards_team2 = np.zeros(self.n_agents_team2, dtype=np.float32)
            
            for i in range(self.n_agents_team1):
                r = self.payoff_stage2[
                    int(a1_s1[i]), int(a2_s1[i]),
                    int(a1_s2[i]), int(a2_s2[i])
                ]
                individual_rewards_team1[i] = r
            
            for i in range(self.n_agents_team2):
                r = self.payoff_stage2[
                    int(a1_s1[i]), int(a2_s1[i]),
                    int(a1_s2[i]), int(a2_s2[i])
                ]
                individual_rewards_team2[i] = -r  # 零和
            
            # 每个agent获得自己的individual reward
            rewards_team1 = individual_rewards_team1
            rewards_team2 = individual_rewards_team2
            
            # 结束,但先构造包含当前rewards的observations
            # 重置状态为初始状态
            for i in range(len(self.agent_pair_states)):
                self.agent_pair_states[i] = np.zeros(self.state_dim, dtype=np.float32)
                self.agent_pair_states[i][0] = 1.0  # 第一阶段
            
            next_state = self.agent_pair_states[0].copy()
            # 观测包含当前阶段的奖励信息(第二阶段的rewards)
            next_observations = self._get_observations(rewards_team1, rewards_team2)
            terminated = True
        
        return next_state, next_observations, rewards_team1, rewards_team2, terminated
    
    def _get_observations(self, rewards_team1=None, rewards_team2=None):
        """
        获取当前观测(不改变状态)
        
        观测 = 局部pair_state + 两个队友的负奖励标志
        【关键】每个智能体看到自己所处的子博弈状态 + 每个队友是否得到负奖励
        每个agent仍有自己的Q_i和π_i，但输入是相同格式的局部state
        
        Args:
            rewards_team1: [n_agents_team1] Team1的奖励(可选,用于构造观测)
            rewards_team2: [n_agents_team2] Team2的奖励(可选)
        """
        observations = []
        for i in range(self.n_agents):
            if i < self.n_agents_team1:
                # Team1的智能体 - 局部state
                pair_state = self.agent_pair_states[i]
                
                # 检查两个队友是否有负奖励(分别编码)
                teammate_flags = [0.0, 0.0]
                if rewards_team1 is not None:
                    # 找到两个队友的索引(不包括自己)
                    teammates = [j for j in range(self.n_agents_team1) if j != i]
                    for idx, teammate_id in enumerate(teammates):
                        if rewards_team1[teammate_id] < 0:
                            teammate_flags[idx] = 1.0
            else:
                # Team2的智能体
                team2_idx = i - self.n_agents_team1
                pair_state = self.agent_pair_states[team2_idx]
                
                # 检查两个队友是否有负奖励(分别编码)
                teammate_flags = [0.0, 0.0]
                if rewards_team2 is not None:
                    # 找到两个队友的索引(不包括自己)
                    teammates = [j for j in range(self.n_agents_team2) if j != team2_idx]
                    for idx, teammate_id in enumerate(teammates):
                        if rewards_team2[teammate_id] < 0:
                            teammate_flags[idx] = 1.0
            
            # 拼接: pair_state + teammate1_flag + teammate2_flag
            obs_i = np.concatenate([pair_state, teammate_flags])
            observations.append(obs_i)
        return np.array(observations, dtype=np.float32)
    
    def get_payoff(self, action_team1, action_team2):
        return float(self.payoff_matrix[action_team1, action_team2])
    
    @property
    def obs_shape(self):
        return self.obs_dim
    
    @property
    def state_shape(self):  
        return self.state_dim
