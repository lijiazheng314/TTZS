#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
DIGing分布式优化算法实现
基于Algorithm 2: DIGing - A Decentralized Optimization Algorithm
"""

import numpy as np
import paddle


class ConsensusMatrix:
    """共识矩阵 C_l = [c_l(i,j)]_{N×N}"""
    
    def __init__(self, n_agents, matrix_type='metropolis'):
        """
        Args:
            n_agents: 智能体数量
            matrix_type: 'metropolis', 'uniform', 'custom'
        """
        self.n_agents = n_agents
        self.matrix_type = matrix_type
        self.C = self._build_consensus_matrix()
    
    def _build_consensus_matrix(self):
        """构建共识矩阵"""
        if self.matrix_type == 'metropolis':
            # Metropolis权重: c(i,j) = 1/(1 + max(deg(i), deg(j)))
            # 假设全连接图
            degree = self.n_agents - 1
            weight = 1.0 / (1.0 + degree)
            
            C = np.ones((self.n_agents, self.n_agents)) * weight
            # 对角线元素
            np.fill_diagonal(C, 1.0 - (self.n_agents - 1) * weight)
            
            return C
        
        elif self.matrix_type == 'uniform':
            # 均匀权重
            weight = 1.0 / self.n_agents
            return np.ones((self.n_agents, self.n_agents)) * weight
        
        else:
            raise NotImplementedError(f"Unknown matrix type: {self.matrix_type}")
    
    def get_neighbors_weights(self, agent_id):
        """获取智能体i的邻居权重 c_l(i,j)"""
        return self.C[agent_id, :]
    
    def is_doubly_stochastic(self):
        """验证是否为双随机矩阵"""
        row_sum = np.sum(self.C, axis=1)
        col_sum = np.sum(self.C, axis=0)
        return np.allclose(row_sum, 1.0) and np.allclose(col_sum, 1.0)


class DIGingOptimizer:
    """
    DIGing分布式优化器
    
    算法伪代码:
        θ^i_{l+1} = Σ_{j∈N} c_l(i,j)·θ^j_l - α·γ^i_l
        γ^i_{l+1} = Σ_{j∈N} c_l(i,j)·γ^j_l + ∇g^i(θ^i_{l+1}) - ∇g^i(θ^i_l)
    """
    
    def __init__(self, n_agents, consensus_matrix, alpha=0.001):
        """
        Args:
            n_agents: 智能体数量
            consensus_matrix: ConsensusMatrix实例
            alpha: 学习率/步长
        """
        self.n_agents = n_agents
        self.C_matrix = consensus_matrix
        self.alpha = alpha
        
        # 存储每个智能体的参数和梯度追踪向量
        self.theta = [None] * n_agents  # θ^i_l
        self.gamma = [None] * n_agents  # γ^i_l (梯度追踪)
        
        # 迭代计数
        self.iteration = 0
    
    def initialize(self, initial_params, initial_gradients):
        """
        初始化参数和梯度追踪向量
        
        Args:
            initial_params: List[dict] - 每个智能体的初始参数 θ^i_0
            initial_gradients: List[dict] - 每个智能体的初始梯度 ∇g^i(θ^i_0)
        """
        for i in range(self.n_agents):
            self.theta[i] = {}
            self.gamma[i] = {}
            
            for name, param in initial_params[i].items():
                # 深拷贝参数
                self.theta[i][name] = param.clone().detach()
                
                # 初始化γ^i_0 = ∇g^i(θ^i_0)
                if name in initial_gradients[i]:
                    self.gamma[i][name] = initial_gradients[i][name].clone().detach()
                else:
                    self.gamma[i][name] = paddle.zeros_like(param)
    
    
    def collective_step(self, all_params, all_gradients, all_new_gradients):
        """
        所有智能体的集体更新步骤
        
        Args:
            all_params: List[dict] - 所有智能体的当前参数 [θ^1_l, ..., θ^N_l]
            all_gradients: List[dict] - 当前梯度 [∇g^1(θ^1_l), ...]
            all_new_gradients: List[dict] - 新参数的梯度 [∇g^1(θ^1_{l+1}), ...]
        
        Returns:
            updated_all_params: List[dict] - 更新后的所有参数
        """
        # 首先存储当前参数
        for i in range(self.n_agents):
            self.theta[i] = {k: v.clone().detach() for k, v in all_params[i].items()}
        
        updated_all_params = []
        
        for i in range(self.n_agents):
            weights = self.C_matrix.get_neighbors_weights(i)
            
            updated_params = {}
            new_gamma = {}
            
            for param_name in all_params[i].keys():
                # 步骤1: 参数共识
                consensus_term = paddle.zeros_like(all_params[i][param_name])
                
                for j in range(self.n_agents):
                    consensus_term += weights[j] * self.theta[j][param_name]
                
                theta_new = consensus_term - self.alpha * self.gamma[i][param_name]
                
                # 步骤2: 梯度追踪
                gamma_consensus = paddle.zeros_like(all_params[i][param_name])
                
                for j in range(self.n_agents):
                    gamma_consensus += weights[j] * self.gamma[j][param_name]
                
                grad_new = all_new_gradients[i][param_name]
                grad_old = all_gradients[i][param_name]
                
                gamma_new_val = gamma_consensus + grad_new - grad_old
                
                updated_params[param_name] = theta_new
                new_gamma[param_name] = gamma_new_val
            
            # 更新γ
            self.gamma[i] = {k: v.clone().detach() for k, v in new_gamma.items()}
            
            updated_all_params.append(updated_params)
        
        # 更新θ为新值
        for i in range(self.n_agents):
            self.theta[i] = {k: v.clone().detach() 
                           for k, v in updated_all_params[i].items()}
        
        self.iteration += 1
        
        return updated_all_params
    
    def get_consensus_error(self):
        """计算共识误差: max_ij ||θ^i - θ^j||"""
        if self.theta[0] is None:
            return float('inf')
        
        max_error = 0.0
        
        for param_name in self.theta[0].keys():
            for i in range(self.n_agents):
                for j in range(i+1, self.n_agents):
                    diff = self.theta[i][param_name] - self.theta[j][param_name]
                    error = float(paddle.norm(diff))
                    max_error = max(max_error, error)
        
        return max_error


class DIGingWrapper:
    """DIGing优化器的Paddle接口封装"""
    
    def __init__(self, parameters_list, consensus_matrix, learning_rate=0.001):
        """
        Args:
            parameters_list: List[paddle.nn.Layer.parameters()] 
                           每个智能体的参数列表
            consensus_matrix: ConsensusMatrix实例
            learning_rate: 学习率α
        """
        self.n_agents = len(parameters_list)
        self.parameters_list = parameters_list
        self.consensus_matrix = consensus_matrix
        self.learning_rate = learning_rate
        
        # 创建DIGing优化器
        self.diging = DIGingOptimizer(
            n_agents=self.n_agents,
            consensus_matrix=consensus_matrix,
            alpha=learning_rate
        )
        
        # 初始化标志
        self.initialized = False
        
        # 存储上一步的梯度(用于梯度追踪)
        self.prev_gradients = None
    
    def _extract_params_dict(self, param_list):
        """将参数列表转换为字典"""
        params_dict = {}
        for idx, param in enumerate(param_list):
            params_dict[f'param_{idx}'] = param
        return params_dict
    
    def _extract_grads_dict(self, param_list):
        """提取梯度字典"""
        grads_dict = {}
        for idx, param in enumerate(param_list):
            if param.grad is not None:
                grads_dict[f'param_{idx}'] = param.grad.clone()
            else:
                grads_dict[f'param_{idx}'] = paddle.zeros_like(param)
        return grads_dict
    
    def step(self):
        """执行一步DIGing优化"""
        # 收集所有智能体的参数和梯度
        all_params = []
        all_gradients = []
        
        for i in range(self.n_agents):
            params_dict = self._extract_params_dict(self.parameters_list[i])
            grads_dict = self._extract_grads_dict(self.parameters_list[i])
            
            all_params.append(params_dict)
            all_gradients.append(grads_dict)
        
        # 初始化
        if not self.initialized:
            self.diging.initialize(all_params, all_gradients)
            self.prev_gradients = all_gradients  # 保存初始梯度
            self.initialized = True
        
        # 执行集体更新 (使用上一步梯度进行梯度追踪)
        updated_params = self.diging.collective_step(
            all_params, self.prev_gradients, all_gradients
        )
        
        # 更新保存的梯度为当前梯度
        self.prev_gradients = all_gradients
        
        # 将更新后的参数设置回网络
        for i in range(self.n_agents):
            for idx, param in enumerate(self.parameters_list[i]):
                param_name = f'param_{idx}'
                param.set_value(updated_params[i][param_name])
    
    def clear_grad(self):
        """清除所有智能体的梯度"""
        for params in self.parameters_list:
            for param in params:
                if param.grad is not None:
                    param.clear_gradient()
    
    def get_consensus_error(self):
        """获取共识误差"""
        return self.diging.get_consensus_error()
