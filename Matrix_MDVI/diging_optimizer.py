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
    
    def __init__(self, n_agents, matrix_type='metropolis'):
        self.n_agents = n_agents
        self.matrix_type = matrix_type
        self.C = self._build_consensus_matrix()
    
    def _build_consensus_matrix(self):
        if self.matrix_type == 'metropolis':
            degree = self.n_agents - 1
            weight = 1.0 / (1.0 + degree)
            
            C = np.ones((self.n_agents, self.n_agents)) * weight
            np.fill_diagonal(C, 1.0 - (self.n_agents - 1) * weight)
            
            return C
        
        elif self.matrix_type == 'uniform':
            weight = 1.0 / self.n_agents
            return np.ones((self.n_agents, self.n_agents)) * weight
        
        else:
            raise NotImplementedError(f"Unknown matrix type: {self.matrix_type}")
    
    def get_neighbors_weights(self, agent_id):
        return self.C[agent_id, :]
    
    def is_doubly_stochastic(self):
        row_sum = np.sum(self.C, axis=1)
        col_sum = np.sum(self.C, axis=0)
        return np.allclose(row_sum, 1.0) and np.allclose(col_sum, 1.0)


class DIGingOptimizer:
    
    def __init__(self, n_agents, consensus_matrix, alpha=0.001):
        self.n_agents = n_agents
        self.C_matrix = consensus_matrix
        self.alpha = alpha
        
        self.theta = [None] * n_agents
        self.gamma = [None] * n_agents
        
        self.iteration = 0
    
    def initialize(self, initial_params, initial_gradients):
        for i in range(self.n_agents):
            self.theta[i] = {}
            self.gamma[i] = {}
            
            for name, param in initial_params[i].items():
                self.theta[i][name] = param.clone().detach()
                
                if name in initial_gradients[i]:
                    self.gamma[i][name] = initial_gradients[i][name].clone().detach()
                else:
                    self.gamma[i][name] = paddle.zeros_like(param)
    
    
    def collective_step(self, all_params, all_gradients, all_new_gradients):
        for i in range(self.n_agents):
            self.theta[i] = {k: v.clone().detach() for k, v in all_params[i].items()}
        
        updated_all_params = []
        
        for i in range(self.n_agents):
            weights = self.C_matrix.get_neighbors_weights(i)
            
            updated_params = {}
            new_gamma = {}
            
            for param_name in all_params[i].keys():
                consensus_term = paddle.zeros_like(all_params[i][param_name])
                
                for j in range(self.n_agents):
                    consensus_term += weights[j] * self.theta[j][param_name]
                
                theta_new = consensus_term - self.alpha * self.gamma[i][param_name]
                
                gamma_consensus = paddle.zeros_like(all_params[i][param_name])
                
                for j in range(self.n_agents):
                    gamma_consensus += weights[j] * self.gamma[j][param_name]
                
                grad_new = all_new_gradients[i][param_name]
                grad_old = all_gradients[i][param_name]
                
                gamma_new_val = gamma_consensus + grad_new - grad_old
                
                updated_params[param_name] = theta_new
                new_gamma[param_name] = gamma_new_val
            
            self.gamma[i] = {k: v.clone().detach() for k, v in new_gamma.items()}
            
            updated_all_params.append(updated_params)
        
        for i in range(self.n_agents):
            self.theta[i] = {k: v.clone().detach() 
                           for k, v in updated_all_params[i].items()}
        
        self.iteration += 1
        
        return updated_all_params
    
    def get_consensus_error(self):
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
    
    def __init__(self, parameters_list, consensus_matrix, learning_rate=0.001):
        self.n_agents = len(parameters_list)
        self.parameters_list = parameters_list
        self.consensus_matrix = consensus_matrix
        self.learning_rate = learning_rate
        
        self.diging = DIGingOptimizer(
            n_agents=self.n_agents,
            consensus_matrix=consensus_matrix,
            alpha=learning_rate
        )
        
        self.initialized = False
        
        self.prev_gradients = None
    
    def _extract_params_dict(self, param_list):
        params_dict = {}
        for idx, param in enumerate(param_list):
            params_dict[f'param_{idx}'] = param
        return params_dict
    
    def _extract_grads_dict(self, param_list):
        grads_dict = {}
        for idx, param in enumerate(param_list):
            if param.grad is not None:
                grads_dict[f'param_{idx}'] = param.grad.clone()
            else:
                grads_dict[f'param_{idx}'] = paddle.zeros_like(param)
        return grads_dict
    
    def step(self):
        all_params = []
        all_gradients = []
        
        for i in range(self.n_agents):
            params_dict = self._extract_params_dict(self.parameters_list[i])
            grads_dict = self._extract_grads_dict(self.parameters_list[i])
            
            all_params.append(params_dict)
            all_gradients.append(grads_dict)
        
        if not self.initialized:
            self.diging.initialize(all_params, all_gradients)
            self.prev_gradients = all_gradients
            self.initialized = True
        
        updated_params = self.diging.collective_step(
            all_params, self.prev_gradients, all_gradients
        )
        
        self.prev_gradients = all_gradients
        
        for i in range(self.n_agents):
            for idx, param in enumerate(self.parameters_list[i]):
                param_name = f'param_{idx}'
                param.set_value(updated_params[i][param_name])
    
    def clear_grad(self):
        for params in self.parameters_list:
            for param in params:
                if param.grad is not None:
                    param.clear_gradient()
    
    def get_consensus_error(self):
        return self.diging.get_consensus_error()
