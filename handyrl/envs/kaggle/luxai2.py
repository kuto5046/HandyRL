# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# kaggle_environments licensed under Copyright 2020 Kaggle Inc. and the Apache License, Version 2.0
# (see https://github.com/Kaggle/kaggle-environments/blob/master/LICENSE for details)

# wrapper of Hungry Geese environment from kaggle

import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# You need to install kaggle_environments, requests
from kaggle_environments import make

from ...environment import BaseEnvironment
# import sys 
# sys.append('/home/user/work/lux')
from lux.kit import obs_to_game_state, GameState, EnvConfig
from luxai_s2 import LuxAI_S2
from exp.exp031.src.observation import make_input
from exp.exp031.src.early_step_policy import _early_setup
from exp.exp031.src.unet import LuxUNetModel
from exp.exp031.src.validation import get_valid_robot_policy_map

MODEL_PATH = '/home/user/work/exp/exp031/models/best_robot_model.pth'
TEACHER_MODEL_PATH = '/home/user/work/exp/exp031/models/best_robot_model.pth'

seed = 2022

class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        """ 
        envをwrapして初期化
        """
        self.env = LuxAI_S2()
        self.reset()


    def init_env_cfg(self, zero_queue_cost):
        """Falseにしたい場合resetの後に手動実行する"""
        env_cfg = EnvConfig()
        if zero_queue_cost:
            env_cfg.ROBOTS['LIGHT'].ACTION_QUEUE_POWER_COST = 0
            env_cfg.ROBOTS['HEAVY'].ACTION_QUEUE_POWER_COST = 0
        self.env.env_cfg = env_cfg
        self.env.state.env_cfg = env_cfg 

    def reset(self, args={}, zero_queue_cost=True):
        """ 
        envをresetしてupdate
        """
        self.init_env_cfg(zero_queue_cost)
        obs = self.env.reset()
        self.update((obs, {}, {'player_0':False, 'player_1': False}, {}), True)        

        while self.env.state.real_env_steps < 0:
            action = dict()
            for player in self.players():
                _obs = self.obs_list[-1][player]
                action[player] = _early_setup(self.env.state.env_steps, _obs, player, self.env.env_cfg)
            obs, _, _, _ = self.env.step(action)
            self.update((obs, {}, {'player_0':False, 'player_1': False}, action), False)    

    # def config_update(self):
    #     self.env_cfg = EnvConfig()
    #     # if random.random() > 0.0:
    #     self.env_cfg.ROBOTS['LIGHT'].ACTION_QUEUE_POWER_COST = 0
    #     self.env_cfg.ROBOTS['HEAVY'].ACTION_QUEUE_POWER_COST = 0
        

    def update(self, info, reset):
        obs, rewards, dones, last_actions = info
        
        if reset:
            self.obs_list = []
            self.reward_list = []
        self.obs_list.append(obs)
        self.last_actions = last_actions # ここはreturnした全ての行動をdictで持つ
        self.done = dones["player_0"] and dones["player_1"]
        self.reward_list.append(rewards)
        # self.config_update()


    def step(self, actions):
        # state transition
        obs, rewards, dones, infos = self.env.step(actions)
        self.update((obs, rewards, dones, actions), False)


    def terminal(self):
        if self.env.state.env_steps > 150: # debug
            return True 
        return self.done


    def outcome(self):
        # assert self.done == True
        reward = self.reward_list[-1]
        if reward['player_0'] < reward['player_1']:
            return {'player_0': -1, 'player_1': 1}
        elif reward['player_0'] > reward['player_1']:
            return {'player_0': 1, 'player_1': -1}
        else:
            return {'player_0': 0, 'player_1': 0}   

    def get_game_state(self, player):
        obs = self.obs_list[-1][player]
        step = self.env.state.env_steps
        game_state = obs_to_game_state(step, self.env.env_cfg, obs)
        return game_state 


    def players(self):
        return ['player_0', 'player_1']

    def player_idx(self, player):
        return int(player[-1])
    
    def net(self):
        model = LuxUNetModel(n_channel=19, n_robot_class=9)
        model.load_state_dict(torch.load(MODEL_PATH))
        model = self.fix_net_parameters(model)
        return model 


    def teacher_net(self):
        model = LuxUNetModel(n_channel=19, n_robot_class=9)
        model.load_state_dict(torch.load(MODEL_PATH))
        for param in model.parameters():
            param.requires_grad = False
        return model 
 

    def fix_net_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False
        layers = list(model.children())
        value_layer = layers[-1]
        policy_layer = layers[-3]
        print(f'train only following layer: \n{policy_layer} \n{value_layer}')
        for param in value_layer.parameters():
            param.requires_grad = True
        for param in policy_layer.parameters():
            param.requires_grad = True
        return model 

    def observation(self, player=None):
        own_team = self.player_idx(player)
        obs = self.obs_list[-1][player]
        state = make_input(obs, obs['real_env_steps'], own_team)
        if len(self.obs_list) >= 2:
            prev_obs = self.obs_list[-2][player]
            prev_state = make_input(prev_obs, prev_obs['real_env_steps'], own_team)
        else:
            prev_state = state.copy()
        return (state, prev_state)

    # Should be defined if you use multiplayer simultaneous action game
    def turns(self):
        # players to move
        # return [p for p in self.players() if self.obs_list[-1][p]['status'] == 'ACTIVE']
        return self.players() 

    # Should be defined if you use immediate reward
    def reward(self):
        rewards = {}
        for player in self.players():
            # factoryが消滅したら-1/10
            factories_count = len(self.obs_list[-1]['player_0']['factories'][player])
            prev_factories_count = len(self.obs_list[-2]['player_0']['factories'][player])
            assert factories_count <= prev_factories_count
            rewards[player] = (factories_count - prev_factories_count) / 10  # 負の報酬 win rewardsを超えないように10でわる
        return rewards


    # Should be defined in all games
    def legal_actions(self, player):
        """ 
        有効な行動を1として返す
        """
        game_state = self.get_game_state(player)
        robot_legal_actions = get_valid_robot_policy_map(game_state, player)
        return robot_legal_actions

    def __str__(self):
        # 状況を可視化するもの
        target_player = 'player_1'
        print(f'step:{self.env.state.real_env_steps}', file=sys.stderr)
        print(f'factories: {self.env.state.factories[target_player].keys()}', file=sys.stderr)
        print(f'units: {self.env.state.units[target_player].keys()}', file=sys.stderr)