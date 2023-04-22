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
from collections import deque 
from exp.exp045.src.observation import make_input, get_team_lichen_map
from exp.exp045.src.early_step_policy import _early_setup
from exp.exp045.src.unet import LuxUNetModel
from exp.exp045.src.validation import get_valid_robot_policy_map

MODEL_PATH = '/home/user/work/exp/exp045/models/best_all_model.pth'
TEACHER_MODEL_PATH = '/home/user/work/exp/exp045/models/best_all_model.pth'

seed = 2022

class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        """ 
        envをwrapして初期化
        """
        self.env = LuxAI_S2()
        self.reset()
        self.n_stack = 1


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
            self.obs_list = deque(maxlen=4)
            self.reward_list = deque(maxlen=4)
        self.obs_list.append(obs)
        self.last_actions = last_actions # ここはreturnした全ての行動をdictで持つ
        self.done = dones["player_0"] and dones["player_1"]
        self.reward_list.append(rewards)

        p = random.random()
        # 1/4 action queue cost
        if p < 0.25:
            self.init_env_cfg(zero_queue_cost=False)
        else:
            self.init_env_cfg(zero_queue_cost=True)
        # self.config_update()


    def step(self, actions):
        # state transition
        obs, rewards, dones, infos = self.env.step(actions)
        self.update((obs, rewards, dones, actions), False)


    def terminal(self):
        # if self.env.state.env_steps > 150: # debug
        #     return True 
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
        model = LuxUNetModel(n_channel=17, n_robot_class=10, n_factory_class=4, n_stack=self.n_stack)
        model.load_state_dict(torch.load(MODEL_PATH))
        model = self.fix_net_parameters(model)
        return model 


    def teacher_net(self):
        model = LuxUNetModel(n_channel=17, n_robot_class=10, n_factory_class=4, n_stack=self.n_stack)
        model.load_state_dict(torch.load(TEACHER_MODEL_PATH))
        for param in model.parameters():
            param.requires_grad = False
        self.set_bn_eval(model)
        return model
 

    def set_bn_eval(self, model):
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


    def fix_net_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False
        for param in model.value_net.parameters():
            param.requires_grad=True
        self.set_bn_eval(model)
        return model 

    def observation(self, player=None):
        own_team = self.player_idx(player)
        obses = [obs[player] for obs in list(self.obs_list)[-self.n_stack:]]
        assert len(obses)==self.n_stack
        real_env_steps = obses[-1]['real_env_steps']
        state = np.stack([make_input(obs, real_env_steps, own_team) for obs in obses])
        return state

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

            # heavyが食われたら-1/20=-0.05
            heavy_unit_count = len([unit_id for unit_id, unit in self.obs_list[-1]['player_0']['units'][player].items() if unit['unit_type']=='HEAVY'])
            prev_heavy_unit_count = len([unit_id for unit_id, unit in self.obs_list[-2]['player_0']['units'][player].items() if unit['unit_type']=='HEAVY'])
            if heavy_unit_count < prev_heavy_unit_count:
                rewards[player] = (heavy_unit_count - prev_heavy_unit_count) / 20
        return rewards


    # Should be defined in all games
    def legal_actions(self, player):
        """ 
        有効な行動を1として返す
        """
        game_state = self.get_game_state(player)
        robot_legal_actions = get_valid_robot_policy_map(game_state, player)
        return robot_legal_actions


    def get_info(self, player):
        n_factories0 = len(self.env.state.factories['player_0'].keys())
        n_units0 = len(self.env.state.units['player_0'].keys())
        n_lichen0 = get_team_lichen_map(
            self.env.state.board.lichen,
            self.env.state.board.lichen_strains,
            self.env.state.teams['player_0'].factory_strains
        ).sum()
        power = sum([f.power for f in self.env.state.factories[player].values()]) + sum([u.power for u in self.env.state.units[player].values()])
        ice = sum([f.cargo.ice for f in self.env.state.factories[player].values()]) + sum([u.cargo.ice for u in self.env.state.units[player].values()])
        water = sum([f.cargo.water for f in self.env.state.factories[player].values()]) + sum([u.cargo.water for u in self.env.state.units[player].values()])
        return f"[{player}] factories:{n_factories0} units:{n_units0} lichen:{n_lichen0} power:{power} ice:{ice}, water:{water}"


    def __str__(self):
        # 状況を可視化するもの
        player0_info = self.get_info('player_0')
        player1_info = self.get_info('player_1')
        return f'step:{self.env.state.real_env_steps} {player0_info} {player1_info}'