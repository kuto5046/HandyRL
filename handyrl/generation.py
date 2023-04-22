# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation

import random
import bz2
import pickle

import numpy as np
import time 
from .util import softmax
from exp.exp045.agent import get_factory_actions, get_robot_actions
from exp.exp045.src.observation import robot_action_to_label


class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def generate(self, models, args):
        # episode generation
        moments = []
        hidden = {}
        for player in self.env.players():
            hidden[player] = models[player].init_hidden()

        err = self.env.reset()
        if err:
            return None
        start_time = time.time()
        while not self.env.terminal():
            moment_keys = ['observation', 'selected_prob', 'action_mask', 'action', 'value', 'reward', 'return']
            moment = {key: {p: None for p in self.env.players()} for key in moment_keys}

            turn_players = self.env.turns()
            observers = self.env.observers()
            output_actions = {}
            for player in self.env.players():
                if player not in turn_players + observers:
                    continue
                if player not in turn_players and player in args['player'] and not self.args['observation']:
                    continue

                obs = self.env.observation(player)
                model = models[player]
                # start_time = time.time()
                outputs = model.inference(obs, hidden[player])
                # print(f'{time.time()-start_time:.4f}s')
                hidden[player] = outputs.get('hidden', None)
                v = outputs.get('value', None)

                moment['observation'][player] = obs
                moment['value'][player] = v

                # step = self.env.env.state.env_steps
                if player in turn_players:
                    # _obs = self.env.obs_list[-1][player]
                    assert self.env.env.state.real_env_steps >= 0
                    actions = dict()
                    game_state = self.env.get_game_state(player)

                    # greedyでなくサンプリングにしたほうがいいかも
                    actions = get_factory_actions(game_state, outputs['factory_policy'], player, actions)
                    actions = get_robot_actions(game_state, outputs['robot_policy'], player, actions)
                    output_actions[player] = actions

                    units = game_state.units[player]
                    map_size = self.env.env.env_cfg.map_size
                    selected_prob_map = np.zeros((map_size, map_size), dtype=np.float32)
                    # 有効な行動の場合1となっている
                    action_mask_map = self.env.legal_actions(player)
                    action_map = np.zeros((map_size, map_size), dtype=np.float32)
                    for unit_id, action in sorted(actions.items()):
                        if 'unit' not in unit_id:
                            continue 

                        label = robot_action_to_label(action[0])
                        x, y = units[unit_id].pos
                        p_ = outputs['robot_policy'][:, x,y]
                        action_mask = np.ones_like(p_) * 1e32
                        legal_actions = np.where(action_mask_map[:, x, y] == 1)[0]
                        action_mask[legal_actions] = 0
                        p = softmax(p_ - action_mask)
                        selected_prob_map[x, y] = p[label]
                        action_map[x,y] = label

                    # 有効な行動が1になっている。それ以外を1e32にして1を0にする
                    moment['selected_prob'][player] = selected_prob_map
                    moment['action_mask'][player] = action_mask_map
                    moment['action'][player] = action_map

            err = self.env.step(output_actions)
            # print(f'[Gen] step: {self.env.env.state.real_env_steps} {self.env}')
            if err:
                return None

            reward = self.env.reward()
            for player in self.env.players():
                moment['reward'][player] = reward.get(player, None)

            moment['turn'] = turn_players
            moments.append(moment)

        # print(f"[Gen] {time.time() - start_time}s")
        if len(moments) < 1:
            return None

        for player in self.env.players():
            ret = 0
            for i, m in reversed(list(enumerate(moments))):
                ret = (m['reward'][player] or 0) + self.args['gamma'] * ret
                moments[i]['return'][player] = ret

        episode = {
            'args': args, 'steps': len(moments),
            'outcome': self.env.outcome(),
            'moment': [
                bz2.compress(pickle.dumps(moments[i:i+self.args['compress_steps']]))
                for i in range(0, len(moments), self.args['compress_steps'])
            ]
        }

        return episode

    def execute(self, models, args):
        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')
        return episode
