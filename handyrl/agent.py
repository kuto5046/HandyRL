# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# agent classes

import random

import numpy as np

from .util import softmax
from exp.exp036.agent import get_factory_actions, get_robot_actions
from lux.kit import EnvConfig
env_cfg = EnvConfig()

class RandomAgent:
    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        return random.choice(actions)

    def observe(self, env, player, show=False):
        return [0.0]


class RuleBasedAgent(RandomAgent):
    def __init__(self, player):
        from agents_store.abishek.agent import Agent
        self.agent = Agent(player, env_cfg)

    def reset(self, env, show=False):
        self.agent.env_cfg = env.env.env_cfg
    
    def action(self, env, player, show=False):
        _obs = env.obs_list[-1][player]
        step = env.env.state.env_steps
        if step < 0:
            return self.agent.early_setup(step, _obs)
        else:
            return self.agent.act(step, _obs)


def print_outputs(env, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(prob, v)
    else:
        if v is not None:
            print('v = %f' % v)
        if prob is not None:
            print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, model, temperature=0.0, observation=True):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.hidden = None
        self.temperature = temperature
        self.observation = observation

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()

    def plan(self, obs):
        outputs = self.model.inference(obs, self.hidden)
        self.hidden = outputs.pop('hidden', None)
        return outputs

    def action(self, env, player, show=False):
        real_env_steps = env.env.state.real_env_steps
        assert real_env_steps >= 0
        obs = env.observation(player)
        outputs = self.plan(obs)
        actions = dict()
        game_state = env.get_game_state(player)
        actions = get_factory_actions(game_state, outputs['factory_policy'], player, actions)
        actions = get_robot_actions(game_state, outputs['robot_policy'], player, actions)
        return actions 


    def observe(self, env, player, show=False):
        v = None
        if self.observation:
            obs = env.observation(player)
            outputs = self.plan(obs)
            v = outputs.get('value', None)
            if show:
                print_outputs(env, None, v)
        return v


class ImitationAgent(RandomAgent):
    """ 
    評価用の模倣Agent
    """
    def __init__(self, player):
        from agents_store.exp019.agent import Agent
        # env_cfg.ROBOTS['LIGHT'].ACTION_QUEUE_POWER_COST = 0
        # env_cfg.ROBOTS['HEAVY'].ACTION_QUEUE_POWER_COST = 0
        self.agent = Agent(player, env_cfg)

    def reset(self, env, show=False):
        self.agent.env_cfg = env.env.env_cfg

    def action(self, env, player, show=False):
        _obs = env.obs_list[-1][player]
        step = env.env.state.env_steps
        if step < 0:
            return self.agent.early_setup(step, _obs)
        else:
            return self.agent.act(step, _obs)
