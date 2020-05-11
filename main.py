from pypokerengine.api.game import setup_config, start_poker
from relepo.player_v0 import Player_v0
from relepo.player_v1 import Player_v1
from pprint import pprint
import numpy as np
import pandas as pd
import pickle

CONFIG = {
    'env': {'max_round': 100, 'initial_stack': 100, 'small_blind_amount': 5},
    'players': [
        ['v0', 'Player_v0', None, None, True, {}],
        # ['v1q', 'Player_v1', 'v1-1_10000_vs_v0.pickle', None, True, {}],
        ['v1', 'Player_v1', 'v1-1_10000_vs_v0.pickle', None, True, {}]
    ],
    'train': 1000,
    'demo': 0
}


def calc_score(game_result):
    total_stacks = sum([i['stack'] for i in game_result['players']])
    scores = {i['name']: i['stack'] / total_stacks for i in game_result['players']}

    return scores


def train(config, num_episodes=1000):
    player_names = [i['name'] for i in config.players_info]
    scores = pd.DataFrame(columns=player_names)

    for i_episode in range(1, num_episodes + 1):
        game_result = start_poker(config, verbose=0)
        scores = pd.DataFrame.append(scores, calc_score(game_result), ignore_index=True)

        if i_episode % 100 == 0:
            print('\rTraining episode {}/{}, scores are {}.'.
                  format(i_episode, num_episodes, scores.tail(100).mean().round(2).to_dict()), end='')

    print()
    print('Players performance:')
    print(scores.mean())
    print()


def env_create(config):
    print("Loading environment")
    pypoker_config = setup_config(**config['env'])

    for k, v in enumerate(config['players']):
        p_name, p_class, p_load, p_save, p_show, p_params = v
        if p_load:
            player = pickle.load(open(p_load, 'rb'))
        else:
            player = globals()[p_class](**p_params)

        if p_show:
            print(player)

        config['players'][k][1] = player
        pypoker_config.register_player(name=p_name, algorithm=player)

    print()
    return pypoker_config


def env_save(config):
    for p_name, p_instance, p_load, p_save, p_show, p_params in config['players']:
        if p_save:
            print('{} saved to {}'.format(p_name, p_save))
            pickle.dump(p_instance, open(p_save, 'wb'))


if __name__ == '__main__':
    env_config = env_create(CONFIG)

    if CONFIG['train']:
        train(env_config, num_episodes=CONFIG['train'])

    if CONFIG['demo']:
        for p_name, p_instance, p_load, p_save, p_show, p_params in CONFIG['players']:
            if 'verbose' in p_instance.__dict__.keys(): p_instance.verbose = 1

        for i in range(CONFIG['demo']):
            start_poker(env_config, verbose=1)

    env_save(CONFIG)
