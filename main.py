from pypokerengine.api.game import setup_config, start_poker

from relepo.player_v0 import Player_v0
from relepo.player_v1 import Player_v1
from relepo.player_v2 import Player_v2
from relepo.player_v3 import Player_v3
from relepo.player_v4 import Player_v4
import pandas as pd
import pickle
import plot_utils
import time
import logging
import sys

CONFIG = {
    'env': {'max_round': 100, 'initial_stack': 100, 'small_blind_amount': 5},
    'players': [
        ['v0', 'Player_v0', None, None, True, {}],
        # ['v0b', 'Player_v0', None, None, True, {}],
        # ['v1-train', 'Player_v1', None, 'pretrained/v1-1_10000_vs_v0.pickle', True, {}],
        # ['v1-test', 'Player_v1', 'pretrained/v1-1_10000_vs_v0.pickle', None, True, {}],
        # ['v2-train', 'Player_v2', None, 'pretrained/v2-1_2000_vs_v0.pickle', True, {}],
        # ['v2-test', 'Player_v2', 'pretrained/v2-1_2000_vs_v0.pickle', None, True, {}],
        # ['v3-train', 'Player_v3', None, 'pretrained/v3-1_2000_vs_v0.pickle', True, {}],
        ['v3-test', 'Player_v3', 'pretrained/v3-1_2000_vs_v0.pickle', None, True, {}],
        ['v4-train', 'Player_v4', 'pretrained/v4-1_vs_v0_v3.pickle', 'pretrained/v4-1_vs_v0_v3.pickle', True, {}],
    ],
    'run': 500,
    'verbose': 0
}


def calc_score(game_result):
    total_stacks = sum([i['stack'] for i in game_result['players']])
    scores = {i['name']: i['stack'] / total_stacks for i in game_result['players']}

    return scores


def run(config, num_episodes=1000, verbose=0):
    player_names = [i['name'] for i in config.players_info]
    scores = pd.DataFrame(columns=player_names)

    for i_episode in range(1, num_episodes + 1):
        game_result = start_poker(config, verbose=verbose)
        scores = pd.DataFrame.append(scores, calc_score(game_result), ignore_index=True)

        if i_episode % 100 == 0:
            print('\rTraining episode {}/{}, scores are {}'.
                  format(i_episode, num_episodes, scores.tail(100).mean().round(2).to_dict()), end='')
    print()
    print('Players performance:')
    print(scores.mean())
    print()

    # plot_utils.plot_scores(scores)


def env_create(config):
    print("Loading environment")
    pypoker_config = setup_config(**config['env'])

    for k, v in enumerate(config['players']):
        p_name, p_class, p_load, p_save, p_show, p_params = v
        if p_load:
            player = pickle.load(open(p_load, 'rb'))
        else:
            player = globals()[p_class](**p_params, name=p_name)

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

    print()


if __name__ == '__main__':
    logging_level = logging.DEBUG if CONFIG['verbose'] else logging.WARNING
    logging.basicConfig(level=logging_level, stream=sys.stdout,
                        format="\033[1;32m%(name)s\033[1;0m:%(levelname)s:%(message)s")

    time_start = time.time()

    env_config = env_create(CONFIG)

    if CONFIG['run']:
        run(env_config, num_episodes=CONFIG['run'], verbose=CONFIG['verbose'])

    env_save(CONFIG)

    print('Done in {:0.2f} sec'.format(time.time() - time_start))
    input("Press [enter] to continue")
