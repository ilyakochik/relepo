import pandas as pd
import pickle
import time
import logging
import sys
from pypokerengine.api.game import setup_config, start_poker
from pathlib import Path
from pympler import asizeof
from relepo.player_v0 import Player_v0
from relepo.player_v1 import Player_v1
from relepo.player_v2 import Player_v2
from relepo.player_v3 import Player_v3
from relepo.player_v4 import Player_v4
from relepo.player_v5 import Player_v5
from relepo.player_v6 import Player_v6
from relepo.player_v7 import Player_v7

CONFIG = {
    'env': {'max_round': 100, 'initial_stack': 100, 'small_blind_amount': 5},
    'folder': 'pretrained/',
    'players': [
        ['v0a', 'Player_v0', None, {}, 0],
        # ['v4', 'Player_v4', 'load', {}, 0],
        # ['v5-nn182-21', 'Player_v5', 'update', {'inner_layers': (182, 21)}, 0],
        # ['v5-nn104', 'Player_v5', 'update', {'inner_layers': (104,)}, 0],
        # ['v6-nn104', 'Player_v6', 'update', {'inner_layers': (104,)}, 0],
        ['v7-nn104', 'Player_v7', '', {'hidden_layers': (104, ), 'batch': 3}, logging.DEBUG],
    ],
    'run': 10,
    'verbose': 1
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

    # utils.plot_scores(scores)


def env_create(config):
    print("Loading environment")
    pypoker_config = setup_config(**config['env'])

    for k, v in enumerate(config['players']):
        p_name, p_class, p_store, p_params, p_logging = v

        # load or create file
        pickle_file = Path(config['folder'] + p_name + '.pickle')
        if p_store in ['load', 'update'] and pickle_file.is_file():
            player = pickle.load(pickle_file.open(mode='rb'))
            print('\033[1;34m{}\033[1;0m {}'.format('Loaded', player))
            print('{:.2f} Kb'.format(asizeof.asizeof(player) / 2 ** 10))
        else:
            player = globals()[p_class](**p_params, name=p_name)
            print('\033[1;34m{}\033[1;0m {}'.format('Created', player))
            print('{:.2f} Kb'.format(asizeof.asizeof(player) / 2 ** 10))

        config['players'][k][1] = player
        pypoker_config.register_player(name=p_name, algorithm=player)

        # set-up logging
        logging.getLogger(globals()[p_class].__module__).setLevel(
            p_logging if p_logging else logging.WARNING
        )  # TODO setting at class level, not at instance

    print()
    return pypoker_config


def env_save(config):
    for p_name, p_instance, p_store, p_params, p_logging in config['players']:
        if p_store in ['save', 'update']:
            pickle_file = Path(config['folder'] + p_name + '.pickle')

            print('\033[1;34m{}\033[1;0m {} to {}'.format('Saved', p_name, pickle_file))
            pickle.dump(p_instance, pickle_file.open(mode='wb'))

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
    # input("Press [enter] to continue")
