from pypokerengine.api.game import setup_config, start_poker
from relepo.relepo_player_v0 import relepo_player_v0
from relepo.relepo_player_v1 import relepo_player_v1
from pprint import pprint
import pickle

Q_LOAD = 1
Q_SAVE = 1
Q_FILE = 'relepo_player_v1_Q.pickle'
TRAIN = 1
DEMO = 0


def calc_score(game_result, eval_player):
    stacks = {i['name']: i['stack'] for i in game_result['players']}
    return stacks[eval_player] / sum(stacks.values())


def train(config, eval_player='p1', num_episodes=1000):
    scores = []
    for i_episode in range(num_episodes):
        print('\rEpisode {}/{}.'.format(i_episode + 1, num_episodes), end='')

        game_result = start_poker(config, verbose=0)

        scores.append(calc_score(game_result, eval_player))
    print()

    return sum(scores) / len(scores)


Q = None
if Q_LOAD:
    Q = pickle.load(open(Q_FILE, 'rb'))
    if isinstance(Q, dict): print("Q loaded (size: {})".format(len(Q)))

Q_sorted = {}
for k1, v1 in Q.items():
    for k2, v2 in v1.items():
        Q_sorted[(k1, k2)] = v2

pprint(sorted(Q_sorted.items(), key=lambda v: v[1]))

eval_player = relepo_player_v1(Q)
config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)
config.register_player(name='p1', algorithm=eval_player)
config.register_player(name='p2', algorithm=relepo_player_v0())

if TRAIN:
    performance = train(config, eval_player='p1', num_episodes=1000)
    print("Player performance is {}".format(performance))

if DEMO:
    eval_player.verbose = 1
    game_result = start_poker(config, verbose=1)

if Q_SAVE:
    pickle.dump(eval_player.Q, open(Q_FILE, 'wb'))
    print("Q saved (size: {})".format(len(eval_player.Q)))
