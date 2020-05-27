import numpy as np
import logging
import tensorflow as tf
from . import utils
from pypokerengine.players import BasePokerPlayer
from pprint import pprint, pformat

log = logging.getLogger(__name__)


class Player_v5(BasePokerPlayer):
    """ Epsilon-greedy policy with NN batch learning for Q-function approximation
    via NN (hole cards and community cards -> 3 discreet actions) """

    def __init__(self, batch=1000, memory=10 ** 5, alpha=0.1, gamma=0.8,
                 epsilon=0.05, inner_layers=(21,), name=None):
        super().__init__()

        self._history_reset()
        self._hole_card = None
        self._current_state = None
        self._round_start_stack = None

        suits = ['S', 'C', 'D', 'H']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self._state_ids = [s + r for s in suits for r in ranks]
        self._action_ids = ['fold', 'call', 'raise']

        self._w = tf.keras.models.Sequential()
        self._w.add(tf.keras.layers.Flatten(input_shape=(7, 52)))
        for i in inner_layers:
            self._w.add(tf.keras.layers.Dense(i, activation='relu'))
        self._w.add(tf.keras.layers.Dense(3))
        self._w_compile()

        self._loss = [0]
        self._batch = batch
        self._memory = memory
        self._D_states = np.zeros(shape=(memory, 7, 52), dtype=bool)
        self._D_rewards = np.zeros(shape=(memory, 3))
        self._D_i = 0
        self._age = 0

        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon

        self.name = name

    def _w_compile(self):
        self._w.compile(optimizer='rmsprop', loss='mse')

    def _action_random(self, valid_actions):
        """ Select random action and amount """
        action_id = np.random.choice(len(valid_actions))

        action = valid_actions[action_id]['action']
        amount_range = valid_actions[action_id]['amount']

        if not isinstance(amount_range, dict):
            amount = amount_range
        else:
            amount = round(np.random.uniform(low=amount_range['min'], high=amount_range['max']), -1)

        log.debug('{}: playing random {} {}'.format(self.name, action, amount))

        return action, amount

    def _action_e_greedy(self, valid_actions):
        """ Epsilon-greedy policy within 3 discreet actions (raise amount random) """
        Q = dict(zip(self._action_ids, self._current_state['Q']))

        valid_actions_ = {i['action']: i['amount'] for i in valid_actions}
        actions = {k: v for k, v in Q.items() if k in valid_actions_}

        log.debug('{}: Q values for state: {}'.format(self.name, utils.iter_round(actions)))

        if np.random.uniform() <= self._epsilon:  # exploration
            return self._action_random(valid_actions=valid_actions)
        else:  # exploitation
            action = max(actions, key=actions.get)

            if not isinstance(valid_actions_[action], dict):
                amount = valid_actions_[action]
            else:
                amount = round(np.random.uniform(low=valid_actions_[action]['min'],
                                                 high=valid_actions_[action]['max']), -1)

            log.debug('{}: playing greedy {} {}'.format(self.name, action, amount))

            return action, amount

    def _get_Q(self, state):
        """ NN approximation for Q values """
        Q = self._w(np.array([state])).numpy()[0]

        return Q

    def _learn(self):
        """ Update Q-values based on batch learning end of round update with function approximation """
        if len(self._history_actions) == 0: pass

        discounts = np.array([self._gamma ** i for i in range(len(self._history_rewards))])
        for i, v in enumerate(self._history_states):
            action = self._history_actions[i]
            action_id = self._action_ids.index(action)
            state, Q = v['state'], v['Q']

            # add to memory
            reward = np.sum(self._history_rewards[(i + 1):] * discounts[:-(i + 1)])
            Q_old = Q[action_id]
            Q[action_id] = Q_old + self._alpha * (reward - Q_old)
            self._D_rewards[self._D_i] = Q
            self._D_states[self._D_i] = state

            log.debug('{}: added to memory i={} updated reward for {} {:.2f}->{}'.format(
                self.name, self._D_i, action, Q_old,
                utils.iter_round(dict(zip(self._action_ids, self._D_rewards[self._D_i])))
            ))

            # train the model
            if (self._D_i + 1) % self._batch == 0:
                self._w.fit(
                    self._D_states[self._D_i + 1 - self._batch: self._D_i + 1],
                    self._D_rewards[self._D_i + 1 - self._batch: self._D_i + 1],
                    verbose=0
                )

                # evaluating on random subset of previous history
                subset_ids = np.random.choice(range(0, self._D_i + 1), size=self._batch, replace=False)
                loss = self._w.evaluate(
                    self._D_states[subset_ids],
                    self._D_rewards[subset_ids],
                    verbose=0
                )
                self._loss.append(loss)  # TODO: loss potentially uncapped

                log.debug('{}: retrained with loss {:.2f}->{:.2f} on {}:{}'.format(
                    self.name, self._loss[-2], self._loss[-1],
                    self._D_i + 1 - self._batch, self._D_i + 1))
                log.debug('\n' + self._w_to_str())

            self._D_i = (self._D_i + 1) % self._memory
            self._age += 1

    def _history_append(self, state=None, action=None, reward=None):
        """ Append any of history elements: state, action, reward """
        if state is not None: self._history_states.append(state)
        if action is not None: self._history_actions.append(action)
        if reward is not None: self._history_rewards.append(reward)

    def _history_reset(self):
        """ Reset all history of states, actions and rewards """
        self._history_states = []
        self._history_actions = []
        self._history_rewards = []

    def _calc_state(self, round_state):
        """ Calculate state ID and Q-values """
        state = np.full((7, 52), False)
        cards = self._hole_card + round_state['community_card']
        for round_id, card in enumerate(cards):
            card_id = self._state_ids.index(card)
            state[round_id, card_id] = True

        Q = self._get_Q(state)

        return {'state': state, 'Q': Q}

    def _calc_action(self, action, amount):
        """ Calculate action ID using just action taken """
        return action

    def _calc_reward(self, round_state, winners=None):
        """ Calculate reward as change in stack """
        if winners is None: return 0
        stack_change = self._get_stack(players=round_state['seats']) - self._round_start_stack

        return stack_change

    def _get_stack(self, players, uuid=None):
        """ Get current stack of any player (own stack by default) """
        if uuid is None: uuid = self.uuid

        stacks = [i['stack'] for i in players if i['uuid'] == uuid]
        return stacks.pop()

    """ Altering default methods """

    def __str__(self):
        ret = []
        ret.append(super().__str__())
        ret.append(f'batch={self._batch}, memory={self._memory}, alpha={self._alpha}, ' +
                   f'gamma={self._gamma}, epsilon={self._epsilon}, name={self.name}')
        ret.append(self._w_to_str())
        ret.append('Age is {}, last performance is {}'.format(
            self._age, utils.iter_round(self._loss[-3:])
        ))

        return '\n'.join(ret)

    def _w_to_str(self):
        """ Get string summary of a TensorFlow model """
        ret = []
        for i, v in enumerate(self._w.layers):
            layer = v.__class__.__name__

            if layer == 'Dense':
                weights = np.sort(np.abs(v.get_weights()[0].flatten()))
                max_weights = np.mean(weights[-5:])

                layer_str = 'Layer {}-{}, params {} (top-5={:.2f}, 0.1 top > {}, 0.01 top > {})'.format(
                    layer,
                    v.units,
                    v.count_params(),
                    max_weights,
                    (weights < max_weights / 10).sum(),
                    (weights < max_weights / 100).sum())
            else:
                layer_str = f'Layer {layer}'

            ret.append(layer_str)

        return '\n'.join(ret)

    def __setstate__(self, state):
        """ Pickle the TensorFlow model with config and weights separately """
        self.__dict__.update(state)

        config, weights = self._w['config'], self._w['weights']
        self._w = tf.keras.models.Sequential.from_config(config)
        self._w.set_weights(weights)
        self._w_compile()

    def __getstate__(self):
        """ Un-pickle the TensorFlow model with config and weights separately """
        config = self._w.get_config()
        weights = self._w.get_weights()
        self._w = {'config': config, 'weights': weights}

        return self.__dict__

    """ PyPokerEngine interfaces """

    def declare_action(self, valid_actions, hole_card, round_state):
        self._history_append(state=self._calc_state(round_state=round_state))
        self._history_append(reward=self._calc_reward(round_state=round_state))

        action, amount = self._action_e_greedy(valid_actions)
        self._history_append(action=self._calc_action(action=action, amount=amount))

        return action, amount

    def receive_game_start_message(self, game_info):
        self._round_start_stack = self._get_stack(players=game_info['seats'])

    def receive_round_start_message(self, round_count, hole_card, seats):
        self._hole_card = hole_card

    def receive_street_start_message(self, street, round_state):
        self._current_state = self._calc_state(round_state=round_state)

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        self._history_append(reward=self._calc_reward(round_state=round_state, winners=winners))

        self._learn()

        self._history_reset()
        self._round_start_stack = self._get_stack(round_state['seats'])
