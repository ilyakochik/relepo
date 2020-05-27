import numpy as np
import logging
from pypokerengine.players import BasePokerPlayer
from pprint import pprint, pformat

log = logging.getLogger(__name__)


class Player_v1(BasePokerPlayer):
    """ End-of episode MC-alpha control storing only hole cards and action without amount """

    def __init__(self, alpha=0.01, gamma=0.8, epsilon=0.05, name=None):
        super().__init__()

        self._history_reset()
        self._hole_card = None
        self._current_state = None
        self._round_start_stack = None

        self._Q = {}
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon

        self.name = name

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
        if self._current_state not in self._Q: return self._action_random(valid_actions=valid_actions)

        valid_actions_ = {i['action']: i['amount'] for i in valid_actions}
        actions = {k: v for k, v in self._Q[self._current_state].items() if k in valid_actions_}

        if len(actions) == 0: return self._action_random(valid_actions=valid_actions)

        log.debug('{}: Q values for state {}: {}'.format(self.name, self._current_state, actions))

        if np.random.uniform() <= self._epsilon:  # exploration
            return self._action_random(valid_actions=valid_actions)
        else:  # exploitation
            action = max(actions, key=actions.get)

            if not isinstance(valid_actions_[action], dict):
                amount = valid_actions_[action]
            else:
                amount = round(np.random.uniform(low=valid_actions_[action]['min'], high=valid_actions_[action]['max']),
                               -1)

            log.debug('{}: playing greedy {} {}'.format(self.name, action, amount))

            return action, amount

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
        """ Calculate state ID using hole cards only """

        return tuple(np.sort(self._hole_card))

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

    def _updateQ_alpha(self):
        """ Update Q-values based on MC-alpha end of round update """
        if len(self._history_actions) == 0: pass
        discounts = np.array([self._gamma ** i for i in range(len(self._history_rewards))])
        for i, state in enumerate(self._history_states):
            action = self._history_actions[i]

            if state not in self._Q: self._Q[state] = {}
            if action not in self._Q[state]: self._Q[state][action] = 0

            Q_old = self._Q[state][action]
            self._Q[state][action] += self._alpha * (
                    np.sum(self._history_rewards[(i + 1):] * discounts[:-(i + 1)]) - Q_old)

            log.debug('{}: updating Q for {} {} from {:0.2f} to {:0.2f} (rewards {})'.
                      format(self.name, state, action, Q_old, self._Q[state][action], self._history_rewards))

        # pprint(self.Q)

    def __str__(self):
        ret_str = super().__str__() + '\n'
        ret_str += 'alpha={}, gamma={}, epsilon={}, name={}\n'. \
            format(self._alpha, self._gamma, self._epsilon, self.name)

        Q_sorted = []

        for k1, v1 in self._Q.items():
            for k2, v2 in v1.items():
                Q_sorted.append([(k1[0][1], k1[1][1], k2), round(v2, 0)])

        Q_sorted = sorted(Q_sorted, key=lambda v: v[1])
        last_n = min(5, len(Q_sorted))
        ret_str += 'Q values (last {} of {})\n'.format(last_n, len(self._Q))
        ret_str += pformat(Q_sorted[-last_n:])

        return ret_str

    # we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        self._history_append(state=self._calc_state(round_state=round_state))
        self._history_append(reward=self._calc_reward(round_state=round_state))

        action, amount = self._action_e_greedy(valid_actions)
        self._history_append(action=self._calc_action(action=action, amount=amount))

        return action, amount  # action returned here is sent to the poker engine

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

        self._updateQ_alpha()

        self._history_reset()
        self._round_start_stack = self._get_stack(round_state['seats'])
