import numpy as np
import logging
from pypokerengine.players import BasePokerPlayer

log = logging.getLogger(__name__)


class Player_v0(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    """ Random actions player """

    def __init__(self, probs={}, name=None):
        super().__init__()
        self._probs = {
            3: [0.0, 0.95, 0.05],
            2: [0.0, 0.95],
            1: [1.0]
        }
        self._probs.update(probs)

        self.name = name

    def _action_random(self, valid_actions):
        n_actions = len(valid_actions)
        action_id = np.random.choice(n_actions, p=self._probs[n_actions])

        action = valid_actions[action_id]['action']
        amount_range = valid_actions[action_id]['amount']

        if not isinstance(amount_range, dict):
            amount = amount_range
        else:
            amount = round(np.random.uniform(low=amount_range['min'], high=amount_range['max']), -1)

        log.debug('{}: playing random {} {}'.format(self.name, action, amount))

        return action, amount

    def __str__(self):
        return f'name={self.name}, probs {self._probs}'

    def declare_action(self, valid_actions, hole_card, round_state):
        action, amount = self._action_random(valid_actions)
        return action, amount  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
