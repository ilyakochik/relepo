import numpy as np
from pypokerengine.players import BasePokerPlayer


class Player_v0(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    """ Random actions player """

    def _action_random(self, valid_actions):
        action_id = np.random.choice(len(valid_actions))

        action = valid_actions[action_id]['action']
        amount_range = valid_actions[action_id]['amount']

        if not isinstance(amount_range, dict):
            amount = amount_range
        else:
            amount = round(np.random.uniform(low=amount_range['min'], high=amount_range['max']), -1)

        return action, amount

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