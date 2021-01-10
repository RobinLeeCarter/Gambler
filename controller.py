from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure, cm

# import utils
import outcome


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        # hyperparameters
        self.theta = 0.1    # accuracy
        self.p_heads = 0.4  # probability heads
        self.gamma = 1.0

        self.states = np.arange(101)
        self.V = np.zeros(shape=self.states.shape, dtype=float)
        self.policy = np.zeros(shape=self.states.shape, dtype=float)

    def run(self):
        i: int = 0
        cont: bool = True

        while cont:
            delta: float = 0.0
            for state in self.states:
                v = self.V[state]
                action_values = self.get_action_values(state)
                self.V[state] = np.max(action_values)
                delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                cont = False

        # output deterministic policy with maximal action
        for state in self.states:
            action_values = self.get_action_values(state)
            max_value = np.max(action_values)
            max_action_bool = (action_values == max_value)
            argmax_actions = np.flatnonzero(max_action_bool)
            # prefer large bets over small or zero bets to get the game over with if equally good
            self.policy[state] = np.max(argmax_actions)

    # get expected_value of each action
    def get_action_values(self, state: int) -> np.ndarray:
        max_action = min(state, 100-state)
        actions = np.arange(max_action+1)
        action_values = np.zeros(shape=actions.shape, dtype=float)
        for action in actions:
            outcomes = self.get_outcomes(state, action)
            for outcome_ in outcomes:
                action_values[action] += outcome_.p * (outcome_.reward + self.gamma * self.V[outcome_.new_state])
        return action_values

    def get_outcomes(self, state: int, action: int) -> List[outcome.Outcome]:
        # heads
        new_state = state + action
        heads_outcome = outcome.Outcome(p=self.p_heads, new_state=new_state)
        if new_state == 100:
            heads_outcome.reward = 1.0
            heads_outcome.is_terminal = True

        # tails
        new_state = state - action
        tails_outcome = outcome.Outcome(p=1-self.p_heads, new_state=new_state)
        if new_state == 0:
            tails_outcome.is_terminal = True

        return [heads_outcome, tails_outcome]
