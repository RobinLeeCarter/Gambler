from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

# import utils
import outcome


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        # hyperparameters
        # breaking ties
        self.action_value_round_dp: int = 7
        # accuracy
        self.theta_power_of_ten: int = -7
        self.theta = 10**self.theta_power_of_ten
        # self.p_heads = 0.5   # probability heads original
        # self.p_heads = 0.25  # probability heads first variation
        self.p_heads = 0.55  # probability heads second variation
        self.gamma = 1.0     # 0.99... is interesting
        self.minimal_action: bool = False    # take the minimal wager of the argmax or the maximal wager

        self.states = np.arange(101)
        self.non_terminal_states = self.states[1:-1]
        self.V = np.zeros(shape=self.states.shape, dtype=float)
        self.policy = np.zeros(shape=self.states.shape, dtype=float)

    def run(self):
        i: int = 0
        cont: bool = True

        while cont:
            delta: float = 0.0

            for state in self.non_terminal_states:
                v = self.V[state]
                action_values = self.get_action_values(state)
                self.V[state] = np.max(action_values)
                delta = max(delta, abs(v - self.V[state]))
            if self.verbose:
                print(f"iteration = {i}\tdelta = {round(delta, -self.theta_power_of_ten+1):f}")
            if delta < self.theta:
                cont = False
            i += 1

        # output deterministic policy with maximal action
        for state in self.non_terminal_states:
            action_values = self.get_action_values(state)
            action_values = np.round(action_values, 5)
            max_value = np.max(action_values)
            max_action_bool = (action_values == max_value)
            argmax_actions = np.flatnonzero(max_action_bool)

            if self.minimal_action:
                # prefer small bets if equally good
                self.policy[state] = np.min(argmax_actions)
            else:
                # prefer large bets to get the game over with if equally good
                self.policy[state] = np.max(argmax_actions)

        if self.verbose:
            print(self.V)
        self.graph_v()
        if self.verbose:
            print(self.policy)
        self.graph_policy()

    # get expected_value of each action
    def get_action_values(self, state: int) -> np.ndarray:
        max_action = min(state, 100-state)
        actions = np.arange(max_action+1)
        action_values = np.zeros(shape=actions.shape, dtype=float)
        for action in actions[1:]:  # exclude zero action to make the gambler always bet
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

    def graph_v(self):
        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()
        # x = np.arange(self.states_shape[0])
        # y = np.arange(self.states_shape[1])
        # x_grid, y_grid = np.meshgrid(x, y)

        # rows, cols = policy.shape
        ax.bar(self.states, self.V, width=1.0, align="edge")

        # extent=[0.5, cols-0.5, 0.5, rows-0.5]
        ax.set_xlim(xmin=0, xmax=100)
        ax.set_ylim(ymin=0, ymax=1.0)

        # contour_set = ax.contour(x_grid, y_grid, policy, levels=self.actions)
        # ax.clabel(contour_set, inline=True, fontsize=10)
        # ax.set_title(f'Policy {iteration}')
        plt.show()

    def graph_policy(self):
        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()
        # x = np.arange(self.states_shape[0])
        # y = np.arange(self.states_shape[1])
        # x_grid, y_grid = np.meshgrid(x, y)

        # rows, cols = policy.shape
        ax.bar(self.states, self.policy, width=1.0, align="edge")

        # extent=[0.5, cols-0.5, 0.5, rows-0.5]
        ax.set_xlim(xmin=0, xmax=100)
        # ax.set_ylim(ymin=0, ymax=20)

        # contour_set = ax.contour(x_grid, y_grid, policy, levels=self.actions)
        # ax.clabel(contour_set, inline=True, fontsize=10)
        # ax.set_title(f'Policy {iteration}')
        plt.show()
