#!/usr/bin/env python3

import numpy as np

class BayesianAgent:
    """
    bayesian agent
    """

    def __init__(self, initial, budget=0):
        """
        distribution: the type of distribution
        """
        self.belief = initial
        self.profit = [0, 0]
        self.budget = budget
        # the first entry of self.profit is the profit earned when the first
        # outcome takes place, the second entry of self.profit is the profit
        # earned when the second outcome takes place


    def agent_belief_update(self, market_belief):
        """
        The agent will update his or her belief based on the current budget and market belief
        """

        if (self.belief > 1 - 1/(np.exp(self.budget)) + market_belief/(np.exp(self.budget))):
            self.belief = 1 - 1/(np.exp(self.budget)) + market_belief/(np.exp(self.budget))
        elif (self.belief < market_belief / np.exp(self.budget)):
            self.belief = market_belief / np.exp(self.budget)


    def agent_profit_update(self, market_belief):
        """
        Calculate the agent's profit
        """
        self.profit[0] += -1 * np.log(market_belief) + np.log(self.belief)
        self.profit[1] += -1 * np.log(1 - market_belief) + np.log(1 - self.belief)
