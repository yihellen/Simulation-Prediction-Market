#!/usr/bin/env python3

import numpy as np

class BayesianAgent:
    """
    bayesian agent
    """

    def __init__(self, initial):
        """
        distribution: the type of distribution
        """
        self.belief = initial
        self.profit = [0, 0]
        # the first entry of self.profit is the profit earned when the first
        # outcome takes place, the second entry of self.profit is the profit
        # earned when the second outcome takes place


    def agent_belief_update(self, posted_num_trade, current_market_price):
        """
        The agent takes a look at the current market price, and adjusts his or
        her belief
        miu = (n*chi+miu)/(n+1)
        :return: posterior: the updated posterior
        """
        prior = self.belief
        self.belief = (posted_num_trade * current_market_price + prior)/\
        (posted_num_trade + 1)


    def agent_profit_update(self, market_belief):
        """
        Calculate the agent's profit
        """
        self.profit[0] += -1 * np.log(market_belief) + np.log(self.belief)
        self.profit[1] += -1 * np.log(1 - market_belief) + np.log(1 - self.belief)
