#!/usr/bin/env python3
"""

LMSR based prediction market for events that follow Bernoulli distribution
Main Executable

"""
import os
import argparse
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from Agents.bayesian_agent import BayesianAgent


def parse_command_line():
    """
    argument parser
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "true_probability", type=float, help="specify the true probability of the\
         occurrence of an outcome"
    )
    parser.add_argument(
        "num_iteration", type=int, help="number of repeated experiments"
    )
    parser.add_argument(
        "num_agents", type=int, help="specify the number of agents in the market"
    )
    # parser.add_argument(
    #     "epsilon", type=float, help="the maximum difference in prices upon\
    #      convergence"
    # )
    parser.add_argument('--budget', dest='budget', action='store_true')
    parser.add_argument('--no-budget', dest='budget', action='store_false')
    parser.add_argument(
        "budget", type=float, help="specify range of budget"
    )
    args = parser.parse_args()

    return args


def main():

    args = parse_command_line()
    true_prob = args.true_probability
    # epsilon = args.epsilon
    num_agents = args.num_agents
    num_iteration = args.num_iteration
    true_outcomes = bernoulli.rvs(size=num_iteration,p=1-true_prob)
    # do the same for the scenario where the agents are not constrained by budgets
    # and the scenario where the agents are constrained by budgets
    # initialize the beliefs of each agents (Uniform distribution between 0 & 1)
    agent_initial_beliefs = np.random.uniform(size=(num_agents,))
    if (args.budget):
        agent_initial_budget = np.random.uniform(low=0.0, high=args.budget, size=(num_agents,))
    agent_profit = np.zeros((num_agents, ))
    agent_idx_list = list(np.arange(num_agents))
    # initialize the budget of each agent （see the result with no budget
    # constraint to decide what values the budgets should take)
    folder = 'sim_res_budget_{}_iter_{}_prob_{}_agentnum_{}'.format(int(args.budget), num_iteration, true_prob, num_agents)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for trial in range(num_iteration):
        true_outcome = true_outcomes[trial]
        agents = []
        if (args.budget):
            for i in range(num_agents):
                agents.append(BayesianAgent(agent_initial_beliefs[i], agent_initial_budget[i]))
        else:
            for i in range(num_agents):
                agents.append(BayesianAgent(agent_initial_beliefs[i]))
        market_belief = 0.5 # initial belief of the market maker
        prev_market_belief = 0.5
        iter = 0
        random.shuffle(agent_idx_list)
        while (iter <  num_agents):
            agent_idx = agent_idx_list[iter]
            # agent update belief based on market prices
            agent = agents[agent_idx]
            if (args.budget):
                agent.agent_belief_update(market_belief)
            agent.agent_profit_update(market_belief)
            prev_market_belief = market_belief
            market_belief = agent.belief
            iter += 1
        for i in range(num_agents):
            agent_profit[i] += agents[i].profit[true_outcome]
        agent_current_beliefs = []
        for i in range(num_agents):
            agent_current_beliefs.append(agents[i].belief)
        agent_temp_profit = []
        for i in range(num_agents):
            agent_temp_profit.append(agents[i].profit[true_outcome])
        # plot the graph of the agents' profit against the agents' current belief
        plt.scatter(agent_current_beliefs, agent_temp_profit)
        plt.xlabel('Agent current beliefs on outcome 0 (the first outcome)')
        plt.ylabel('Agent earned profit')
        plt.title('Agents\' profits on the {}-th trial when outcome is {}'.format(trial, true_outcome))
        plt.savefig('./{}/profit_curr_belief_trial_{}.png'.format(folder, trial))
        plt.close()

    # plot the graph of the agents' profit against the agents' initial belief
    plt.scatter(agent_initial_beliefs, agent_profit / num_iteration)
    plt.xlabel('Agent initial belief on outcome 0 (the first outcome)')
    plt.ylabel('Agent earned profit')
    plt.title('Agents\' profits when true probability for outcome 0 is {} (num_iter = {})'.format(true_prob, num_iteration))
    plt.savefig('./{}/profit_init_belief_prob_{}_iter_{}.png'.format(folder, true_prob, num_iteration))
    plt.close()

    # if there is budget, plot agents' budget against the agents' profit
    if (args.budget):
        plt.scatter(agent_initial_budget, agent_profit / num_iteration)
        plt.xlabel('Agent initial budget')
        plt.ylabel('Agent earned profit')
        plt.title('Agents\' profits when true probability for outcome 0 is {} (num_iter = {})'.format(true_prob, num_iteration))
        plt.savefig('./{}/profit_init_budget_prob_{}_iter_{}.png'.format(folder, true_prob, num_iteration))
        plt.close()






if __name__ == '__main__':
    main()
