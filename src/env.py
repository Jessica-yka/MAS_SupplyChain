"""
Multi-Period Inventory Management Problem (IMP)

This file is the reimplementation of the IMP from the OR-Gym library, an open-source project developed to bring
reinforcement learning to the operations research community. OR-Gym is licensed under the MIT License. For more
information, please visit the OR-Gym GitHub repository: https://github.com/hubbs5/or-gym.
"""
from typing import Callable

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from config import env_configs, get_env_configs

from utils import visualize_state, parse_stage_agent_id

np.random.seed(0)


class InventoryManagementEnv(MultiAgentEnv):
    """
    Inventory Management Environment

    A multi-period, multi-echelon production-inventory system for a single non-perishable product sold in discrete
    quantities. Each stage in the supply chain consists of an inventory holding area and a production area. The
    inventory holding area stores the materials necessary for production at that stage. One unit of inventory
    produces one unit of product at each stage. There are lead times for transferring products between stages. The
    outgoing material from stage i serves as the feed material for production at stage i-1. Stages are numbered in
    ascending order: Stages = {0, 1, ..., M-1}, with the zeroth stage being the retailer. Production at each stage is
    limited by the stage's production capacity and available inventory.

    At the beginning of each time period, the following sequence of events occurs:

    1) Check deliveries: Each stage receives incoming inventory replenishment shipments that have arrived after the
       stage's respective lead time.
    2) Check orders and demands: Each stage places replenishment orders to their  respective suppliers. Replenishment
       orders are filled according to the available production capacity and inventory at the suppliers. Customer demand
       occurs at the retailer and is filled based on the available  inventory at the retailer.
    3) Deliver orders and demands: Each stage delivers as many products as possible to satisfy  downstream demand or
       replenishment orders. Unfulfilled sales and replenishment orders are backlogged, with backlogged sales taking
       priority in the following period.
    4) Compute profits: Each stage computes the profit and cost for product sales, material orders, backlog penalties,
       and surplus inventory holding costs.
    """

    def __init__(
        self, num_stages: int, num_agents_per_stage: int, num_periods: int, init_inventories: list, lead_times: list, demand_dist: str, demand_fn: Callable,
        prod_capacities: list, sale_prices: list, order_costs: list, backlog_costs: list, holding_costs: list, profit_rates: list, 
        supply_relations: dict, demand_relations: dict, stage_names: list, init_seed: int = 0):
        """
        Initialize the inventory management environment

        :param num_stages: number of stages (M)
        :param num_agents_per_stage: number of agents in each stage (X_m)
        :param num_periods: number of periods (N)
        :param init_inventories: initial inventory quantities (I0)
        :param lead_times: lead times (L)
        :param demand_fn: demand function (D)
        :param prod_capacities: production capacities (c)
        :param sale_prices: unit sale prices (p)
        :param order_costs: unit order costs (r)
        :param backlog_costs: unit backlog costs for unfulfilled orders (k)
        :param holding_costs: unit inventory holding costs (h)
        :param relations: the supply&demand relations of each companies {company: [upstream, downstream]}
        :param stage_names: stage names
        :param init_seed: initial seed
        """
        super().__init__()

        # Check the validity of inputs
        assert num_stages >= 2, "The number of stages should be at least 2."
        assert num_periods >= 1, "The number of periods should be at least 1."
        assert len(init_inventories) == num_stages * num_agents_per_stage, \
            "The number of initial inventories quantities should be the total number of agents in the environment (i.e., number of stages * number of agents per stage)."
        assert min(init_inventories) >= 0, "The initial inventory quantities should be non-negative."
        assert np.prod(lead_times.shape) == num_stages * num_agents_per_stage * num_agents_per_stage, "The number of lead times should be the total number of agent pair in the environment."
        assert np.min(lead_times) >= 0, "The lead times should be non-negative."
        assert len(prod_capacities) == num_stages * num_agents_per_stage, "The number of production capacities should be the total number of agents in the environment."
        assert np.min(prod_capacities) > 0, "The production capacities should be positive."
        assert len(sale_prices) == num_stages * num_agents_per_stage, "The number of unit sale prices should be the total number of agents in the environment."
        assert np.min(sale_prices) >= 0, "The unit sale prices should be non-negative."
        assert len(order_costs) == num_stages * num_agents_per_stage, "The number of unit order costs should be the total number of agents in the environment."
        assert np.min(order_costs) >= 0, "The unit order costs should be non-negative."
        assert len(backlog_costs) == num_stages * num_agents_per_stage, \
            "The number of unit backlog costs for unfulfilled orders should be the total number of agents in the environment."
        assert np.min(backlog_costs) >= 0, "The unit penalties for unfulfilled orders should be non-negative."
        assert len(holding_costs) == num_stages * num_agents_per_stage, \
            "The number of unit inventory holding costs should be the total number of agents in the environment."
        assert np.min(holding_costs) >= 0, "The unit inventory holding costs should be non-negative."
        assert len(profit_rates) == num_stages * num_agents_per_stage, \
            "The number of profit rates should be the total number of agents in the environment."
        assert len(supply_relations) == num_stages and len(supply_relations[0]) == num_agents_per_stage, \
            "The number of supply relations should be the total number of agents in the environment"
        assert len(demand_relations) == num_stages and len(demand_relations[0]) == num_agents_per_stage, \
            "The number of demand relations should be the total number of agents in the environment"
        assert len(stage_names) == num_stages, "The number of stage names should equal to the number of stages."

        # Set the environment configurations
        self.num_stages = num_stages
        self.num_agents_per_stage = np.max([num_agents_per_stage])
        self.num_periods = num_periods
        self.stage_names = stage_names
        self.demand_dist = demand_dist
        self.init_inventories = np.array(init_inventories, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.lead_times = np.array(lead_times, dtype=int).reshape(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage)
        self.max_lead_time = np.max(self.lead_times)
        self.demand_fn = demand_fn
        self.prod_capacities = np.array(prod_capacities, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.max_production = np.max(self.prod_capacities)
        self.sale_prices = np.array(sale_prices, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.order_costs = np.array(order_costs, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.backlog_costs = np.array(backlog_costs, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.holding_costs = np.array(holding_costs, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.profit_rates = np.array(profit_rates, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.supply_relations = supply_relations
        self.demand_relations = demand_relations

        # Create all variables
        self.period = 0
        self.inventories = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.orders = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.arriving_orders = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.sales = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.backlogs = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.demands = np.zeros(self.num_periods + 1, dtype=int)
        self.profits = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.total_profits = np.zeros(self.num_periods + 1, dtype=int)
        self.lead_time_pad = np.zeros(shape=(self.num_stages, self.num_agents_per_stage), dtype=int)

        # Compute the upper bounds for state variables
        max_production = self.max_production
        max_sale_price = np.max(self.sale_prices)
        max_order_cost = np.max(self.order_costs)
        max_backlog_cost = np.max(self.backlog_costs)
        max_holding_cost = np.max(self.holding_costs)
        max_lead_time = self.max_lead_time
        max_order = max_production
        max_inventory = max_order * self.num_periods

        # Set the observation and action spaces
        max_observations = np.concatenate((
            [max_production + 1, max_sale_price + 1, max_order_cost + 1, max_backlog_cost + 1, max_holding_cost + 1,
             max_lead_time + 1, max_inventory + 1, max_inventory + 1, max_inventory + 1],
            np.ones(2 * self.max_lead_time) * (max_order + 1)), axis=0)
        max_coop = np.array([1 for _ in range(num_agents_per_stage)]) # 1 represent a coop relation, 0 otherwise
        # self.agent_observation_space = spaces.MultiDiscrete(nvec=max_observations, seed=init_seed)
        # self.agent_action_order_space = spaces.Discrete(n=max_order + 1, start=0, seed=init_seed)
        self.observation_space = spaces.Dict({
            f"stage_{stage}_agent_{agent}": spaces.MultiDiscrete(nvec=max_observations, seed=init_seed + stage)
            for stage in range(self.num_stages) for agent in range(num_agents_per_stage)
        })
        self.action_order_space = spaces.Dict({
            f"stage_{stage}_agent_{agent}": spaces.Discrete(n=max_order + 1, start=0, seed=init_seed + stage)
            for stage in range(self.num_stages) for agent in range(num_agents_per_stage)
        })
        self.action_supply_space = spaces.Dict({
            f"stage_{stage}_agent_{agent}": spaces.MultiDiscrete(nvec=max_coop, seed=init_seed + stage)
            for stage in range(self.num_stages) for agent in range(num_agents_per_stage)
        })
        self.action_demand_space = spaces.Dict({
            f"stage_{stage}_agent_{agent}": spaces.MultiDiscrete(nvec=max_coop, seed=init_seed + stage)
            for stage in range(self.num_stages) for agent in range(num_agents_per_stage)
        })


        self.state_dict = {f"stage_{m}_agent_{x}": None for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        """
        Reset the environment variables including:
            - I: inventory at each stage
            - O: order placed by each stage
            - R: arriving order for each stage
            - S: sales by each stage
            - B: backlog for each stage
            - D: customer demand at the retailer
            - P: profit at each stage

        :param seed: seed for the new episode
        :param options: options
        :return: states, infos
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset all variables
        self.period = 0
        self.inventories.fill(0)
        self.orders.fill(0)
        self.arriving_orders.fill(0)
        self.sales.fill(0)
        self.backlogs.fill(0)
        self.demands.fill(0)
        self.profits.fill(0)
        self.total_profits.fill(0)

        # Set the initial condition and state
        self.inventories[:, :, 0] = self.init_inventories # (stage, agent, period)
        self.update_state()

        return self.state_dict, {}

    def update_state(self) -> None:
        """
        Update the environment state including the current stage features, inventory, backlog, upstream backlog,
        supply relations, demand relations, 
        previous sales, arriving deliveries, 

        State: s_{m,x,t} = [c_m_x, p_m_x, r_m_x, k_m_x, h_m_x, L_m_x, I_{m,x,t-1}, B_{m,x,t-1}, B_{m+1,x,t-1},
        sup_rel, dem_rel, 
        S_{m,x,t-L_max}, ..., S_{m,x,t-1}, 0, ..., 0, R_{m,x,t-L_m}, ..., R_{m,x,t-1}]
        """
        t = self.period
        states = dict()
        states["prod_capacities"] = self.prod_capacities
        states["sale_prices"] = self.sale_prices
        states["order_costs"] = self.order_costs
        states["backlog_costs"] = self.backlog_costs
        states["holding_costs"] = self.holding_costs
        states["lead_times"] = self.lead_times
        states["inventories"] = self.inventories[:, :, t]
        states["backlogs"] = self.backlogs[:, :, t]
        states["upstream_backlogs"] = np.zeros((self.num_stages, self.num_agents_per_stage), dtype=int)
        states["upstream_backlogs"][:-1] = self.backlogs[1:, :, t]
        states["suppliers"] = self.supply_relations
        states["customers"] = self.demand_relations

        lt_max = self.max_lead_time
        states["recent_sales"] = np.zeros(shape=(self.num_stages, self.num_agents_per_stage, lt_max), dtype=int)
        if t >= lt_max:
            states["recent_sales"][:, :, (-2 * lt_max):-lt_max] = self.sales[:, :, (t - lt_max + 1):(t + 1)]
        elif t > 0:
            states["recent_sales"][:, :, (-lt_max - t):-lt_max] = self.sales[:, :, 1:(t + 1)]

        states["arriving_deliveries"] = np.zeros(shape=(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage, lt_max), dtype=int)
        for m in range(self.num_stages):
            for x in range(self.num_agents_per_stage):
                for j in range(self.num_agents_per_stage):
                    if self.supply_relations[m][x][j] == 1:
                        lt = self.lead_times[m][x][j]
                        if t >= lt:
                            states["arriving_deliveries"][m, x, j, (-2 * lt_max):-lt_max] = self.arriving_orders[m, x, j, (t - lt + 1):(t + 1)]
                        elif t > 0:
                            states["arriving_deliveries"][m, x, j, (-lt_max - t):-lt_max] = self.arriving_orders[m, x, j, 1:(t + 1)]

        # self.state_dict = {f"stage_{m}_agent_{x}": states[m][x] for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
        self.state_dict = {}
        for m in range(self.num_stages):
            for x in range(self.num_agents_per_stage):
                agent_state = []
                agent_state.append(states["prod_capacities"][m][x])
                agent_state.append(states["sale_prices"][m][x])
                agent_state.append(states["order_costs"][m][x])
                agent_state.append(states["backlog_costs"][m][x])
                agent_state.append(states["holding_costs"][m][x])
                agent_state.append(states["lead_times"][m][x])
                agent_state.append(states["inventories"][m][x])
                agent_state.append(states["backlogs"][m][x])
                agent_state.append(states["upstream_backlogs"][m][x])
                agent_state.append(states["suppliers"][m][x])
                agent_state.append(states["customers"][m][x])
                agent_state.append(states["recent_sales"][m][x])
                agent_state.append(states["arriving_deliveries"][m][x])

                self.state_dict[f"stage_{m}_agent_{x}"] = agent_state

        return self.state_dict

    def step(self, order_dict: dict[str, int], sup_dict: dict[str, list], dem_dict: dict[str, list]) -> tuple[dict, dict, dict, dict, dict]:
        """
        Take a step and return the next observation

        :param action_dict: action (order quantity) for each stage
        :return: states, rewards, terminations, truncations, infos
        """
        assert np.all(f"stage_{m}_agent_{x}" in order_dict for m in range(self.num_stages) for x in range(self.num_agents_per_stage)), \
            "Order quantities for all stages are required."
        assert np.all(order_dict[f"stage_{m}_agent_{x}"] >= 0 for m in range(self.num_stages) for x in range(self.num_agents_per_stage)), \
            "Order quantities must be non-negative integers."

        # Get the inventory at the beginning of the period
        self.period += 1
        t = self.period
        M = self.num_stages
        current_inventories = self.inventories[:, :, t - 1]
        self.orders[:, :, :, t] = np.stack([order_dict[f"stage_{m}_agent_{x}"]*self.supply_relations[m][x] for m in range(self.num_stages) for x in range(self.num_agents_per_stage)]).reshape(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage)
        self.supply_relations = np.stack([sup_dict[f"stage_{m}_agent_{x}"] for m in range(self.num_stages) for x in range(self.num_agents_per_stage)]).reshape(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage)                                                                                                                                    
        # self.demand_relations = np.stack([dem_dict[f"stage_{m}_agent_{x}"] for m in range(self.num_stages) for x in range(self.num_agents_per_stage)]).reshape(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage)
        
        self.demands[t] = int(self.demand_fn(t))

        # Add the delivered orders
        # I_{m,t} <- I_{m,t-1} + R_{m,t-L_m} (after delivery)
        for m in range(self.num_stages):
            for i in range(self.num_agents_per_stage):
                for j in range(self.num_agents_per_stage):
                    if self.supply_relations[m][i][j] == 1:
                        lt = self.lead_times[m][i][j]
                        if t >= lt:
                            current_inventories[m][i][j] += self.arriving_orders[m, i, j, t - lt] # the order placed before the lead time

        # Compute the fulfilled orders
        # R_{m,t} = min(B_{m+1,t-1} + O_{m,t}, I_{m+1,t-1} + R_{m+1,t-L_{m+1}}, c_{m+1}), m = 0, ..., M - 2
        cum_orders = np.sum(self.orders, axis=1)
        print("cum orders\n", cum_orders)
        print("orders\n", self.orders[:-1, :, :, t])
        print("backlogs\n", self.backlogs[1:, :, t - 1])
        print("current inventories\n", current_inventories[1:])
        self.arriving_orders[:-1, :, t] = np.minimum(
            np.minimum(self.backlogs[1:, :, t - 1] + cum_orders[:-1, :, t], current_inventories[1:]),
            self.prod_capacities[1:])
        # R_{M-1,t} = O_{M-1,t}
        self.arriving_orders[M - 1, :, t] = cum_orders[M - 1, :, t] # the manufacturers at the top of supply chain
        print("arriving orders\n", self.arriving_orders[:, :, t])
        exit()
        # Compute the sales
        cum_arriving_orders = np.sum(self.arriving_orders, axis=2)
        # S_{m,t} = R_{m-1,t}, m = 1, ..., M - 1
        self.sales[1:, :, t] =cum_arriving_orders[:-1, :, t]
        # S_{0,t} = min(B_{0,t-1} + D_{t}, I_{0,t-1} + R_{0,t-L_m}, c_0)

        self.sales[0, :, t] = np.minimum(
            np.minimum(self.backlogs[0, :, t - 1] + self.demands[t], current_inventories[0]),
            self.prod_capacities[0])
        
        # Compute the backlogs
        # B_{m,t} = B_{m,t-1} + O_{m-1,t} - S_{m,t}, m = 1, ..., M - 1
        self.backlogs[1:, :, t] = self.backlogs[1:, :, t - 1] + cum_orders[:-1, :, t] - self.sales[1:, :, t]
        # B_{0,t} = B_{0,t-1} + D_{t} - S_{0,t}
        self.backlogs[0, :, t] = self.backlogs[0, :, t - 1] + self.demands[t] - self.sales[0, :, t]

        # Compute the inventory at the end of the period
        # I_{m,t} = I_{m,t-1} + R_{m,t-L_m} - S_{m,t} (after sales)
        self.inventories[:, :, t] = current_inventories - self.sales[:, :, t]

        # Compute the profits
        # P_{m,t} = p_m S_{m,t} - r_m R_{m,t} - k_m B_{m,t} - h_m I_{m,t}
        print("order costs", self.order_costs)
        print("arriving orders", self.arriving_orders[:, :, :, t])
        exit()
        self.profits[:, :, t] = self.sale_prices * self.sales[:, :, t] - self.order_costs * self.arriving_orders[:, :, :, t] \
                             - self.backlog_costs * self.backlogs[:, :, t] - self.holding_costs * self.inventories[:, :, t] \
        

        self.total_profits[t] = np.sum(self.profits[:, :, t])

        # Determine rewards and terminations
        rewards = {f"stage_{m}_agent_{x}": self.profits[m, x, t] for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
        all_termination = self.period >= self.num_periods
        terminations = {f"stage_{m}_agent_{x}": all_termination for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
        terminations["__all__"] = all_termination
        truncations = {f"stage_{m}_agent_{x}": False for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
        truncations["__all__"] = False
        infos = {f"stage_{m}_agent_{x}": {} for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}

        # Update the state
        self.update_state()

        return self.state_dict, rewards, terminations, truncations, infos

    def _parse_state(self, state: list, stage_agent_id_name: str) -> dict:
        """
        Parse a single stage state

        :param state: state
        :return: parsed state
        """
        lt_max = self.max_lead_time

        return {
            'prod_capacity': state[0],
            'sale_price': state[1],
            'num_periods_cost': state[2],
            'backlog_cost': state[3],
            'holding_cost': state[4],
            'lead_times': state[5],
            'inventory': state[6],
            'backlog': state[7],
            'upstream_backlog': state[8],
            "suppliers": state[9], 
            "customers": state[10], 
            'sales': state[11].tolist(),
            'deliveries': state[12].tolist(),
        }

    def parse_state(self, state_dict: dict = None) -> dict:
        """
        Parse the state dictionary

        :param state_dict: state dictionary
        :return: parsed state dict
        """
        if state_dict is None:
            state_dict = self.state_dict

        parsed_state = {}

        for stage_agent_id_name, state in state_dict.items():
            parsed_state[stage_agent_id_name] = self._parse_state(state, stage_agent_id_name)

        return parsed_state


def env_creator(env_config):
    """
    Create the environment
    """

    return InventoryManagementEnv(
        num_stages=env_config['num_stages'],
        num_agents_per_stage=env_config['num_agents_per_stage'],
        num_periods=env_config['num_periods'],
        init_inventories=env_config['init_inventories'],
        lead_times=env_config['lead_times'],
        demand_dist=env_config['demand_dist'],
        demand_fn=env_config['demand_fn'],
        prod_capacities=env_config['prod_capacities'],
        sale_prices=env_config['sale_prices'],
        order_costs=env_config['order_costs'],
        backlog_costs=env_config['backlog_costs'],
        holding_costs=env_config['holding_costs'],
        profit_rates=env_config['profit_rates'],
        supply_relations=env_config['supply_relations'],
        demand_relations=env_config['demand_relations'], 
        stage_names=env_config['stage_names'],
    )


if __name__ == '__main__':

    ec = get_env_configs(env_configs['basic'])
    im_env = env_creator(env_config=ec)
    im_env.reset()
    print(f"stage_names = {im_env.stage_names}")
    print(f"state_dict = {im_env.state_dict}")
    print(f"state_dict = {im_env.parse_state(im_env.state_dict)}")
    print(f"observation_space = {im_env.observation_space}")
    print(f"observation_sample = {im_env.observation_space.sample()}")
    print(f"action_order_space = {im_env.action_order_space}")
    print(f"action_order_sample = {im_env.action_order_space.sample()}")
    print(f"action_supply_space = {im_env.action_supply_space}")
    print(f"action_demand_space = {im_env.action_demand_space}")
    visualize_state(env=im_env, rewards={}, t=-1, save_prefix="test")
    num_agents_per_stage = im_env.num_agents_per_stage

    for t in range(im_env.num_periods):
        print("period", t)
        sup_dict = {}
        dem_dict = {}
        for m in range(im_env.num_stages):
            for x in range(im_env.num_agents_per_stage):
                if m == 0: # retailer
                    sup_dict[f"stage_{m}_agent_{x}"] = np.zeros(im_env.num_agents_per_stage, dtype=int)
                    sup_dict[f"stage_{m}_agent_{x}"][(t+x+1)%num_agents_per_stage] = 1
                    dem_dict[f"stage_{m}_agent_{x}"] = np.zeros(im_env.num_agents_per_stage, dtype=int)
                    dem_dict[f"stage_{m}_agent_{x}"][0] = 1
                elif m == im_env.num_stages - 1: # manufacturer
                    sup_dict[f"stage_{m}_agent_{x}"] = np.zeros(im_env.num_agents_per_stage, dtype=int)
                    sup_dict[f"stage_{m}_agent_{x}"][0] = 1
                    dem_dict[f"stage_{m}_agent_{x}"] = np.zeros(im_env.num_agents_per_stage, dtype=int)
                    dem_dict[f"stage_{m}_agent_{x}"][(t+x+1)%num_agents_per_stage] = 1
                else:
                    sup_dict[f"stage_{m}_agent_{x}"] = np.zeros(im_env.num_agents_per_stage, dtype=int)
                    sup_dict[f"stage_{m}_agent_{x}"][(t+x+1)%num_agents_per_stage] = 1
                    dem_dict[f"stage_{m}_agent_{x}"] = np.zeros(im_env.num_agents_per_stage, dtype=int)
                    dem_dict[f"stage_{m}_agent_{x}"][(t+x+1)%num_agents_per_stage] = 1
        print("sup_dict", sup_dict)

        next_state_dict, rewards, terminations, truncations, infos = im_env.step(
            order_dict={f"stage_{m}_agent_{x}": np.array([4 for _ in range(num_agents_per_stage)]) for m in range(im_env.num_stages) for x in range(im_env.num_agents_per_stage)}, 
            sup_dict=sup_dict,
            dem_dict=dem_dict
        )
        
        print('-' * 80)
        print(f"period = {t}")
        print(f"next_state_dict = {next_state_dict}")
        print(f"next_state_dict = {im_env.parse_state(next_state_dict)}")
        print(f"rewards = {rewards}")
        print(f"terminations = {terminations}")
        print(f"truncations = {truncations}")
        print(f"infos = {infos}")
        visualize_state(env=im_env, rewards=rewards, t=t, save_prefix='test')

