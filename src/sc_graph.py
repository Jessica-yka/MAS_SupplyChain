import numpy as np
import networkx as nx
import copy
from collections import defaultdict

class agent_profile():
    def __init__(self, stage_idx, agent_idx, role, prod_capacity, sale_price, init_inventory, backlog_cost: int, 
                 holding_cost: int, order_cost: int, prod_cost: int, lead_times, suppliers, customers, delievery, 
                 downstream_agents: list, upstream_agents: list):
        self.name = f"stage_{stage_idx}_agent_{agent_idx}"
        self.role = role
        self.stage = stage_idx
        self.agent = agent_idx
        self.prod_capacity = prod_capacity
        self.inventory = init_inventory
        self.order_cost = order_cost
        self.lead_times = lead_times
        self.suppliers = suppliers
        self.customers = customers
        self.backlog_cost = backlog_cost
        self.holding_cost = holding_cost
        self.prod_cost = prod_cost
        self.downstream_agents = downstream_agents
        self.upstream_agents = upstream_agents

        
        self.sale_price = sale_price
        self.sales = 0
        
        self.backlog = 0
        
        self.fulfilled_rate = 1
        self.delivery = delievery
        

    
    def update(self, sales: int, delivery: list, inventory: int, backlog: int, req_order: list, suppliers: np.array, customers: np.array):

        self.sales = sales
        self.delivery = delivery
        self.inventory = inventory
        self.backlog = backlog
        self.fulfilled_rate = sales / req_order
        self.suppliers = suppliers
        self.customers = customers

    def get_node_features(self):
        return np.array([self.prod_capacity, self.sale_price, self.sales, self.inventory, self.backlog, self.backlog_cost])
    
    def get_edge_features(self):
        return self.lead_time # order cost?
    
    def get_node_text_attributes(self):

        attributes = []
        # prod capacity
        attributes.append(f"Production capacity is {self.prod_capacity} units")
        # Prod cost
        attributes.append(f"Production cost is {self.prod_cost} units")
        # Holding cost
        attributes.append(f"Holding cost is {self.holding_cost} units")
        # Backlog cost
        attributes.append(f"Backlog cost is {self.backlog_cost} units")
        # Backlog
        attributes.append(f"Current backlog is {self.backlog} units")
        # Inventory
        attributes.append(f"The current inventory is {self.inventory} units")
        # price
        attributes.append(f"Price is {self.sale_price} units")
        # Fulfilled_rate
        attributes.append(f"The order fulfilled rate in the last round is {self.fulfilled_rate}")
        
        return attributes
    

    def get_edge_text_attributes(self, agent_profiles: dict):

        attributes = defaultdict(list)
        # supply_relations
        for agent_name in self.suppliers: # TO-CHECK: the format of suppliers
            attributes[(self.name, agent_profiles[agent_name])].append(f"supplier")
        # demand_relations
        for agent_name in self.customers:
            attributes[(self.name, agent_profiles[agent_name])].append(f"customer")
        # lead time
        for agent_name in self.upstream_agents:
            attributes[(self.name, agent_profiles[agent_name])].append(f"lead time is {self.lead_times[agent]} units")
        # Order cost
        for agent_name in self.upstream_agents:
            attributes[(self.name, agent_profiles[agent_name])].append(f"order cost is {self.order_cost[agent]} units")
        # delievery
        for agent_name in self.delivery:
            attributes[(self.name, agent_profiles[agent_name])].append(f"Has {self.delivery} units to be delivered in {} days")
        


def create_agent_profiles(env_config: dict):

    agent_profiles = []
    num_stages = env_config['num_stages']
    num_agents_per_stage = env_config['num_agents_per_stage']
    for i in range(num_stages):
        for j in range(num_agents_per_stage):
            agent_profiles.append(agent_profile(stage_idx=i, agent_idx=j, role=env_config["stage_names"][i], prod_capacity=env_config['prod_capacities'][i*num_agents_per_stage+j],  
                                                sale_price=env_config['sale_prices'][i*num_agents_per_stage+j], init_inventory=env_config['init_inventories'][i*num_agents_per_stage+j], 
                                                backlog_cost=env_config['backlog_costs'][i*num_agents_per_stage+j], holding_cost=env_config['holding_costs'][i*num_agents_per_stage+j], 
                                                order_cost=env_config['order_costs'][i*num_agents_per_stage:(i+1)*num_agents_per_stage], prod_cost=env_config['prod_costs'][i*num_agents_per_stage:(i+1)*num_agents_per_stage], 
                                                lead_times=env_config['lead_times'][i][j], suppliers=env_config['supply_relations'][i][j], customers=env_config['demand_relations'][i][j]))
    
    return agent_profiles




 
# Old version for GML
# class SupplyChain_Graph():

#     def __init__(self, agent_profiles: list, num_stages: int, num_agents_per_stage: int):
        
#         self.init_G = self._build_nx_graph(agent_profiles=agent_profiles)
#         self.num_stages = num_stages
#         self.num_agents_per_stage = num_agents_per_stage
#         self.reset_G()

#     def reset_G(self):
#         self.G = copy.deepcopy(self.init_G)

#     def _build_nx_graph(self, agent_profiles):

#         G = nx.DiGraph()

#         for ag in agent_profiles:
#             G.add_node(ag.name, prod_capacity=ag.prod_capacity, sale_price=ag.sale_price, sales=ag.sales, 
#                 inventory=ag.inventory, backlog=ag.backlog, backlog_cost=ag.backlog_cost, stage=ag.stage)
            
#         for ag in agent_profiles:
#             if ag.role == "manufacturer":
#                 continue
#             lead_time = ag.lead_time
#             order_cost = ag.order_cost
#             stage_idx = ag.stage
#             for sup_idx in range(len(lead_time)):
#                 G.add_edge(u_of_edge=ag.name, v_of_edge=f"stage_{stage_idx+1}_agent_{sup_idx}", lead_time=lead_time[sup_idx], order_cost=order_cost[sup_idx])
            
#             for sup_idx, label in enumerate(ag.suppliers):
#                 if label == 1:
#                     G.add_edge(u_of_edge=f"stage_{stage_idx+1}_agent_{sup_idx}", v_of_edge=ag.name, supplier=True)
#                     G.add_edge(u_of_edge=ag.name, v_of_edge=f"stage_{stage_idx+1}_agent_{sup_idx}", customer=True)

#         return G
    
#     def update_graph(self, state_dict: dict, past_req_orders: dict):
#         # update dynamic info, such as sales, delievery, past_req_order, backlog, upstream_backlog
#         for i in range(self.num_stages):
#             for j in range(self.num_agents_per_stage):
#                 agent_state = state_dict[f"stage_{i}_agent_{j}"]
#                 sales = agent_state["sales"]
#                 backlog = agent_state["backlog"]
#                 upstream_backlog = agent_state["upstream_backlog"]
#                 deliveries = agent_state["deliveries"]
#                 inventory = agent_state["inventory"]
#                 pr_orders = past_req_orders.get(f"stage_{i}_agent_{j}", [])
#                 suppliers = agent_state["suppliers"]

#                 self.G.nodes[f"stage_{i}_agent_{j}"]["inventory"] = inventory
#                 self.G.nodes[f"stage_{i}_agent_{j}"]['backlog'] = backlog
#                 self.G.nodes[f"stage_{i}_agent_{j}"]['upstream_backlog'] = upstream_backlog
#                 self.G.nodes[f"stage_{i}_agent_{j}"]['sales'] = sales[-1] 
 
#                 if i < self.num_stages - 1:
#                     for k in range(self.num_agents_per_stage): # to upstream suppliers except the manufacturers
#                         # add new directional edge for indicating delivery
#                         if sum(deliveries[k]):
#                             for day in range(len(deliveries[k])):
#                                 if deliveries[k][-day] > 0:        
#                                     self.G.add_edge(f"stage_{i+1}_agent_{k}", f"stage_{i}_agent_{j}")
#                                     self.G[f"stage_{i+1}_agent_{k}"][f"stage_{i}_agent_{j}"][f'deliveries in {day} days'] = deliveries[k][-day]

#                     if len(pr_orders) > 0:
#                         for k in range(self.num_agents_per_stage):      
#                             if pr_orders[k] > 0:
#                                 self.G[f"stage_{i}_agent_{j}"][f"stage_{i+1}_agent_{k}"]['past_req_orders'] = pr_orders[k]
                    
#                     for k, label in enumerate(suppliers):
#                         if label == 1:
#                             self.G.add_edge(f"stage_{i+1}_agent_{k}", f"stage_{i}_agent_{j}", supplier=True)
#                             self.G.add_edge(f"stage_{i}_agent_{j}", f"stage_{i+1}_agent_{k}", customer=True)
#                         else: # label == 0
#                             self.G.add_edge(f"stage_{i+1}_agent_{k}", f"stage_{i}_agent_{j}", supplier=False)
#                             self.G.add_edge(f"stage_{i}_agent_{j}", f"stage_{i+1}_agent_{k}", customer=False)

