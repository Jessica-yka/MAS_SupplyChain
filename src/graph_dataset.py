import numpy as np
from tqdm import tqdm
from agent_profile import agent_profile

def create_agent_profiles(env_config: dict):
    agent_profiles = []
    for i in range(env_config['num_stages']):
        for j in range(env_config['num_agents_per_stage']):
            agent_profiles.append(agent_profile(stage_idx=i, agent_idx=j, prod_capacity=env_config['prod_capacity'][i][j], prod_cost=env_config['prod_cost'][i][j], 
                                                price=env_config['price'][i][j], sales=env_config['sales'][i][j], inventory=env_config['inventory'][i][j], 
                                                backlog=env_config['backlog'][i][j], backlog_cost=env_config['backlog_cost'][i][j], inventory_cost=env_config['inventory_cost'][i][j], 
                                                order_cost=env_config['order_cost'][i][j], lead_time=env_config['lead_time'][i][j], delivery=env_config['delivery'][i][j]))
            
class GraphData():

    def __init__(self, agent_profiles):
        
        self.agent_profiles = agent_profiles

    def create_node_feat_matrix(self):

        node_features = []
        for ap in self.agent_profiles:
            node_features.append(ap.get_node_features)
        return np.stack(node_features)

    def create_edge_feat_matrix(self):

        
        pass
