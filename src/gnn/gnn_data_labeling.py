# %% [markdown]
# ## Label graph data with optimal solution
# 1. Contextualization: Convert numerical value to text-based attributes
# 
# 2. Optimization: Use MIP to get the optimal solution

# %%
import numpy as np
from tqdm import tqdm
import json
import sys
import os
from gurobipy import Model, GRB
sys.path.append('src')
from model.utils import save_data_to_json, read_data_from_json
# sys.path.append('gnn')
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# %%
env_config_name = "large_graph_test"
save_datapath = f"src/gnn/gnn_dataset"


# %% [markdown]
# ### Optimization
def check_data(data):
    if data["profit"] is None or data["profit"] < 0:
        print("Spot error")
        demands = data['demands']
        orders = data['orders']
        inventory = data['inventory']
        lead_times = data['lead_times']
        prod_capacity = data["prod_capacity"]
        sale_price = data['sale_price']
        order_costs = data['order_costs']
  
        print("demand in each round", [sum(demands[i]) for i in range(num_period)])
        print("optimal order in each round", [sum(orders[i]) for i in range(num_period-10)])
        print("optimal profit", data['profit'])
        print("sale price", sale_price)
        print("est sales", data['sales_t'])
        print("est inventory", data["inventory_t"])
        print("est arrived order", data['arrival_record'])
        print("init inventory", inventory)
        print("prod_capacity", prod_capacity)
        print("order cost", order_costs)
        print("lead times", lead_times)

        return False
    
    return True
# %%
def optimize_order(demands, num_period, num_agents_per_stage, inventory, lead_times, 
                   prod_capacity, sale_price, order_costs, prod_cost, holding_cost, backlog_cost):
    
    # Create a new model
    model = Model(name="orders")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 60)
    
    num_buffer = 10
    # Add decision variables
    order = [model.addVar(name=f"t{t+1}_order2agent{j}", lb=0) for t in range(num_period+num_buffer) for j in range(num_agents_per_stage)]
    order_cost_t = [model.addVar(name=f"t{t+1}_order_cost", lb=0) for t in range(num_period)]
    z_inv = [model.addVar(name="auxiliary for determing sales", vtype=GRB.BINARY) for t in range(num_period)]
    z_pc = [model.addVar(name="auxiliary for determing sales", vtype=GRB.BINARY) for t in range(num_period)]
    z_dem = [model.addVar(name="auxiliary for determing sales", vtype=GRB.BINARY) for t in range(num_period)]
    sales_t = [model.addVar(name=f"t{t+1}_sales", lb=0) for t in range(num_period)]
    inventory_t = [model.addVar(name=f"inventory_at_time{t}", lb=0) for t in range(num_period)]  # y >= 0
    arrived_order_t = [model.addVar(name=f"arrived orders at time{t}", lb=0) for t in range(num_period)]


    # Set the objective function: 
    model.setObjective(sum(sales_t[t]*sale_price - order_cost_t[t] - 
                        inventory_t[t]*holding_cost for t in range(num_period)), GRB.MAXIMIZE)

    # Add constraints
    # about sales
    M = 999
    for t in range(num_period):
        model.addConstr(sales_t[t] <= prod_capacity, "sales less than production capacity")
        model.addConstr(sales_t[t] <= inventory_t[t], "sales less than inventory")
        model.addConstr(sales_t[t] <= sum(demands[t][idx] for idx in range(num_agents_per_stage)))
        model.addConstr(sales_t[t] >= prod_capacity - M * (1 - z_pc[t]), "sales bigger than pc if z_pc")
        model.addConstr(sales_t[t] >= inventory_t[t] - M * (1 - z_inv[t]), "sales bigger than inv if z_inv")
        model.addConstr(sales_t[t] >= sum(demands[t][idx] for idx in range(num_agents_per_stage)) - M * (1 - z_dem[t]), "sales bigger than inv if z_dem")\

        model.addConstr(z_inv[t] + z_pc[t] + z_dem[t] == 1, "only_one_z_active")

    # about inventory 
    model.addConstr(inventory_t[0] == inventory)
    for t in range(1, num_period):
        model.addConstr(inventory_t[t] == inventory_t[t-1]-sales_t[t-1]+arrived_order_t[t], "inventory level at time t")   # the inventory equals to t

    # about buffer order:
    # all 0
    for t in range(num_buffer):
        for idx in range(num_agents_per_stage):
            model.addConstr(order[t*num_agents_per_stage+idx]==0)

    # about order cost
    for t in range(num_buffer, num_buffer+num_period):
        model.addConstr(order_cost_t[t-num_buffer] == sum(order[t*num_agents_per_stage+idx]*order_costs[idx] for idx in range(num_agents_per_stage))) 

    # about arrived_orders
    for t in range(num_buffer, num_buffer+num_period):
        model.addConstr(arrived_order_t[t-num_buffer] == sum(order[(t-lead_times[idx])*num_agents_per_stage+idx] for idx in range(num_agents_per_stage)))

    # Optimize the model
    model.optimize()

    # Print the results
    if model.status == GRB.OPTIMAL:
        # print(f"Optimal solution found:")
        # print(f"Optimal objective value (z) = {model.objVal}")

        label = []
        for t in range(num_buffer, num_period): #discard the tails
            t_order = []
            for idx in range(num_agents_per_stage):
                t_order.append(order[t*num_agents_per_stage+idx].x)
            label.append((t_order))
        
        sales_record = []
        for t in range(num_period):
            sales_record.append(sales_t[t].x)
        
        inventory_record = []
        for t in range(num_period):
            inventory_record.append(inventory_t[t].x)

        arrival_record = []
        for t in range(num_period):
            arrival_record.append(arrived_order_t[t].x)
        return label, model.ObjVal, sales_record, inventory_record, arrival_record
    else:
        print("No optimal solution found.")
        return None, None, None, None



if __name__ == "__main__":
    
    dataset = read_data_from_json(read_path=f"{save_datapath}/gnn_data_Env({env_config_name}).json")
    # %%
    num_data = len(dataset)
    num_period = len(dataset[0]['demands'])
    num_agents_per_stage = len(dataset[0]["lead_times"])

    for i in tqdm(range(num_data)):
        demands = dataset[i]['demands']
        inventory = dataset[i]['inventory']
        lead_times = dataset[i]['lead_times']
        prod_capacity = dataset[i]["prod_capacity"]
        sale_price = dataset[i]['sale_price']
        order_costs = dataset[i]['order_costs']
        prod_cost = dataset[i]['prod_cost']
        holding_cost = dataset[i]['holding_cost']
        backlog_cost = dataset[i]['backlog_cost']

        orders, profit, sales_record, inventory_record, arrival_record = optimize_order(demands=demands, num_period=num_period, num_agents_per_stage=num_agents_per_stage, 
                                inventory=inventory, lead_times=lead_times, prod_capacity=prod_capacity, sale_price=sale_price, 
                                order_costs=order_costs, prod_cost=prod_cost, holding_cost=holding_cost, backlog_cost=backlog_cost)
        dataset[i]['orders'] = orders
        dataset[i]['profit'] = profit
        dataset[i]['sales_t'] = sales_record
        dataset[i]['inventory_t'] = inventory_record
        dataset[i]['arrival_record'] = arrival_record
                    

    # %%
    for i in range(num_data):
        if not check_data(data=dataset[i]):
            raise ValueError
        
    save_data_to_json(data=dataset, save_path=f"{save_datapath}/gnn_data_Env({env_config_name}).json")
    # %%
    print("================stats==================")
    print("average profit", np.mean([dataset[i]['profit'] for i in range(num_data)]))
    print("std profit", np.std([dataset[i]['profit'] for i in range(num_data)]))
