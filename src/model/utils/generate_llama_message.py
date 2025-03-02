
from env import InventoryManagementEnv
import networkx as nx
import pandas as pd


upstream_reliability_msg = lambda stage_id: (
    f"Task: Which upstream compan(ies) at stage{stage_id} has high reliability, in terms of production capacity, delivery time, order fullfillment, and price? "
    "Provide the answer in brackets (e.g., [stage_x_agent_y])."
)
demand_msg = (
    "Task: What is your estimated demand from downstream in the next round? Provide the answer in brackets (e.g., [10]). \n"
)
event_upstream_price_msg = lambda event, stage: (
    f"Task: Based on the provided supply chain graph, how would the {event} affect any of your upstream suppliers reliability in terms of price? Answer either positive or negative if it happens to your supplier(s), otherwise answer 'neutral'\n"
    "Please state your reason in 1-2 sentences."
)
event_upstream_production_msg = lambda event, stage: (
    f"Task: Based on the provided supply chain graph, how would the {event} affect any of your upstream suppliers reliability in terms of production capacity? Answer either positive or negative if it happens to your supplier(s), otherwise answer 'neutral'\n"
    "Please state your reason in 1-2 sentences."
)
event_upstream_lead_time_msg = lambda event, stage: (
    f"Task: Based on the provided supply chain graph, how would the {event} affect any of your upstream suppliers reliability in terms of delivery time? Answer either positive or negative if it happens to your supplier(s), otherwise answer 'neutral'\n"
    "Please state your reason in 1-2 sentences."
)
event_downstream_msg = lambda event, stage: (
    f"Task: How would the {event} affect any of your downstream customers in terms of demand? Answer either positive or negative if it happens to your customer(s), otherwise answer 'neutral'\n"
    "Please state your reason in 1-2 sentences."
)
task1_msg = (
    "Task1: What is the order quantity you would like to place with each supplier for this round? You can only place orders to your upstream suppliers\n"
    "Please consider the downstream demand and upstream suppliers' reliability when making decision. State your reason in 1-2 sentences first "
    "and then provide your action as a list following this format. E.g.,[(\"agent0\": 4), (\"agent1\": 2)].\n\n"
)
task2_msg = (
    "Task2: Do you want to remove anyone from your upstream supplier list?\n"
    "Please consider the supplier's reliability when making decision. State your reason in 1-2 sentences first. "
    "provide your action as a list following this format (e.g., [0, 1] for removing agent0 and agent1 as suppliers, [] for doing nothing)\n\n"
)
task3_msg = (
    "Task3: Do you want to add anyone as your new supplier(s) given other available upstream suppliers in the environment?\n"
    "Please consider the company's reliability when making decision. State your reason in 1-2 sentences first. "
    "provide your action as a list following this format (e.g., [2, 3] for adding agent2 and agent3 as suppliers, [] for doing nothing)\n\n"
)
task4_msg = (
    "Task4: What is the price you would like to set for the products in the next round?",
    "Please consider the recent sales and the pricing of competitors when setting price. "
    "Please state your reason in 1-2 sentences first "
    "and then provide your action as a list following this format (e.g., [8])\n\n"
)


gold_rule_msg = (
    "\n\n"
    "Please follow the output format strictly. \n"
    "Golden rule of this game: Open orders should always equal to \"expected downstream orders + backlog\". "
    "If open orders are larger than this, the inventory will rise (once the open orders arrive). "
    "If open orders are smaller than this, the backlog will not go down and it may even rise. "
    "The price should cover both the production cost and order cost. "
    "If price is larger than the sum of two costs, there is a profit. "
    "Otherwise there is a loss. "
    "You can only place order to your upstream suppliers. "
    "Please consider the lead time and place your order in advance. "
    "Please consider the lead time and order costs when selecting your suppliers. "
    "Please consider the recent sales when deciding the order quantity. "
    "Please consider the order cost and the pricing of competitors when setting price. "
    "Remember that your upstream has its own lead time, so do not wait until your inventory runs out. "
    "Also, avoid ordering too many units at once. "
    "Try to spread your orders over multiple rounds to prevent the bullwhip effect. "
    "Anticipate future demand changes and adjust your orders accordingly to maintain a stable inventory level.\n\n"
)

def get_df_nodes(emergent_events: dict, state: dict, stage_id: int, agent_id: int, num_agents_per_stage: int, num_stages: int):
    # 'order_cost', 'production_cost', 'production_capacity', 'inventory', 'backlog', 'upstream_backlog', 'sales'
    df_nodes = pd.DataFrame(columns=['id', "node_attr"])
    agent_state = state[f"stage_{stage_id}_agent_{agent_id}"]
    n_rows = 0
    node_attr = f"name: stage_{stage_id}_agent_{agent_id}, price: {agent_state['sale_price']}, production cost: {agent_state['prod_cost']}, production capacity: {agent_state['prod_capacity']}, inventory: {agent_state['inventory']}, backlog: {agent_state['backlog']}, upstream backlog: {agent_state['upstream_backlog']}, sales: {agent_state['sales']}"
    df_nodes.loc[n_rows, ['id', 'node_attr']] = [stage_id*num_agents_per_stage+agent_id, node_attr]
    n_rows += 1

    for comp_agent_id in range(num_agents_per_stage):
        node_attr = f"name: stage_{stage_id}_agent_{comp_agent_id}"
        df_nodes.loc[n_rows, ['id', 'node_attr']] = [stage_id*num_agents_per_stage+comp_agent_id, node_attr]
        n_rows += 1
    for supp_idx in range(num_agents_per_stage):
        node_attr = f"name: stage_{stage_id+1}_agent_{supp_idx}, price: {agent_state['order_costs'][supp_idx]}, production capacity: {state[f'stage_{stage_id+1}_agent_{supp_idx}']['prod_capacity']}"
        df_nodes.loc[n_rows, ['id', 'node_attr']] = [(stage_id+1)*num_agents_per_stage+supp_idx, node_attr]
        n_rows += 1
    for down_stage_id in range(0, stage_id):
        for down_agent_id in range(num_agents_per_stage):
            node_attr = f"name: stage_{down_stage_id}_agent_{down_agent_id}"
            df_nodes.loc[n_rows, ['id', 'node_attr']] = [down_stage_id*num_agents_per_stage+down_agent_id, node_attr]
            n_rows += 1
    for upp_stage_id in range(stage_id+2, num_stages): 
        for upp_agent_id in range(num_agents_per_stage):
            node_attr = f"name: stage_{upp_stage_id}_agent_{upp_agent_id}"
            df_nodes.loc[n_rows, ['id', 'node_attr']] = [upp_stage_id*num_agents_per_stage+upp_agent_id, node_attr]
            n_rows += 1
    for eid, event in enumerate(emergent_events['events']):
        node_attr = f"{event}"
        df_nodes.loc[n_rows, ['id', 'node_attr']] = [num_stages*num_agents_per_stage+eid, node_attr]
        n_rows += 1
    return df_nodes



def get_df_edges(emergent_events: dict, state: dict, past_req_orders: list, num_agents_per_stage: int, num_stages: int, stage_id: int, agent_id: int):
    # "lead time": [],
    # "requested_order": [],
    # "deliverying": [],
    # "is the supplier of": [],
    target_agent = f"stage_{stage_id}_agent_{agent_id}"
    agent_state = state[target_agent]
    df_edges = pd.DataFrame(columns=['src', 'dst', 'edge_attr'])

    row_idx = 0
    # lead time
    for src in range(num_agents_per_stage):
        lt = agent_state["lead_times"][src]
        df_edges.loc[row_idx, ['src', 'dst', 'edge_attr']] = [(stage_id+1)*num_agents_per_stage+src, stage_id*num_agents_per_stage+agent_id, f"has delivery time of {lt} day(s)"]
        row_idx += 1
    # requested order
    for i, _ in enumerate(past_req_orders):
        if past_req_orders[i] != 0:
            df_edges.loc[row_idx, ['src', 'dst', 'edge_attr']] = [(stage_id-1)*num_agents_per_stage+i, stage_id*num_agents_per_stage+agent_id, f"requested {past_req_orders[i]} unit(s)"]
            row_idx += 1
    # deliverying
    for src in range(num_agents_per_stage):
        if agent_state['suppliers'][src] == 1:
            deliverying = agent_state['deliveries'][src][-agent_state['lead_times'][src]:]
            if len(deliverying) > 0:
                for day in range(len(deliverying)):
                    if deliverying[day] != 0:
                        df_edges.loc[row_idx, ['src', 'dst', 'edge_attr']] = [(stage_id+1)*num_agents_per_stage+src, stage_id*num_agents_per_stage+agent_id, f"delivering {deliverying[day]} unit(s) of products in {day} day(s)"]
                        row_idx += 1
         
    # is the supplier of
    for m in range(num_stages-1):
        for x in range(num_agents_per_stage):
            for dst in range(num_agents_per_stage):
                if state[f"stage_{m}_agent_{x}"]['suppliers'][dst] == 1:
                    df_edges.loc[row_idx, ['src', 'dst', 'edge_attr']] = [(m+1)*num_agents_per_stage+dst, m*num_agents_per_stage+x, f"is the supplier of"]
                    row_idx += 1
    # emergent events
    for eid, event in enumerate(emergent_events['events']):
        affected_agents = emergent_events['affected_agents'][eid]
        for (aff_stage_id, aff_agent_id) in affected_agents:
            df_edges.loc[row_idx, ['src', 'dst', 'edge_attr']] = [num_stages*num_agents_per_stage+eid, aff_stage_id*num_agents_per_stage+aff_agent_id, "affects"]

    return df_edges



def generate_graph_description(emergent_events: dict, state: dict, past_req_orders: list, stage_id: int, agent_id: int, num_stages: int, num_agents_per_stage: int):
                          
    df_nodes = get_df_nodes(emergent_events=emergent_events, state=state, stage_id=stage_id, agent_id=agent_id, num_stages=num_stages, num_agents_per_stage=num_agents_per_stage)
    df_edges = get_df_edges(emergent_events=emergent_events, state=state, stage_id=stage_id, agent_id=agent_id, num_stages=num_stages, num_agents_per_stage=num_agents_per_stage, past_req_orders=past_req_orders)
    

    return df_nodes, df_edges

    # return (
    #     f" - Lead Time: {lead_times} round(s)\n"
    #     f" - Order costs: {order_costs} unit(s)\n"
    #     f" - Production costs: {prod_cost} unit(s)\n"
    #     f" - Inventory Level: {state['inventory']} unit(s)\n"
    #     f" - Production capacity: {state['prod_capacity']} unit(s)\n"
    #     f" - Current Backlog (you owing to the downstream): {state['backlog']} unit(s)\n"
    #     f" - Upstream Backlog (your upstream owing to you): {state['upstream_backlog']} unit(s)\n"
    #     f" - Previous Sales (in the recent round(s), from old to new): {state['sales']}\n"
    #     f" - In the last round, you placed orders to upstream suppliers: {req_orders}\n"
    #     f" - Arriving Deliveries (in this and the next round(s), from near to far): {arriving_delieveries}\n"
    #     f" - Your upstream suppliers are: {suppliers}\n" 
    #     f" - Other available upstream agents in the environment are: {non_suppliers}\n"
    # )




def generate_questions(emergent_events: dict, action_order_dict: dict, period: int, stage_id: int, agent_id: int, im_env, guided_cot: bool, enable_graph_change: bool, enable_price_change: bool):
    down_order = []
    for down_agent_id in range(im_env.num_agents_per_stage):
        dr = action_order_dict[f'stage_{stage_id - 1}_agent_{down_agent_id}'][agent_id]
        if dr != 0:
            down_order.append(f"from stage_{stage_id-1}_agent{down_agent_id}: {dr}")
    down_order = "; ".join(down_order)
    prompt = (
        f"Now this is round {period}. You are stage_{stage_id}_agent_{agent_id} in the supply chain. \n"
        f"Your downstream order from the agents at stage {stage_id-1} for this round is: {down_order}. \n\n"
    )

    thinking_pipeline = []

    # To let the model explicitly think about the demand and upstream reliability
    if guided_cot:
        for ev in emergent_events['events']:
            thinking_pipeline += [
                event_upstream_price_msg(ev, stage_id+1),
                event_upstream_production_msg(ev, stage_id+1),
                event_upstream_lead_time_msg(ev, stage_id+1),
            ]
        thinking_pipeline += [
            upstream_reliability_msg(stage_id+1),
            demand_msg,
        ]

    task_msg = ""
    num_tasks = 0        

    task_msg += task1_msg
    num_tasks += 1

    if stage_id < im_env.num_stages - 1 and enable_graph_change: # Ask for supplier updates if it is allowed or it is not a manufacturer
        task_msg += f"{task2_msg}\n"
        task_msg += f"{task3_msg}\n"
        num_tasks += 2

    # Ask for price decision
    if enable_price_change:
        task_msg += f"{task4_msg}\n"
        num_tasks += 1

    task_msg = f"There are {num_tasks} task(s) for you to make decision(s). \n\n" + task_msg

    task_msg += f"{gold_rule_msg}\n"

    return prompt, thinking_pipeline, task_msg
