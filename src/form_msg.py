from utils import get_state_description, get_demand_description  

task1_msg = (
    "Task1: Do you want to remove anyone from your upstream supplier list?\n"
    "Please consider the lead time and order cost when making decision. State your reason in 1-2 sentences first "
    "and then provide your action as a list following this format (e.g., [0, 1] for removing agent0 and agent1 as suppliers, [] for doing nothing)\n") 
task2_msg = (
    "Task2: Do you want to add anyone as your new supplier(s) given other available upstream suppliers in the environment?\n"
    "Please state your reason in 1-2 sentences first "
    "and then provide your action as a list following this format (e.g., [2, 3] for adding agent2 and agent3 as suppliers, [] for doing nothing)\n"
)
task3_msg = (
    "Task3: What is the order quantity you would like to place with each supplier for this round? You can only place orders to your upstream suppliers\n"
    "Please consider the lead time and order cost when making decision. State your reason in 1-2 sentences first "
    "and then provide your action as a list following this format. E.g.,[(\"agent0\": 4), (\"agent1\": 2)].\n"
)
gold_rule_msg = (
    "Please follow the output format strictly. \n"
    "Golden rule of this game: Open orders should always equal to \"expected downstream orders + backlog\". "
    "If open orders are larger than this, the inventory will rise (once the open orders arrive). "
    "If open orders are smaller than this, the backlog will not go down and it may even rise. "
    "You can only place order to your upstream suppliers. "
    "Please consider the lead time and place your order in advance. "
    "Please consider the lead time and order costs when selecting your suppliers. "
    "Please consider the recent sales when deciding the order quantity. "
    "Remember that your upstream has its own lead time, so do not wait until your inventory runs out. "
    "Also, avoid ordering too many units at once. "
    "Try to spread your orders over multiple rounds to prevent the bullwhip effect. "
    "Anticipate future demand changes and adjust your orders accordingly to maintain a stable inventory level.\n\n"
)
least_lead_time = (
    "Task: Which upstream company has the least lead time to you? "
    "Provide the answer in brackets (e.g., [agent4])."
)
lowest_order_cost = (
    "Task: Which upstream company has the lowest order cost? "
    "Provide the answer in brackets (e.g., [agent5])."
)
    
def generate_msg(im_env, enable_graph_change: bool, action_order_dict: dict, past_req_orders: list, stage_state: dict, period: int, stage: int, cur_agent_idx: int):

    if stage != 0:
        down_order = []
        for down_agent_idx in range(im_env.num_agents_per_stage):
            dr = action_order_dict[f'stage_{stage - 1}_agent_{down_agent_idx}'][cur_agent_idx]
            if dr != 0:
                down_order.append(f"from agent{down_agent_idx}: {dr}")
        down_order = "; ".join(down_order)
        downstream_order = f"Your downstream order from the agents at stage {stage-1} for this round is: {down_order}. "
    else:
        downstream_order = ""

    demand_description = get_demand_description(im_env.demand_dist)
    agent_name = f"stage_{stage}_agent_{cur_agent_idx}"
    message = (
        f"Now this is the round {period + 1}, "
        f"and you are stage_{stage}_agent_{cur_agent_idx} at the stage {stage}: {im_env.stage_names[stage]} in the supply chain. "
        f"Given your current state:\n{get_state_description(state=stage_state, past_req_orders=past_req_orders, G=im_env.sc_graph.G, agent_name=agent_name, state_format=im_env.state_format, enable_graph_change=enable_graph_change)}\n\n"
        )

    if stage == 0:
        message += f"{demand_description}\n"
    else:
        message += f"{downstream_order}\n"

    state_info = message

    message += get_lead_time_task()
    message += get_order_cost_task()
    # message += get_decision_task(stage=stage, im_env=im_env, enable_graph_change=enable_graph_change)
    
    return message, state_info


def get_lead_time_task():

    task_msg = "\nPlease answer the question based on your understanding of the given supply chain network.\n"
    task_msg += least_lead_time

    return task_msg


def get_order_cost_task():

    task_msg = "\nPlease answer the question based on your understanding of the given supply chain network.\n"
    task_msg += lowest_order_cost

    return task_msg


def get_decision_task(stage: int, im_env, enable_graph_change: bool):

    task_msg = ""
    if stage == im_env.num_stages - 1 or not enable_graph_change: # do not ask for upstream orders for the manufacturer
        num_tasks = 1
        task_msg += f"There are {num_tasks} tasks for you to make decision\n\n"
        task_msg += f"{task3_msg}\n"
    else:
        num_tasks = 3
        task_msg += f"There are {num_tasks} tasks for you to make decision\n\n"
        task_msg += f"{task1_msg}\n"
        task_msg += f"{task2_msg}\n"
        task_msg += f"{task3_msg}\n"

    task_msg += f"{gold_rule_msg}\n"

    return task_msg


