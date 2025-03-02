import sys
from src.model.utils.utils import get_state_description, get_demand_description
from src.model.env import InventoryManagementEnv

task1_msg = (
    "Task1: Do you want to remove anyone from your upstream supplier list?\n"
    "Please consider the lead time and order cost when making decision. State your reason in 1-2 sentences first "
    "and then provide your action as a list following this format (e.g., [0, 1] for removing agent0 and agent1 as suppliers, [] for doing nothing)\n") 
task2_msg = (
    "Task2: Do you want to add anyone as your new supplier(s) given other available upstream suppliers in the environment?\n"
    "Please state your reason in 1-2 sentences first "
    "and then provide your action as a list following this format (e.g., [2, 3] for adding agent2 and agent3 as suppliers, [] for doing nothing)\n"
)
# Please estimate the future downstream demand based on the recent sales, and consider the lead time and order cost when making decision. 
# "Please state your reason in 1-2 sentences first "
task3_msg = (
    "Task3: What is the order quantity you would like to place with each supplier for this round? You can only place orders to your upstream suppliers\n"
    "Please state your reason in 1-2 sentences first "
    "and then provide your action as a list following this format. E.g.,[(\"agent0\": 4), (\"agent1\": 2)].\n"
)
task4_msg = (
    "Task4: What is the price you would like to set for the products?"
    "Please state your reason in 1-2 sentences first "
    "and then provide your action as a list following this format (e.g., [8])"
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
least_lead_time = (
    "Task: Which upstream compan(ies) has the least lead time to you? "
    "Provide the answer in brackets (e.g., [agent4])."
)
lowest_order_cost = (
    "Task: Which upstream compan(ies) has the lowest order cost? "
    "Provide the answer in brackets (e.g., [agent5])."
)
expected_demand = (
    "Task: What is your estimated demand from downstream in the next round? Provide the answer in brackets (e.g., [10]). \n"
)
    
def generate_msg(im_env: InventoryManagementEnv, shutdown_list: list, recovery_list: list, enable_graph_change: bool, enable_price_change: bool, action_order_dict: dict, 
                 past_req_orders: list, stage_state: dict, period: int, stage_id: int, cur_agent_id: int):
    
    agent_name = f"stage_{stage_id}_agent_{cur_agent_id}"
    message = (
        f"Now this is the round {period}, "
        f"and you are {agent_name} at the stage {stage_id}: {im_env.stage_names[stage_id]} in the supply chain. "
        )
    emergent_events = im_env.emergent_events.get(period, {'events': [], 'affected_agents': []})
    for eid, event in enumerate(emergent_events['events']):
        affected_agents = [f"stage_{sid}_agent_{aid}" for (sid, aid) in emergent_events['affected_agents'][eid]]
        affected_agents = ", ".join(affected_agents)
        message += f"An emergent event {event} has happened to {affected_agents}. "
        # if event == "sudden_shutdown":
        #     shutdown_agents = []
        #     for (st_stage_id, st_agent_id) in shutdown_list:
        #         shutdown_agents.append(f"stage_{st_stage_id}_agent_{st_agent_id}")
        #     shutdown_agents = ", ".join(shutdown_agents)
        #     shutdown_message = (
        #         f"There is a sudden shutdown event. "
        #         f"Those agent(s) are closed since now: {shutdown_agents}\n\n"
        #     )
        #     message += shutdown_message
        # if event == "recovery":
        #     recovered_agents = []
        #     for (rec_stage_id, rec_agent_id) in recovery_list:
        #         recovered_agents.append(f"stage_{rec_stage_id}_agent_{rec_agent_id}")
        #     recovered_agents = ", ".join(recovered_agents)
        #     recover_message = (
        #         f"There is a recovery event. "
        #         f"Those agent(s) are re-open since now: {recovered_agents}\n\n"
        #     )
        #     message += recover_message
    message += f"Given your current state:\n{get_state_description(state=stage_state, past_req_orders=past_req_orders, G=None, agent_name=agent_name, state_format=im_env.state_format, enable_graph_change=enable_graph_change)}\n\n"
    if stage_id != 0:
        down_order = []
        for down_agent_id in range(im_env.num_agents_per_stage):
            dr = action_order_dict[f'stage_{stage_id - 1}_agent_{down_agent_id}'][cur_agent_id]
            if dr != 0:
                down_order.append(f"from agent{down_agent_id}: {dr}")
        down_order = "; ".join(down_order)
        downstream_order = f"Your downstream order from the agents at stage {stage_id-1} for this round is: {down_order}. "
        message += f"{downstream_order}\n"
    else: # retailer stage
        demand_description = get_demand_description(im_env.demand_fn)
        message += f"{demand_description}\n"

    state_info = message

    # message += get_lead_time_task()
    # message += get_order_cost_task()
    message += get_decision_task(stage=stage_id, im_env=im_env, enable_graph_change=enable_graph_change, enable_price_change=enable_price_change)
    # message += get_expected_demand_task()

    return message, state_info


def get_lead_time_task():

    task_msg = "\nPlease answer the question based on your understanding of the given supply chain network.\n"
    task_msg += least_lead_time

    return task_msg


def get_order_cost_task():

    task_msg = "\nPlease answer the question based on your understanding of the given supply chain network.\n"
    task_msg += lowest_order_cost

    return task_msg

def get_expected_demand_task():

    task_msg = "\nTask: What is your estimated demand from downstream in the next round? Provide the answer in brackets (e.g., [10]).\n"

    return task_msg


def get_decision_task(stage: int, im_env, enable_graph_change: bool, enable_price_change: bool):

    task_msg = ""
    num_tasks = 0        

    if stage < im_env.num_stages - 1 and enable_graph_change: # Ask for supplier updates if it is allowed or it is not a manufacturer
        task_msg += f"{task1_msg}\n"
        task_msg += f"{task2_msg}\n"
        num_tasks += 2

    # Ask for order quantity
    # task_msg += f"{expected_demand}\n"
    task_msg += f"{task3_msg}\n"
    num_tasks += 1

    # Ask for price decision
    if enable_price_change:
        task_msg += f"{task4_msg}\n"
        num_tasks += 1

    task_msg = f"There are {num_tasks} task(s) for you to make decision(s). \n\n" + task_msg

    task_msg += f"{gold_rule_msg}\n"

    return task_msg


