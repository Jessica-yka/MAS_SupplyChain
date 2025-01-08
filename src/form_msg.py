from utils import get_state_description, get_demand_description  

task1_msg = (
    "Task1: Do you want to remove any upstream suppliers?\n"
    "Please state your reason in 1-2 sentences first "
    "and then provide your action as a list following this format (e.g., [0, 1] for removing agent0 and agent1 as suppliers, [] for doing nothing)\n") 
task2_msg = (
    "Task2: Do you want to add any upstream suppliers?\n"
    "Please state your reason in 1-2 sentences first "
    "and then provide your action as a list following this format (e.g., [2, 3] for adding agent2 and agent3 as suppliers, [] for doing nothing)\n"
)
task3_msg = (
    "Task3: What is the order quantity you would like to place with each supplier for this round?\n"
    "Please state your reason in 1-2 sentences first "
    "and then provide your action as a list following this format. E.g.,[(\"agent0\": 4), (\"agent1\": 2)].\n"
)
gold_rule_msg = (
    "Golden rule of this game: Open orders should always equal to \"expected downstream orders + backlog\". "
    "If open orders are larger than this, the inventory will rise (once the open orders arrive). "
    "If open orders are smaller than this, the backlog will not go down and it may even rise. "
    "Please consider the lead time and place your order in advance. "
    "Remember that your upstream has its own lead time, so do not wait until your inventory runs out. "
    "Also, avoid ordering too many units at once. "
    "Try to spread your orders over multiple rounds to prevent the bullwhip effect. "
    "Anticipate future demand changes and adjust your orders accordingly to maintain a stable inventory level.\n\n"
)
    
def generate_msg(im_env, action_order_dict: dict, stage_state: dict, period: int, stage: int, agent: int):

    if stage != 0:
        downstream_order = f"Your downstream order from the stage {stage} for this round is {action_order_dict[f'stage_{stage - 1}_agent_{agent}']}. "
    else:
        downstream_order = ""

    demand_description = get_demand_description(im_env.demand_dist)
    message = (
        f"Now this is the round {period + 1}, "
        f"and you are at the stage {stage + 1}: {im_env.stage_names[stage]} in the supply chain. "
        f"Given your current state:\n{get_state_description(stage_state)}\n\n"
    )
    state_info = message
    if stage == 0:
        message += f"{demand_description}\n"
    else:
        message += f"{downstream_order}\n"
    
    if stage == im_env.num_stages - 1: # do not ask for upstream orders for the manufacturer
        num_tasks = 1
        message += f"There are {num_tasks} tasks for you to make decision\n\n"
        message += f"{task3_msg}\n"
    else:
        num_tasks = 3
        message += f"There are {num_tasks} tasks for you to make decision\n\n"
        message += f"{task1_msg}\n"
        message += f"{task2_msg}\n"
        message += f"{task3_msg}\n"

    message += f"{gold_rule_msg}\n"
    
    return message, state_info

