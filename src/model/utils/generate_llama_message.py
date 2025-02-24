
from model.utils.utils import get_state_description, get_demand_description
from env import InventoryManagementEnv

upstream_reliability_msg = (
    "Task: Which upstream compan(ies) has high reliability, in terms of production capacity, delivery time, order fullfillment, and price? "
    "Provide the answer in brackets (e.g., [agent5])."
)
demand_msg = (
    "Task: What is your estimated demand from downstream in the next round? Provide the answer in brackets (e.g., [10]). \n"
)
event_upstream_price_msg = lambda event: (
    f"Task: How would the {event} affect your upstream suppliers' price in this round? \n"
    "Please state your reason in 1-2 sentences."
)
event_upstream_production_msg = lambda event: (
    f"Task: How would the {event} affect your upstream suppliers' production capacity in this round? \n"
    "Please state your reason in 1-2 sentences."
)
event_upstream_lead_time_msg = lambda event: (
    f"Task: How would the {event} affect your upstream suppliers' delivery time in this round? \n"
    "Please state your reason in 1-2 sentences."
)
event_downstream_msg = lambda event: (
    f"Task: How would the {event} affect your downstream customers' demand in this round? \n"
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


def get_questions(stage: int, im_env, guided_cot: bool, enable_graph_change: bool, enable_price_change: bool):

    events = im_env.events.get(im_env.period, [])
    thinking_pipeline = []

    # To let the model explicitly think about the demand and upstream reliability
    if guided_cot:
        if len(events) > 0:
            for ev in events:
                thinking_pipeline += [
                    event_upstream_price_msg(ev),
                    event_upstream_production_msg(ev),
                    event_upstream_lead_time_msg(ev),
                ]
        thinking_pipeline += [
            upstream_reliability_msg,
            demand_msg,
        ]

    task_msg = ""
    num_tasks = 0        

    task_msg += task1_msg
    num_tasks += 1

    if stage < im_env.num_stages - 1 and enable_graph_change: # Ask for supplier updates if it is allowed or it is not a manufacturer
        task_msg += f"{task2_msg}\n"
        task_msg += f"{task3_msg}\n"
        num_tasks += 2

    # Ask for price decision
    if enable_price_change:
        task_msg += f"{task4_msg}\n"
        num_tasks += 1

    task_msg = f"There are {num_tasks} task(s) for you to make decision(s). \n\n" + task_msg

    task_msg += f"{gold_rule_msg}\n"

    return thinking_pipeline, task_msg
