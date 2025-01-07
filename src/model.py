import os
import re
import sys
import time
import numpy as np
from typing import List
from tqdm.notebook import tqdm
from autogen import ConversableAgent
from llm_config import llm_config_list
from utils import extract_pairs



def get_state_description(state):
    suppliers = " ".join([f"agent{i}" for i, _ in enumerate(state['suppliers']) if state['suppliers'][i]==1])
    non_suppliers = " ".join([f"agent{i}" for i, _ in enumerate(state['suppliers']) if state['suppliers'][i]==0])
    lead_times = " ".join([f"from agent{i}: {state['lead_times'][i]}" for i, _ in enumerate(state['lead_times'])])
    arriving_delieveries = []
    for i, _ in enumerate(state['suppliers']):
        if state['suppliers'][i] == 1:
            print('state deliveries', state['deliveries'])
            arriving_delieveries.append(f"from agent{i}: {state['deliveries'][i][-state['lead_times'][i]:]}")
    arriving_delieveries = " ".join(arriving_delieveries)
    return (
        f" - Lead Time: {lead_times} round(s)\n"
        f" - Inventory Level: {state['inventory']} unit(s)\n"
        f" - Current Backlog (you owing to the downstream): {state['backlog']} unit(s)\n"
        f" - Upstream Backlog (your upstream owing to you): {state['upstream_backlog']} unit(s)\n"
        f" - Previous Sales (in the recent round(s), from old to new): {state['sales']}\n"
        f" - Arriving Deliveries (in this and the next round(s), from near to far): {arriving_delieveries}\n"
        f" - Your upstream suppliers are: {suppliers}\n" 
        f" - Other upstream suppliers are: {non_suppliers}\n"
    )


def get_demand_description(demand_fn: str) -> str:
    if demand_fn == "constant_demand":
        return "The expected demand at the retailer (stage 1) is a constant 4 units for all 12 rounds."
    elif demand_fn == "uniform_demand":
        return "The expected demand at the retailer (stage 1) is a discrete uniform distribution U{0, 4} for all 12 rounds."
    elif demand_fn == "larger_demand":
        return "The expected demand at the retailer (stage 1) is a discrete uniform distribution U{0, 8} for all 12 rounds."
    elif demand_fn == "seasonal_demand":
        return "The expected demand at the retailer (stage 1) is a discrete uniform distribution U{0, 4} for the first 4 rounds, " \
            "and a discrete uniform distribution U{5, 8} for the last 8 rounds."
    elif demand_fn == "normal_demand":
        return "The expected demand at the retailer (stage 1) is a normal distribution N(4, 2^2), " \
            "truncated at 0, for all 12 rounds."
    else:
        raise KeyError(f"Error: {demand_fn} not implemented.")


def create_agents(stage_names: List[str], num_agents_per_stage: int, llm_config) -> List[ConversableAgent]:
    agents = []
    num_stages = len(stage_names)
    
    for stage, stage_name in enumerate(stage_names):
        for sa_ind in range(num_agents_per_stage):
            agent = ConversableAgent(
                name=f"{stage_name.capitalize()}Agent_{sa_ind}",
                system_message=f"You play a crucial role in a {num_stages}-stage supply chain as the stage {stage + 1} ({stage_name}). "
                    "Your goal is to minimize the total cost by managing inventory and orders effectively.",
                llm_config=llm_config,
                code_execution_config=False,
                human_input_mode="NEVER",
            )
            agents.append(agent)
        
    return agents


def run_simulation(im_env, user_proxy, stage_agents):
   
    demand_description = get_demand_description(im_env.demand_dist) 
    all_state_dicts = {}
    all_action_order_dicts = {}
    all_action_sup_dicts = {}
    all_action_dem_dicts = {}
    all_reward_dicts = {}
    episode_reward = 0
    api_cost = 0
    im_env.reset()
    num_stages = im_env.num_stages
    num_agents_per_stage = im_env.num_agents_per_stage
    
    for period in range(im_env.num_periods):
        state_dict = im_env.parse_state(im_env.state_dict)
        all_state_dicts[period] = state_dict
        action_order_dict = {}
        action_sup_dict = {}
        action_dem_dict = {}
        
        for stage in range(num_stages):
            for agent in range(num_agents_per_stage):
                stage_state = state_dict[f'stage_{stage}_agent_{agent}']

                if stage != 0:
                    downstream_order = f"Your downstream order from the stage {stage} for this round is {action_order_dict[f'stage_{stage - 1}_agent_{agent}']}. "
                else:
                    downstream_order = ""

                message = (
                    f"Now this is the round {period + 1}, "
                    f"and you are at the stage {stage + 1}: {im_env.stage_names[stage]} in the supply chain. "
                    f"Given your current state:\n{get_state_description(stage_state)}\n\n"
                    f"{demand_description} {downstream_order}"
                    "There are three tasks for you to make decision\n"
                    "Task1: Do you want to remove any upstream suppliers?\n\n"
                    "Please state your reason in 1-2 sentences first "
                    "and then provide your action as a list (e.g. [0, 1] for removing agent0 and agent1 as suppliers, [] for doing nothing)\n"
                    "Task2: Do you want to add any upstream suppliers?\n\n"
                    "Please state your reason in 1-2 sentences first "
                    "and then provide your action as a list (e.g. [2, 3] for adding agent2 and agent3 as suppliers, [] for doing nothing)\n"
                    "Task3: What is the order quantity you would like to place with each supplier for this round?\n\n"
                    "Golden rule of this game: Open orders should always equal to \"expected downstream orders + backlog\". "
                    "If open orders are larger than this, the inventory will rise (once the open orders arrive). "
                    "If open orders are smaller than this, the backlog will not go down and it may even rise. "
                    "Please consider the lead time and place your order in advance. "
                    "Remember that your upstream has its own lead time, so do not wait until your inventory runs out. "
                    "Also, avoid ordering too many units at once. "
                    "Try to spread your orders over multiple rounds to prevent the bullwhip effect. "
                    "Anticipate future demand changes and adjust your orders accordingly to maintain a stable inventory level.\n\n"
                    "Please state your reason in 1-2 sentences first "
                    "and then provide your order as supplier-order pair in a list (e.g. [(\"agent0\": 4), (\"agent1\": 2)])."
                )

                chat_result = user_proxy.initiate_chat(
                    stage_agents[stage*num_agents_per_stage+agent],
                    message={'content': ''.join(message)},
                    summary_method="last_msg",
                    max_turns=1,
                    clear_history=False,
                )
                chat_summary = chat_result.summary
                api_cost += chat_result.cost['usage_including_cached_inference']['total_cost']
                # print(chat_summary)
                match = re.findall(r'\[(.*?)\]', chat_summary)
                
                sup_action = state_dict[f'stage_{stage}_agent_{agent}']['suppliers']
                remove_sup = match[0]                
                if remove_sup != "":
                    remove_sup = [int(ind) for ind in remove_sup.split(", ")]
                    for ind in remove_sup:
                        sup_action[ind] = 0
                add_sup = match[1]
                if add_sup != "":
                    add_sup = [int(ind) for ind in add_sup.split(", ")]
                    for ind in add_sup:
                        sup_action[ind] = 1
                action_sup_dict[f'stage_{stage}_agent_{agent}'] = sup_action
                
                # if match:
                #     stage_action = int(match.group(1))
                # else:
                #     stage_action = 0
                stage_order_action = np.zeros(num_agents_per_stage, dtype=int)
                if match[2]:
                    supplier_order_dict = extract_pairs(match[2])
                    for i in range(num_agents_per_stage):
                        stage_order_action[i] = supplier_order_dict.get(f"agent{i}", 0)
                action_order_dict[f'stage_{stage}_agent_{agent}'] = stage_order_action

                print("action sup action", sup_action)
                print("action order action", stage_order_action)


        next_states, rewards, terminations, truncations, infos = im_env.step(order_dict=action_order_dict, sup_dict=action_sup_dict, dem_dict=action_dem_dict)
        next_state_dict = im_env.parse_state(next_states)
        all_state_dicts[period + 1] = next_state_dict
        all_action_order_dicts[period + 1] = action_order_dict
        all_action_sup_dicts[period + 1] = action_sup_dict
        all_action_dem_dicts[period + 1] = action_dem_dict
        all_reward_dicts[period + 1] = rewards
        episode_reward += sum(rewards.values())
        print(
            f"period = {period}, action_order_dict = {action_order_dict}, rewards = {rewards}, episode_reward = {episode_reward}, " \
            f"api_cost = {api_cost}")
        print('=' * 80)

    return episode_reward