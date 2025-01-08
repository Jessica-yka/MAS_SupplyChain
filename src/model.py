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
from form_msg import generate_msg
from utils import visualize_state, save_string_to_file


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


def run_simulation(im_env, user_proxy, stage_agents, config_name):
   
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
    visualize_state(env=im_env, rewards={}, t=-1, save_prefix=config_name)
    
    for period in range(im_env.num_periods):
        state_dict = im_env.parse_state(im_env.state_dict)
        all_state_dicts[period] = state_dict
        action_order_dict = {}
        action_sup_dict = {}
        action_dem_dict = {}
        total_chat_summary = ""
        for stage in range(num_stages):
            for agent in range(num_agents_per_stage):
                stage_state = state_dict[f'stage_{stage}_agent_{agent}']

                message, state_info = generate_msg(stage=stage, agent=agent, stage_state=stage_state, im_env=im_env, \
                                       action_order_dict=action_order_dict, period=period)
                chat_result = user_proxy.initiate_chat(
                    stage_agents[stage*num_agents_per_stage+agent],
                    message={'content': ''.join(message)},
                    summary_method="last_msg",
                    max_turns=1,
                    clear_history=False,
                )
                chat_summary = chat_result.summary
                total_chat_summary += (state_info + chat_summary + '\n\n\n\n')
                api_cost += chat_result.cost['usage_including_cached_inference']['total_cost']
                # print(chat_summary)
                match = re.findall(r'\[(.*?)\]', chat_summary, re.DOTALL)
                
                sup_action = state_dict[f'stage_{stage}_agent_{agent}']['suppliers']
                if stage < num_stages - 1:
                    remove_sup = match[0]                
                    if remove_sup != "":
                        remove_sup = remove_sup.replace("agent", "")
                        remove_sup = [int(ind) for ind in remove_sup.split(", ")]
                        for ind in remove_sup:
                            sup_action[ind] = 0
                    add_sup = match[1]
                    if add_sup != "":
                        add_sup = add_sup.replace("agent", "")
                        add_sup = [int(ind) for ind in add_sup.split(", ")]
                        for ind in add_sup:
                            sup_action[ind] = 1
                action_sup_dict[f'stage_{stage}_agent_{agent}'] = sup_action
                
                stage_order_action = np.zeros(num_agents_per_stage, dtype=int)
                if stage < num_stages - 1:
                    match2 = match[2]
                else:
                    match2 = match[0]
                if match2:
                    supplier_order_dict = extract_pairs(match2)
                    for i in range(num_agents_per_stage):
                        stage_order_action[i] = supplier_order_dict.get(f"agent{i}", 0)
                action_order_dict[f'stage_{stage}_agent_{agent}'] = stage_order_action

                # print("action sup action", sup_action)
                # print("action order action", stage_order_action)
        save_string_to_file(data=total_chat_summary, save_path=config_name, t=period)
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
        visualize_state(env=im_env, rewards=rewards, t=period, save_prefix=config_name)

    return episode_reward