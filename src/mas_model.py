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
from utils import update_sup_action

np.random.seed(0)

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




def run_simulation(im_env, user_proxy, stage_agents, config_name, round:int=0):
    
    all_state_dicts = {}
    all_action_order_dicts = {}
    all_action_price_dicts = {}
    all_action_sup_dicts = {}
    all_action_dem_dicts = {}
    all_reward_dicts = {}
    episode_reward = 0
    api_cost = 0
    shutdown_list = None
    recovery_list = None
    # print("reset")
    im_env.reset()
    num_stages = im_env.num_stages
    num_agents_per_stage = im_env.num_agents_per_stage
    llm_agent_set = im_env.llm_agent_set
    enable_graph_change = im_env.enable_graph_change
    enable_price_change = im_env.enable_price_change
    visualize_state(env=im_env, rewards={}, t=-1, save_prefix=config_name)
    
    for period in range(im_env.num_periods):
        # retrieve the latest env info
        state_dict = im_env.parse_state(im_env.state_dict)
        if period > 0:
            past_req_orders = all_action_order_dicts[period]
        else:
            past_req_orders = dict()
        # update the nx supply chain graph with the latest env
        im_env.sc_graph.update_graph(state_dict=state_dict, past_req_orders=past_req_orders) 

        all_state_dicts[period] = state_dict
        action_order_dict = {}
        action_price_dict = {}
        action_sup_dict = {}
        action_dem_dict = {}
        total_chat_summary = ""
        emergent_events = im_env.emergent_events[period]
        for event in emergent_events:
            if event == "demand_surge":
                print("There is a sudden demand surge. ")
                im_env.create_demand_surge()
            if event == "sudden_shutdown":
                print("There is a sudden shutdown event. ")
                shutdown_list = im_env.shut_seq[period]
                for stage_id, agent_id in shutdown_list:
                    state_dict = im_env.create_shutdown_event(stage_id, agent_id, state_dict)
                    
                # while True:
                #     print("Input the stage and agent id to close the company. Press enter to close the session.")
                #     input_stage_id = input("stage id (int)")
                #     input_agent_id = input("agent id (int)")
                #     try:
                #         input_stage_id = int(input_stage_id)
                #         input_agent_id = int(input_agent_id)
                #         im_env.create_sudden_shutdown(input_stage_id, input_agent_id)
                #     except ValueError:
                #         print("The session is closed.")
                #         break
            if event == "recovery":
                print("Here is a recovery event. ")
                recovery_list = im_env.rec_seq[period]
                for (stage_id, agent_id) in recovery_list:
                    im_env.create_recovery_event(stage_id, agent_id)
                # while True:
                #     im_env.get_all_shutdown_agents()
                #     print("Input the stage and agent id to re-open the compan(ies). Press enter to close the session.")
                #     input_stage_id = input("stage id (int)")
                #     input_agent_id = input("agent id (int)")
                #     try:
                #         input_stage_id = int(input_stage_id)
                #         input_agent_id = int(input_agent_id)
                #         im_env.create_recovery_event(input_stage_id, input_agent_id)
                #     except ValueError:
                #         print("The session is closed.")
                #         break
        for stage_id in range(num_stages):
            for agent_id in range(num_agents_per_stage):
                
                if im_env.running_agents[stage_id][agent_id] == 0:
                    action_sup_dict[f"stage_{stage_id}_agent_{agent_id}"] = np.zeros(num_agents_per_stage, dtype=int)
                    action_order_dict[f"stage_{stage_id}_agent_{agent_id}"] = np.zeros(num_agents_per_stage, dtype=int)  
                    action_price_dict[f"stage_{stage_id}_agent_{agent_id}"] = 0
                elif (stage_id, agent_id) in llm_agent_set: # just to have only a few agents in the environment to be controlled by LLM
                    stage_state = state_dict[f'stage_{stage_id}_agent_{agent_id}']
                    pr_orders = past_req_orders.get(f'stage_{stage_id}_agent_{agent_id}', [])
                    message, state_info = generate_msg(shutdown_list=shutdown_list, recovery_list=recovery_list, enable_graph_change=enable_graph_change, stage_id=stage_id, \
                                                       cur_agent_id=agent_id, stage_state=stage_state, im_env=im_env, enable_price_change=enable_price_change, 
                                                       action_order_dict=action_order_dict, past_req_orders=pr_orders, period=period)
                    chat_result = user_proxy.initiate_chat(
                        stage_agents[stage_id*num_agents_per_stage+agent_id],
                        message={'content': ''.join(message)},
                        summary_method="last_msg",
                        max_turns=1,
                        clear_history=False,
                    )
                    chat_summary = chat_result.summary
                    total_chat_summary += (message + chat_summary + '\n\n\n\n')
                    api_cost += chat_result.cost['usage_including_cached_inference']['total_cost']
                    # print(chat_summary)
                    match = re.findall(r'\[(.*?)\]', chat_summary, re.DOTALL)

                    if enable_graph_change:
                        sup_action = state_dict[f'stage_{stage_id}_agent_{agent_id}']['suppliers']
                        if stage_id < num_stages - 1:
                            sup_action = update_sup_action(sup_action=sup_action, rm_match=match[0], add_match=match[1])
                        action_sup_dict[f'stage_{stage_id}_agent_{agent_id}'] = sup_action

                        stage_order_action = np.zeros(num_agents_per_stage, dtype=int)
                        if stage_id < num_stages - 1:
                            match2 = match[2]
                        else:
                            match2 = match[0]
                        if match2:
                            supplier_order_dict = extract_pairs(match2)
                            try:
                                for i in range(num_agents_per_stage):
                                    stage_order_action[i] = supplier_order_dict.get(f"agent{i}", 0) + supplier_order_dict.get(f"stage_{stage_id+1}_agent_{i}", 0)
                            except:
                                pass
                        action_order_dict[f'stage_{stage_id}_agent_{agent_id}'] = stage_order_action
                        print("stage_order_action", stage_order_action)
                    else:
                        sup_action = state_dict[f'stage_{stage_id}_agent_{agent_id}']['suppliers']
                        action_sup_dict[f'stage_{stage_id}_agent_{agent_id}'] = sup_action
                        stage_order_action = np.zeros(num_agents_per_stage, dtype=int)
                        match = match[0]
                        if match:
                            supplier_order_dict = extract_pairs(match)
                            try: # if the string format is valid
                                for i in range(num_agents_per_stage):
                                    stage_order_action[i] = supplier_order_dict.get(f"agent{i}", 0) + supplier_order_dict.get(f"stage_{stage_id+1}_agent_{i}", 0)
                            except:
                                pass
                        action_order_dict[f'stage_{stage_id}_agent_{agent_id}'] = stage_order_action
                        print("stage_order_action", stage_order_action)
                        if sum(stage_order_action)==0:
                            raise AssertionError("order action not recorded")
                    if enable_price_change:
                        action_price_dict[f"stage_{stage_id}_agent_{agent_id}"] = match[-1]
                else:
                    action_sup_dict, action_order_dict, action_price_dict = im_env.no_backlog_env_proxy(stage_id=stage_id, agent_id=agent_id, action_order_dict=action_order_dict, 
                                                                                                        action_sup_dict=action_sup_dict, action_price_dict=action_price_dict)
                    
        
        next_states, rewards, terminations, truncations, infos = im_env.step(order_dict=action_order_dict, sup_dict=action_sup_dict, dem_dict=action_dem_dict, price_dict=action_price_dict)
        next_state_dict = im_env.parse_state(next_states)
        all_state_dicts[period + 1] = next_state_dict
        all_action_order_dicts[period + 1] = action_order_dict
        all_action_sup_dicts[period + 1] = action_sup_dict
        all_action_dem_dicts[period + 1] = action_dem_dict
        all_action_price_dicts[period + 1] = action_price_dict
        all_reward_dicts[period + 1] = rewards
        # episode_reward += sum(rewards.values())
        for (stage_id, agent_id) in im_env.llm_agent_set:
            episode_reward += rewards[f'stage_{stage_id}_agent_{agent_id}']
        # print(
        #     f"period = {period}, action_order_dict = {action_order_dict}, rewards = {rewards}, episode_reward = {episode_reward}, " \
        #     f"api_cost = {api_cost}")
        print(
            f"period = {period}"
        )
        # print(
        #     f"action_order_dict = {action_order_dict},"
        # )
        print(
            f"rewards = {rewards}"
        )
        print(
            f"episode_reward = {episode_reward}"
        )
        print(f"api_cost = {api_cost}")
        print('=' * 80)
        visualize_state(env=im_env, rewards=rewards, t=period, save_prefix=config_name)
        save_string_to_file(data=total_chat_summary, save_path=config_name, t=period, round=round, reward=episode_reward)

    return episode_reward