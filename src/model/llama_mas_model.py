import os
import sys
sys.path.append('/data/yanjia/MAS_SupplyChain')
import torch
import wandb
import gc
import re
import numpy as np
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data.data import Data
import json
import pandas as pd
from utils.seed import seed_everything

from llm_config import parse_args_llama
from src.model import llama_model_path
from src.model import load_model as load_llm_model
from src.model.utils.collate import collate_fn
from src.model.utils.ckpt import _reload_best_model
import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from utils.utils import extract_pairs
from utils.generate_llama_message import generate_questions, generate_graph_description
from utils.utils import visualize_state, save_string_to_file
from utils.utils import update_sup_action
from src.gnn.preprocess.lm_modeling import load_model, load_text2embedding
from src.gnn.preprocess.utils.retrieval import retrieval_via_pcst

from src.model.env import env_creator
from src.model.config import env_configs_list, get_env_configs
from src.model.llm_config import llm_config_list
from src.model.utils.utils import get_demand_description
from src.model.utils.utils import clear_dir


def create_agents(num_stages: int, num_agents_per_stage: int, args) -> List[AutoModelForCausalLM]:
    
    agents = []

    # Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name]

    model = load_llm_model[args.model_name](graph_type='Contextualized Supply Chain Graph', args=args) 
    model = _reload_best_model(model, args)

    agents.append(model)

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
        # im_env.sc_graph.update_graph(state_dict=state_dict, past_req_orders=past_req_orders) 

        all_state_dicts[period] = state_dict
        action_order_dict = {}
        action_price_dict = {}
        action_sup_dict = {}
        action_dem_dict = {}
        total_chat_summary = ""
        t_emergent_events = im_env.emergent_events.get(period, {'events': [], 'affected_agents': []})
        # for event in t_emergent_events:
            # if event == "demand_surge":
            #     print("There is a sudden demand surge. ")
            #     im_env.create_demand_surge()
            # if event == "sudden_shutdown":
            #     print("There is a sudden shutdown event. ")
            #     shutdown_list = im_env.shut_seq[period]
            #     for stage_id, agent_id in shutdown_list:
            #         state_dict = im_env.create_shutdown_event(stage_id, agent_id, state_dict)
                    
            # if event == "recovery":
            #     print("Here is a recovery event. ")
            #     recovery_list = im_env.rec_seq[period]
            #     for (stage_id, agent_id) in recovery_list:
            #         im_env.create_recovery_event(stage_id, agent_id)

        for stage_id in range(num_stages):
            for agent_id in range(num_agents_per_stage):
                
                if im_env.running_agents[stage_id][agent_id] == 0:
                    action_sup_dict[f"stage_{stage_id}_agent_{agent_id}"] = np.zeros(num_agents_per_stage, dtype=int)
                    action_order_dict[f"stage_{stage_id}_agent_{agent_id}"] = np.zeros(num_agents_per_stage, dtype=int)  
                    action_price_dict[f"stage_{stage_id}_agent_{agent_id}"] = 0
                elif (stage_id, agent_id) in llm_agent_set: # just to have only a few agents in the environment to be controlled by LLM
                    # stage_state = state_dict[f'stage_{stage_id}_agent_{agent_id}']
                    pr_orders = past_req_orders.get(f'stage_{stage_id}_agent_{agent_id}', [])
                    # wandb.log({"message": message})

                    # TODO: Formulate question and prompt without making a test dataloader
                    # model
                    model = user_proxy[0]
                    model.eval()
                    # question

                    prompt, thinking_pipeline, task_msg = generate_questions(emergent_events=t_emergent_events, stage_id=stage_id, im_env=im_env, guided_cot=True, action_order_dict=action_order_dict,
                                                                             period=period, agent_id=agent_id, enable_graph_change=enable_graph_change, enable_price_change=enable_price_change)
                    question = prompt+thinking_pipeline[0]
                    # edges
                    df_nodes, df_edges = generate_graph_description(emergent_events=t_emergent_events, state=state_dict, past_req_orders=pr_orders, stage_id=stage_id, agent_id=agent_id, num_stages=num_stages, num_agents_per_stage=num_agents_per_stage)
                    df_nodes.to_csv('df_nodes.csv', index=False)
                    df_edges.to_csv('df_edges.csv', index=False)
                    
                    subg, desc = encode_mas_supply_chain_graph(question=question, df_nodes=df_nodes, df_edges=df_edges, env_name=config_name)
                    samples = [{"id": f"t{period}s{stage_id}a{agent_id}", "question": question, "desc": desc, 'graph': subg, 'label': None}]
                    print('question', question)
                    output = model.inference(samples=collate_fn(samples))
                    print(output['pred'])
                    # match = re.findall(r'\[(.*?)\]', chat_summary, re.DOTALL)
                    exit()
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
                                    stage_order_action[i] = sup_action[i]*(supplier_order_dict.get(f"agent{i}", 0) + supplier_order_dict.get(f"stage_{stage_id+1}_agent_{i}", 0))
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
                                    stage_order_action[i] = sup_action[i]*(supplier_order_dict.get(f"agent{i}", 0) + supplier_order_dict.get(f"stage_{stage_id+1}_agent_{i}", 0))
                            except:
                                pass
                        action_order_dict[f'stage_{stage_id}_agent_{agent_id}'] = stage_order_action
                        print("stage_order_action", stage_order_action)
                        # if sum(stage_order_action)==0:
                        #     raise AssertionError("order action not recorded")
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
        llm_agent_rewards = {}
        round_reward_sum = 0
        for (stage_id, agent_id) in im_env.llm_agent_set:
            llm_agent_rewards[f'stage_{stage_id}_agent_{agent_id}'] = rewards[f'stage_{stage_id}_agent_{agent_id}']
            round_reward_sum += rewards[f'stage_{stage_id}_agent_{agent_id}']
        episode_reward += round_reward_sum
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
            f"llm_agent_rewards = {llm_agent_rewards}"
        )
        print(
            f"round_reward_sum = {round_reward_sum}"
        )
        visualize_state(env=im_env, rewards=rewards, t=period, save_prefix=config_name)
        save_string_to_file(data=total_chat_summary, save_path=config_name, t=period, round=round, reward=round_reward_sum)

    print(
        f"episode_reward = {episode_reward}"
    )
    print(f"api_cost = {api_cost}")
    print('=' * 80)
        
    return episode_reward

def encode_mas_supply_chain_graph(question: list, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, env_name: str):
    mas_model_path = f"output/mas_scm/{env_name}/"

    q_embs, graph = generate_text_embedding(path=mas_model_path, df_nodes=df_nodes, df_edges=df_edges, question=question)
    subg, desc = preprocess(q_embs, graph, df_nodes, df_edges, require_retrieve=True)

    return subg, desc


def generate_text_embedding(path: str, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, question:list):

    model_name = 'sbert'
    def _encode_questions():
        q_embs = text2embedding(model, tokenizer, device, question)

        return q_embs

    def _encode_graph():

        x = text2embedding(model, tokenizer, device, df_nodes.node_attr.tolist())
        e = text2embedding(model, tokenizer, device, df_edges.edge_attr.tolist())
        edge_index = torch.LongTensor([df_edges.src, df_edges.dst])
        graph = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(df_nodes))

        return graph


    os.makedirs(f'{path}/graphs/', exist_ok=True)
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    q_embs = _encode_questions()
    graph = _encode_graph()

    return q_embs, graph


def preprocess(q_embs, graph, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, require_retrieve=True):
    
    if require_retrieve:
        subg, desc = retrieval_via_pcst(graph, q_embs[0], df_nodes, df_edges, topk=12, topk_e=12, cost_e=0.5)
    else:
        subg = graph
        desc = df_nodes.to_csv(index=False)+'\n'+df_edges.to_csv(index=False)

    return subg, desc



def main(args):


    env_config_name = "large_graph_test"
    # create the dir to store the results
    os.makedirs(f"results/{env_config_name}", exist_ok=True)
    clear_dir(f"results/{env_config_name}")
    # create the dir to store the env setup
    os.makedirs(f"env/{env_config_name}", exist_ok=True)
    clear_dir(f"env/{env_config_name}")
    env_config = get_env_configs(env_configs=env_configs_list[env_config_name])
    im_env = env_creator(env_config)

 
    # %%
    print(env_config["demand_dist"])
    print(get_demand_description(env_config["demand_fn"]))

    rewards = []
    for r in tqdm(range(1)):
        print("\n\nNew round starts")
        stage_agents = create_agents(num_stages=env_config["num_stages"], num_agents_per_stage=env_config["num_agents_per_stage"], args=args)
        # stage_agents = []
        reward = run_simulation(im_env=im_env, user_proxy=stage_agents, stage_agents=stage_agents, config_name=env_config_name, round=r)
        rewards.append(reward)
        print(f"rewards = {reward}")
        # if reward < 0:
        #     raise AssertionError("The rewards are negative")

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print(f"Rewards: {rewards}")
    print(f"Mean Episode Reward: {mean_reward}")
    print(f"Standard Deviation of Episode Reward: {std_reward}")

if __name__ == "__main__":
    args = parse_args_llama()
    main(args)