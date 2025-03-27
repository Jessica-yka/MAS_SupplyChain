## Generate Data
# python3 src/gnn/preprocess/graph_qa_data_generation.py --for_train
# python3 src/gnn/preprocess/graph_qa_data_generation.py

## Preprocess Data (Retrieval/KAPING)
# python3 src/gnn/preprocess/supplychain_graphs.py --for_train --require_retrieval

## Train
# LLAMA with GNN
# echo "LLAMA with GNN"
# python3 train.py --dataset supplychain_graphs --model_name graph_llm --seed 0 --batch_size 8 --llm_model_name 7b_chat --gnn_out_dim 4096 --max_txt_len 512 > log_train_graph_llm_7b_chat_8.log 2>&1
# echo "Done 7b_chat"
# python3 train.py --dataset supplychain_graphs --model_name graph_llm --seed 0 --batch_size 4 --llm_model_name 13b_chat --gnn_out_dim 5120 --max_txt_len 512 > log_train_graph_llm_13b_chat_4.log 2>&1
# echo "Done 13b_chat"

# LLAMA with Prompt Tuning
# echo "LLAMA with Prompt Tuning"
# python3 train.py --dataset supplychain_graphs --model_name pt_llm --seed 0 --batch_size 8 --llm_model_name 7b_chat --gnn_out_dim 4096 > log_train_pt_llm_7b_chat_8.log 2>&1
# echo "Done 7b_chat"
# python3 train.py --dataset supplychain_graphs --model_name pt_llm --seed 0 --batch_size 4 --llm_model_name 13b_chat --gnn_out_dim 5120 > log_train_pt_llm_13b_chat_4.log 2>&1
# echo "Done 13b_chat"

# LLAMA with GraphToken
echo "LLAMA with GraphToken"
python3 train.py --dataset supplychain_graphs_baseline --model_name graph_llm --seed 0 --batch_size 6 --llm_model_name 7b_chat --gnn_out_dim 4096 --max_txt_len 800 > log_train_graph_token_llm_7b_chat_8.log 2>&1
echo "Done 7b_chat"
python3 train.py --dataset supplychain_graphs_baseline --model_name graph_llm --seed 0 --batch_size 2 --llm_model_name 13b_chat --gnn_out_dim 5120 --max_txt_len 800 > log_train_graph_token_llm_13b_chat_4.log 2>&1
echo "Done 13b_chat"

# LLAMA with inference only (zero-shot/zeroshot+cot/cot+bag/kaping)
# python3 inference.py --dataset supplychain_graphs_inference --prompting_tech cot --model_name inference_llm --seed 0 --batch_size 8 --llm_model_name 7b_chat --max_txt_len 512 > log_inference_llm_7b_chat_cot.log 2>&1
# python3 inference.py --dataset supplychain_graphs_inference --prompting_tech cot --model_name inference_llm --seed 0 --batch_size 4 --llm_model_name 13b_chat --max_txt_len 512 > log_inference_llm_13b_chat_cot.log 2>&1

# python3 inference.py --dataset supplychain_graphs_inference --prompting_tech cot-bag --model_name inference_llm --seed 0 --batch_size 8 --llm_model_name 7b_chat --max_txt_len 512 > log_inference_llm_7b_chat_cotbag.log 2>&1
# python3 inference.py --dataset supplychain_graphs_inference --prompting_tech cot-bag --model_name inference_llm --seed 0 --batch_size 4 --llm_model_name 13b_chat --max_txt_len 512 > log_inference_llm_13b_chat_cotbag.log 2>&1

# python3 inference.py --dataset supplychain_graphs_inference --prompting_tech kaping --model_name inference_llm --seed 0 --batch_size 8 --llm_model_name 7b_chat --max_txt_len 512 > log_inference_llm_7b_chat_kaping.log 2>&1
# python3 inference.py --dataset supplychain_graphs_inference --prompting_tech kaping --model_name inference_llm --seed 0 --batch_size 4 --llm_model_name 13b_chat --max_txt_len 512 > log_inference_llm_13b_chat_kaping.log 2>&1



