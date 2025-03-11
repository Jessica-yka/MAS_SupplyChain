import os
import torch
import wandb
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import pandas as pd
from src.model.utils.seed import seed_everything
from src.model.llm_config import parse_args_llama
from src.model import load_model, llama_model_path
from src.gnn.dataset import load_dataset
from src.model.utils.evaluate import eval_funcs
from src.model.utils.collate import collate_fn


def main(args):

    # Step 1: Set up wandb
    seed = args.seed
    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_{args.model_name}_seed{seed}",
               config=args)

    seed_everything(seed=seed)
    print(args)

    # dataset = load_dataset[args.dataset](dataset='all_train_questions.csv')
    event_dataset = load_dataset[args.dataset](dataset='all_event_questions.csv', type='events_qa')
    order_fulfill_dataset = load_dataset[args.dataset](dataset='all_order_fulfill_questions.csv', type='order_fulfill_qa')
    price_dataset = load_dataset[args.dataset](dataset='all_price_questions.csv', type='price_qa')
    lead_time_dataset = load_dataset[args.dataset](dataset='all_lead_time_questions.csv', type='lead_time_qa')
    demand_dataset = load_dataset[args.dataset](dataset='all_demand_questions.csv', type='demand_qa')

    event_idx_split = event_dataset.get_idx_split() 
    order_fulfill_idx_split = order_fulfill_dataset.get_idx_split()
    price_idx_split = price_dataset.get_idx_split()
    lead_time_idx_split = lead_time_dataset.get_idx_split()
    demand_idx_split = demand_dataset.get_idx_split()

    # Step 2: Build Node Classification Dataset
    print("load test dataset")
    test_dataset = [price_dataset[i] for i in price_idx_split['test']] + [lead_time_dataset[i] for i in lead_time_idx_split['test']] + \
                    [event_dataset[i] for i in event_idx_split['test']] + [order_fulfill_dataset[i] for i in order_fulfill_idx_split['test']] + \
                    [demand_dataset[i] for i in demand_idx_split['test']]    # test_dataset = [price_dataset[i] for i in price_idx_split['test']] + [lead_time_dataset[i] for i in lead_time_idx_split['test']]

    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)
    
    # Step 3: Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph=event_dataset.graph, graph_type=event_dataset.graph_type, args=args)

    # Step 4. Evaluating
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    print(f'path: {path}')

    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))

    with open(path, "w") as f:
        for _, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    # Step 5. Post-processing & Evaluating
    acc = eval_funcs[args.dataset](path)
    print(f'Test Acc {acc}')
    wandb.log({'Test Acc': acc})


if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
