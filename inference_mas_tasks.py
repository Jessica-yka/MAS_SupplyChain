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
from src.model.utils.collate import collate_fn, collate_mas_fn
from src.model.utils.ckpt import _reload_best_model



def main(args):

    # Step 1: Set up wandb
    seed = args.seed
    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_{args.model_name}_seed{seed}",
               config=args)

    seed_everything(seed=seed)
    print(args)
    mas_dataset = load_dataset[args.dataset](price_dataset='all_price_questions.csv', lead_time_dataset='all_lead_time_questions.csv')
    # event_idx_split = event_dataset.get_idx_split() 
    # supplier_idx_split = supplier_dataset.get_idx_split()
    idx_split = mas_dataset.get_idx_split()

    # Step 2: Build Node Classification Dataset
    print("load test dataset")
    # test_dataset = [event_dataset[i] for i in event_idx_split['test']] + [supplier_dataset[i] for i in supplier_idx_split['test']]
    mas_test_dataset = [mas_dataset[i] for i in idx_split['test']]

    num_data = len(mas_test_dataset)
    mas_test_loader = DataLoader(mas_test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_mas_fn)
    
    # Step 3: Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph_type="Contextualized Supply Chain Graph", args=args)
    if args.model_name != 'inference_llm':
        model = _reload_best_model(model, args)

    # Step 4. Evaluating
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    print(f'path: {path}')

    model.eval()
    progress_bar_test = tqdm(range(num_data))
 
    
    with open(path, "w") as f:
        for i, batch in enumerate(mas_test_loader):
            df = pd.DataFrame({"id": [], "pred": [], "label": [], "question": [], "desc": []})
            with torch.no_grad():
                if batch.get('event_data', None) is not None:
                    event_out = model.inference(batch['event_data'])
                    # print('event output', output)
                    df = pd.concat([df, pd.DataFrame(event_out)], axis=0)
                if batch.get('supplier_data', None) is not None:
                    supplier_out = model.inference(batch['supplier_data'])
                    # print('supplier output', output)
                    df = pd.concat([df, pd.DataFrame(supplier_out)], axis=0)
                if batch.get('price_data', None) is not None:
                    price_out = model.inference(batch['price_data'])
                    # print("price output", output)
                    df = pd.concat([df, pd.DataFrame(price_out)], axis=0)
                if batch.get('lead_time_data', None) is not None:
                    lead_time_out = model.inference(batch['lead_time_data'])
                    # print("lead_time output", output)
                    df = pd.concat([df, pd.DataFrame(lead_time_out)], axis=0)

                downstream_task = mas_dataset.get_downstream_question(id=i, price_out=price_out, lead_time_out=lead_time_out)
                downstream_task = collate_fn([downstream_task])
                output = model.inference(downstream_task)
                df = pd.concat([df, pd.DataFrame(output)], axis=0)

                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")

            progress_bar_test.update(1)

    # Step 5. Post-processing & Evaluating
    # acc = eval_funcs[args.dataset](path)
    # print(f'Test Acc {acc}')
    # wandb.log({'Test Acc': acc})


if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
