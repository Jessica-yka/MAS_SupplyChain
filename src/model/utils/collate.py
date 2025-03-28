from torch_geometric.data import Batch


def collate_fn(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        if k == 'node_id_name_map':
            continue
        batch[k] = [d[k] for d in original_batch]
    if 'graph' in batch:
        batch['graph'] = Batch.from_data_list(batch['graph'])
    return batch

def collate_mas_fn(original_batch):
    batch = {}
    # print('original_batch[0].keys()', original_batch[0].keys())
    for k in original_batch[0].keys():
        # print('original_batch[0][k].keys()', original_batch[0][k].keys())
        batch[k] = {}
        for m in original_batch[0][k].keys():
            batch[k][m] = [d[k][m] for d in original_batch]
        if 'graph' in batch[k]:
            batch[k]['graph'] = Batch.from_data_list(batch[k]['graph'])
    return batch