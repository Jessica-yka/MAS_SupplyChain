import os
import sys
sys.path.append('/home/vislab/Yanjia/MAS_SupplyChain')
from src.gnn.supplychain_graphs import SupplyChainGraphsDataset
from src.gnn.supplychain_mas import SupplyChainMASDataset
from src.gnn.supplychain_graphs_baseline import SupplyChainGraphsBaselineDataset

load_dataset = {
    'supplychain_graphs': SupplyChainGraphsDataset,
    'supplychain_mas': SupplyChainMASDataset,
    'supplychain_graphs_baseline': SupplyChainGraphsBaselineDataset,
}
