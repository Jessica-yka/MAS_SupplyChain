import os
import sys
sys.path.append('/home/vislab/Yanjia/MAS_SupplyChain')
from src.gnn.supplychain_graphs import SupplyChainGraphsDataset
from src.gnn.supplychain_mas import SupplyChainMASTestDataset
from src.gnn.supplychain_mas import SupplyChainMASDataset
from src.gnn.supplychain_graphs_baseline import SupplyChainGraphsBaselineDataset
from src.gnn.supplychain_graphs_inference import SupplyChainGraphsInferenceDataset

load_dataset = {
    'supplychain_graphs': SupplyChainGraphsDataset,
    'supplychain_mas_test': SupplyChainMASTestDataset, # for testing the llama+GNN performance
    'supplychain_mas': SupplyChainMASDataset, # for deployed in the MAS system
    'supplychain_graphs_baseline': SupplyChainGraphsBaselineDataset,
    'supplychain_graphs_inference': SupplyChainGraphsInferenceDataset, 
}
