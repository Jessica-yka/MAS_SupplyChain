import os
import sys
sys.path.append('/home/vislab/Yanjia/MAS_SupplyChain')
from src.gnn.env_sc_graphs import SupplyChainGraphsDataset


load_dataset = {
    'supplychain_graphs': SupplyChainGraphsDataset,

}
