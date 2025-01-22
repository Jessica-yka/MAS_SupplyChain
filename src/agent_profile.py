import numpy as np

class agent_profile():
    def __init__(self, stage_idx, agent_idx, prod_capacity, prod_cost, price, sales, inventory, backlog, backlog_cost, 
                 inventory_cost, order_cost, lead_time, delivery, suppliers, customers):
        self.name = f"stage_{stage_idx}_agent_{agent_idx}"
        self.stage = stage_idx
        self.agent = agent_idx
        self.prod_capacity = prod_capacity
        self.prod_cost = prod_cost
        self.price = price
        self.sales = sales
        self.inventory = inventory
        self.backlog = backlog
        self.backlog_cost = backlog_cost
        self.inventory_cost = inventory_cost
        self.order_cost = order_cost
        self.lead_time = lead_time
        self.fulfilled_rate = None
        self.delivery = delivery
        self.suppliers = suppliers
        self.customers = customers

    
    def update(self, sales, inventory, backlog, req_order, suppliers, customers):

        self.sales = sales
        self.inventory = inventory
        self.backlog = backlog
        self.fulfilled_rate = sales / req_order
        self.suppliers = suppliers
        self.customers = customers

    def get_node_features(self):
        return np.array([self.prod_capacity, self.prod_cost, self.price, self.sales, self.inventory, self.backlog, self.backlog_cost, 
                self.inventory_cost])
    
    def get_edge_features(self):
        return self.lead_time