import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import random

# ----------------------------
# Brain-GNN Core
# ----------------------------

class BrainGNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.node_state = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

        # "Self node"
        self.self_vector = nn.Parameter(torch.randn(hidden_dim))

        # Policy = "free will"
        self.policy = nn.Linear(hidden_dim, 3)

    def forward(self, adj):
        messages = torch.matmul(adj, self.node_state)
        messages = self.message(messages)

        new_states = []
        for i in range(self.num_nodes):
            new_states.append(
                self.update(messages[i], self.node_state[i])
            )

        self.node_state = nn.Parameter(torch.stack(new_states))

        return self.node_state

    def act(self):
        probs = torch.softmax(self.policy(self.self_vector), dim=0)
        return torch.multinomial(probs, 1).item()

# ----------------------------
# Graph Rewrite (Thought)
# ----------------------------

def rewrite_graph(adj, activity):
    adj = adj.clone()

    # stochastic rewiring
    if random.random() < 0.3:
        i, j = random.randint(0, len(adj)-1), random.randint(0, len(adj)-1)
        adj[i][j] = 1 - adj[i][j]

    # strengthen active connections
    for i in range(len(adj)):
        for j in range(len(adj)):
            if activity[i].norm() > 1.0:
                adj[i][j] = min(adj[i][j] + 0.1, 1.0)

    return adj

# ----------------------------
# Arithmetic Task
# ----------------------------

def arithmetic_data():
    # learn: 1 + 1 = 2
    x = torch.tensor([[1.0], [1.0]])
    y = torch.tensor([[2.0]])
    return x, y

# ----------------------------
# Physics Task (F = ma)
# ----------------------------

def physics_data():
    m = np.random.uniform(1, 5)
    a = np.random.uniform(0, 10)
    F = m * a

    return torch.tensor([m, a]), torch.tensor([F])

# ----------------------------
# Training Loop
# ----------------------------

def train():
    num_nodes = 10
    model = BrainGNN(num_nodes)
    adj = torch.rand(num_nodes, num_nodes)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for step in range(500):

        # choose task via "free will"
        action = model.act()

        if action == 0:
            x, y = arithmetic_data()
        else:
            x, y = physics_data()

        states = model(adj)

        # simple readout
        pred = states.mean(dim=0)[0:len(y)]

        loss = ((pred - y)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # graph rewrite = thought
        adj = rewrite_graph(adj, states.detach())

        if step % 50 == 0:
            print(f"Step {step}, Loss {loss.item():.4f}")

    return model


if __name__ == "__main__":
    train()
