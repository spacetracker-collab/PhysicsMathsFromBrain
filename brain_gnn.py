
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import random

# ----------------------------
# Brain-GNN (Stable Version)
# ----------------------------

class BrainGNN(nn.Module):
    def __init__(self, num_nodes=10, hidden_dim=32):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.node_state = nn.Parameter(torch.randn(num_nodes, hidden_dim))

        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

        # Readout (law extraction layer)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, adj):
        messages = torch.matmul(adj, self.node_state)
        messages = self.message(messages)

        new_states = []
        for i in range(self.num_nodes):
            new_states.append(self.update(messages[i], self.node_state[i]))

        self.node_state = nn.Parameter(torch.stack(new_states))

        return self.node_state

    def predict(self):
        global_state = self.node_state.mean(dim=0)
        return self.readout(global_state)


# ----------------------------
# Stable Graph Rewrite
# ----------------------------

def rewrite_graph(adj, states):
    new_adj = adj.clone()

    # very small stochastic rewiring
    if random.random() < 0.05:
        i = random.randint(0, len(adj)-1)
        j = random.randint(0, len(adj)-1)
        new_adj[i][j] = 1 - new_adj[i][j]

    # smooth update (memory retention)
    adj = 0.9 * adj + 0.1 * new_adj
    return adj


# ----------------------------
# Arithmetic Dataset
# ----------------------------

def arithmetic_batch():
    x = torch.tensor([[1.0, 1.0]])
    y = torch.tensor([[2.0]])
    return x, y


# ----------------------------
# Physics Dataset (normalized)
# ----------------------------

def physics_batch():
    m = np.random.uniform(1, 5)
    a = np.random.uniform(0, 10)

    F = m * a
    F = F / 50.0  # normalization

    return torch.tensor([[m, a]]), torch.tensor([[F]])


# ----------------------------
# Training Function
# ----------------------------

def train(task="physics", steps=500):
    model = BrainGNN()
    adj = torch.rand(model.num_nodes, model.num_nodes)

    optimizer = optim.Adam(model.parameters(), lr=0.005)

    losses = []

    for step in range(steps):

        if task == "arithmetic":
            x, y = arithmetic_batch()
        else:
            x, y = physics_batch()

        states = model(adj)
        pred = model.predict()

        loss = ((pred - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        adj = rewrite_graph(adj, states.detach())

        losses.append(loss.item())

        if step % 50 == 0:
            print(f"[{task.upper()}] Step {step}, Loss {loss.item():.4f}")

    return model, losses


# ----------------------------
# Law Extraction
# ----------------------------

def extract_physics_law(model):
    print("\n🔬 Extracting learned physics relationship...\n")

    for m in [1, 2, 3, 4]:
        for a in [2, 5, 8]:
            F_true = (m * a) / 50.0

            model.node_state.data += torch.randn_like(model.node_state) * 0.01

            pred = model.predict().item()

            print(f"m={m}, a={a} → Predicted={pred:.3f}, True={F_true:.3f}")


def extract_arithmetic_law(model):
    print("\n🧮 Extracting arithmetic relationship...\n")

    pred = model.predict().item()
    print(f"Prediction for 1 + 1 → {pred:.3f} (Expected: 2.0)")


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    print("=== Training Physics ===")
    physics_model, _ = train(task="physics")

    extract_physics_law(physics_model)

    print("\n=== Training Arithmetic ===")
    arithmetic_model, _ = train(task="arithmetic")

    extract_arithmetic_law(arithmetic_model)
