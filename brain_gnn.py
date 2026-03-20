import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# ----------------------------
# Brain-GNN Model
# ----------------------------

class BrainGNN(nn.Module):
    def __init__(self, num_nodes=10, hidden_dim=64):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.node_state = nn.Parameter(torch.randn(num_nodes, hidden_dim))

        self.input_encoder = nn.Linear(2, hidden_dim)

        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, adj, x):
        input_embed = self.input_encoder(x)

        # inject input into graph
        for i in range(self.num_nodes):
            self.node_state.data[i] += input_embed.squeeze(0) * 0.05

        messages = torch.matmul(adj, self.node_state)
        messages = self.message(messages)

        new_states = []
        for i in range(self.num_nodes):
            new_states.append(self.update(messages[i], self.node_state[i]))

        self.node_state = nn.Parameter(torch.stack(new_states))

        global_state = self.node_state.mean(dim=0)

        combined = torch.cat([global_state, input_embed.squeeze(0)], dim=0)
        combined = self.interaction(combined)

        return self.readout(combined)


# ----------------------------
# Graph Rewrite
# ----------------------------

def rewrite_graph(adj):
    new_adj = adj.clone()

    if random.random() < 0.02:
        i = random.randint(0, len(adj)-1)
        j = random.randint(0, len(adj)-1)
        new_adj[i][j] = 1 - new_adj[i][j]

    return 0.95 * adj + 0.05 * new_adj


# ----------------------------
# Datasets
# ----------------------------

def physics_batch():
    m = np.random.uniform(1, 5)
    a = np.random.uniform(0, 10)

    F = (m * a) / 50.0

    x = torch.tensor([[m, a]], dtype=torch.float32)
    y = torch.tensor([[F]], dtype=torch.float32)

    return x, y


def arithmetic_batch():
    x = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[2.0]], dtype=torch.float32)
    return x, y


# ----------------------------
# Training
# ----------------------------

def train(task="physics", steps=1000):
    model = BrainGNN()
    adj = torch.rand(model.num_nodes, model.num_nodes)

    optimizer = optim.Adam(model.parameters(), lr=0.003)

    for step in range(steps):

        if task == "physics":
            x, y = physics_batch()
        else:
            x, y = arithmetic_batch()

        pred = model(adj, x)
        loss = ((pred - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        adj = rewrite_graph(adj)

        if step % 100 == 0:
            print(f"[{task.upper()}] Step {step}, Loss {loss.item():.6f}")

    return model


# ----------------------------
# Evaluation
# ----------------------------

def evaluate_physics(model):
    print("\n🔬 Learned Physics Behavior:\n")

    for m in [1, 2, 3, 4]:
        for a in [2, 5, 8]:
            x = torch.tensor([[m, a]], dtype=torch.float32)
            pred = model(torch.eye(model.num_nodes), x).item()
            true = (m * a) / 50.0

            print(f"m={m}, a={a} → Pred={pred:.3f}, True={true:.3f}")


def evaluate_arithmetic(model):
    print("\n🧮 Learned Arithmetic:\n")

    x = torch.tensor([[1.0, 1.0]])
    pred = model(torch.eye(model.num_nodes), x).item()

    print(f"1 + 1 → Pred={pred:.3f}, Expected=2.0")


# ----------------------------
# Symbolic Extraction
# ----------------------------

def extract_symbolic_law(model):
    print("\n🔬 Symbolic Law Discovery...\n")

    data = []
    for _ in range(300):
        m = np.random.uniform(1, 5)
        a = np.random.uniform(0, 10)

        x = torch.tensor([[m, a]], dtype=torch.float32)
        pred = model(torch.eye(model.num_nodes), x).item()

        data.append((m, a, pred))

    data = np.array(data)
    m_vals = data[:, 0]
    a_vals = data[:, 1]
    y_vals = data[:, 2]

    # constant
    c = np.mean(y_vals)
    err_const = np.mean((y_vals - c)**2)

    # linear
    X_linear = np.stack([m_vals, a_vals], axis=1)
    k_linear, _, _, _ = np.linalg.lstsq(X_linear, y_vals, rcond=None)
    pred_linear = X_linear @ k_linear
    err_linear = np.mean((y_vals - pred_linear)**2)

    # multiplicative
    X_mult = (m_vals * a_vals).reshape(-1, 1)
    k_mult, _, _, _ = np.linalg.lstsq(X_mult, y_vals, rcond=None)
    pred_mult = X_mult @ k_mult
    err_mult = np.mean((y_vals - pred_mult)**2)

    print(f"Constant Error: {err_const:.6f}")
    print(f"Linear Error: {err_linear:.6f}")
    print(f"Multiplicative Error: {err_mult:.6f}")

    print("\n🏆 Discovered Law:")

    if err_mult < err_linear and err_mult < err_const:
        print(f"F ≈ {k_mult[0]:.4f} * (m × a)")
    elif err_linear < err_const:
        print(f"F ≈ {k_linear[0]:.4f} * m + {k_linear[1]:.4f} * a")
    else:
        print(f"F ≈ {c:.4f}")


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    print("=== Training Physics ===")
    physics_model = train("physics")

    evaluate_physics(physics_model)
    extract_symbolic_law(physics_model)

    print("\n=== Training Arithmetic ===")
    arithmetic_model = train("arithmetic")

    evaluate_arithmetic(arithmetic_model)
