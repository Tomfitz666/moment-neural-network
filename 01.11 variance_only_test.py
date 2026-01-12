
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from mnn.mnn_core.nn.activation import OriginMnnActivation
from mnn.mnn_core.nn.criterion import CrossEntropyOnMean
from RMNN_model import RMNN_VarOnly
from BPTT_model import RMNN_BPTT_NEW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- hyperparams ----------------
N = 128
M = 784
T = 1
batch_size = 32
epochs = 3

def input_encoder(x):
    """
    x: (B, 1, 28, 28)
    """
    x = x.view(x.size(0), -1)   # (B, 784)

    mu_ff = x
    var_ff = x #+ 1e-3           # Poisson-like variance

    return mu_ff, var_ff
# ---------------- data ----------------
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=False
)

# ---------------- model ----------------
rmnn = RMNN_VarOnly(N, M).to(device)
bptt = RMNN_BPTT_NEW(rmnn, T).to(device)
readout = nn.Linear(N, 10).to(device)

optimizer = torch.optim.Adam(
    list(rmnn.parameters()) + list(readout.parameters()),
    lr=1e-3
)

criterion = CrossEntropyOnMean()

for epoch in range(epochs):
    # -------- training --------
    rmnn.train()
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        mu_ff, var_ff = input_encoder(x)
        mu_ff, var_ff = mu_ff.to(device), var_ff.to(device)

        B = x.size(0)
        mu0 = torch.zeros(B, N, device=device)
        var0 = torch.ones(B, N, device=device) # * 1e-3

        mu_T, var_T = bptt(mu0, var0, mu_ff, var_ff)

        logits = readout(mu_T)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # -------- testing --------
    rmnn.eval()
    correct = 0
    total = 0
    var_mean, var_max = 0.0, 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            mu_ff, var_ff = input_encoder(x)
            mu_ff, var_ff = mu_ff.to(device), var_ff.to(device)

            B = x.size(0)
            mu0 = torch.zeros(B, N, device=device)
            var0 = torch.ones(B, N, device=device) #* 1e-3

            mu_T, var_T = bptt(mu0, var0, mu_ff, var_ff)

            logits = readout(mu_T)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            var_mean += var_T.mean().item()
            var_max = max(var_max, var_T.max().item())

    acc = 100. * correct / total

    print(
        f"Epoch {epoch+1} | "
        f"train loss {total_loss/len(train_loader):.4f} | "
        f"acc {acc:.2f}% | "
        f"Var mean {var_mean/len(test_loader):.2e} | "
        f"Var max {var_max:.2e}"
    )