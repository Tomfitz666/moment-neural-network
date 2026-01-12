
import torch
import torch.nn as nn

from mnn.mnn_core.nn.activation import OriginMnnActivation
from mnn.mnn_core.nn.criterion import CrossEntropyOnMean

# =========================
# RMNN (copy your class)
# =========================
class RMNN(torch.nn.Module):
    def __init__(self, N: int, M: int):
        super().__init__()
        self.N = N
        self.M = M
        self.activation = OriginMnnActivation()
        self.W = torch.nn.Parameter(torch.randn(N, N) / N**0.5)
        self.V = torch.nn.Parameter(torch.randn(N, M) / M**0.5)

    def forward(self, mu, C, mu_ff, C_ff):
        B, N = mu.shape

        mu_bar = (
            torch.einsum('ij,bj->bi', self.W, mu) +
            torch.einsum('ij,bj->bi', self.V, mu_ff)
        )

        A = torch.einsum('ij,bjk->bik', self.W, C)
        Cff_bar = torch.einsum(
            'ij,bjk,kl->bil',
            self.V, C_ff, self.V.T
        )

        sigma2_bar = (
            (A * self.W).sum(dim=2) +
            torch.diagonal(Cff_bar, dim1=1, dim2=2)
        )

        mu_out, sigma2_out = self.activation(mu_bar, sigma2_bar)

        with torch.no_grad():
            chi = torch.sqrt(sigma2_bar + 1e-8) / torch.sqrt(sigma2_out + 1e-8)
        chi = chi.detach()

        Bmat = chi.unsqueeze(2) * A
        C_out = (
            Bmat + Bmat.transpose(1, 2) +
            chi.unsqueeze(2) * Cff_bar * chi.unsqueeze(1)
        )

        return mu_out, C_out


# =========================
# Test script
# =========================
# def test_rmnn_forward():
#     torch.manual_seed(0)
#
#     B = 4      # batch size
#     N = 8      # hidden size
#     M = 10     # input size
#
#     device = "cpu"
#
#     rmnn = RMNN(N, M).to(device)
#
#     # Fake inputs
#     mu = torch.randn(B, N, device=device, requires_grad=True)
#     C = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
#     mu_ff = torch.randn(B, M, device=device)
#     C_ff = torch.eye(M, device=device).unsqueeze(0).repeat(B, 1, 1)
#
#     # Forward
#     mu_out, C_out = rmnn(mu, C, mu_ff, C_ff)
#
#     print("mu_out shape:", mu_out.shape)   # (B, N)
#     print("C_out shape :", C_out.shape)    # (B, N, N)
#
#     # Symmetry check
#     sym_err = (C_out - C_out.transpose(1, 2)).abs().max().item()
#     print("C_out symmetry error:", sym_err)
#
#     # Backward test
#     loss = mu_out.mean() + C_out.mean()
#     loss.backward()
#
#     print("Backward OK")
#     print("Grad W:", rmnn.W.grad.norm().item())
#     print("Grad V:", rmnn.V.grad.norm().item())
#
#
# if __name__ == "__main__":
#     test_rmnn_forward()

class RMNN_BPTT(nn.Module):
    def __init__(self, rmnn: RMNN, T: int):
        super().__init__()
        self.rmnn = rmnn
        self.T = T

    def forward(self, mu0, C0, mu_ff, C_ff):
        """
        mu0   : (B, N)    hidden initial state
        C0    : (B, N, N)
        mu_ff : (B, M)    MNIST input
        C_ff  : (B, M, M)
        """

        mu, C = mu0, C0

        for _ in range(self.T):
            mu, C = self.rmnn(mu, C, mu_ff, C_ff)

        return mu, C

def input_encoder(x):
    """
    x: (B, 1, 28, 28)
    """
    x = x.view(x.size(0), -1)            # (B, 784)
    mu_ff = x
    C_ff = torch.diag_embed(x + 1e-6)    # (B, 784, 784)
    return mu_ff, C_ff

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- hyperparams ----------------
N = 128        # hidden neurons
M = 784        # MNIST
T = 1
batch_size = 32
epochs = 3

# ---------------- data ----------------
train_loader = DataLoader(
    datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.ToTensor()
    ),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.ToTensor()
    ),
    batch_size=batch_size,
    shuffle=False
)

# ===== NEW: evaluation =====
def evaluate(bptt, readout, test_loader, device):
    bptt.eval()
    readout.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    cov_diag_max = []
    cov_diag_mean = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            mu_ff, C_ff = input_encoder(x)
            mu_ff, C_ff = mu_ff.to(device), C_ff.to(device)

            B = x.size(0)
            mu0 = torch.zeros(B, bptt.rmnn.N, device=device)
            C0  = torch.eye(bptt.rmnn.N, device=device).unsqueeze(0).repeat(B,1,1) * 1e-6

            mu_T, C_T = bptt(mu0, C0, mu_ff, C_ff)

            logits = readout(mu_T)
            loss = criterion(logits, y)
            total_loss += loss.item()

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            # ===== covariance diagnostics =====
            diag = torch.diagonal(C_T, dim1=1, dim2=2)
            cov_diag_max.append(diag.max().item())
            cov_diag_mean.append(diag.mean().item())

    return {
        "loss": total_loss / len(test_loader),
        "acc": correct / total,
        "cov_diag_max": max(cov_diag_max),
        "cov_diag_mean": sum(cov_diag_mean) / len(cov_diag_mean),
    }

# ---------------- model ----------------
rmnn = RMNN(N, M).to(device)
bptt = RMNN_BPTT(rmnn, T).to(device)

readout = nn.Linear(N, 10).to(device)

optimizer = torch.optim.Adam(
    list(rmnn.parameters()) + list(readout.parameters()),
    lr=1e-3
)

#criterion = nn.CrossEntropyLoss()
criterion = CrossEntropyOnMean()

# ---------------- training ----------------
# for epoch in range(epochs):
#     total_loss = 0.0

#     for x, y in train_loader:
#         x, y = x.to(device), y.to(device)

#         mu_ff, C_ff = input_encoder(x)
#         mu_ff, C_ff = mu_ff.to(device), C_ff.to(device)

#         B = x.size(0)

#         mu0 = torch.zeros(B, N, device=device)
#         C0  = torch.eye(N, device=device).unsqueeze(0).repeat(B,1,1) * 1e-6

#         mu_T, _ = bptt(mu0, C0, mu_ff, C_ff)

#         logits = readout(mu_T)
#         loss = criterion(logits, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1}, loss = {total_loss / len(train_loader):.4f}")

for epoch in range(epochs):
    rmnn.train()
    readout.train()

    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        mu_ff, C_ff = input_encoder(x)
        mu_ff, C_ff = mu_ff.to(device), C_ff.to(device)

        B = x.size(0)
        mu0 = torch.zeros(B, N, device=device)
        C0  = torch.eye(N, device=device).unsqueeze(0).repeat(B,1,1) * 1e-6

        mu_T, C_T = bptt(mu0, C0, mu_ff, C_ff)

        logits = readout(mu_T)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # ===== NEW: test =====
    test_stats = evaluate(bptt, readout, test_loader, device)

    print(
        f"Epoch {epoch+1} | "
        f"train loss {train_loss:.4f} | "
        f"test loss {test_stats['loss']:.4f} | "
        f"acc {test_stats['acc']*100:.2f}% | "
        f"Var mean {test_stats['cov_diag_mean']:.2e} | "
        f"Var max {test_stats['cov_diag_max']:.2e}"
    )
    