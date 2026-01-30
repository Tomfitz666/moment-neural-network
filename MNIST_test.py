
import torch
import torch.nn as nn
import numpy as np
from mnn.mnn_core.mnn_utils import Mnn_Core_Func
from momentactivationtrio import MnnActivateTrio
from RMNN_model import RMNN
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms







def input_encoder(data, scale=1.0, eps=1e-4):
    data = torch.flatten(data, start_dim=1)
    mu = data * scale

    # variance ≈ mu(1-mu)  （Bernoulli / firing-rate 合理）
    var = mu * (1.0 - mu)
    var = torch.clamp(var, min=eps)

    C = torch.diag_embed(var)
    return mu, C


class RMNNClassifier(nn.Module):
    def __init__(self, rmnn, N, num_classes=10, T=3):
        super().__init__()
        self.rmnn = rmnn
        self.T = T
        self.readout = nn.Linear(N, num_classes)

    def forward(self, mu0, C0, mu_ff, C_ff):
        """
        mu0 : (B, N)
        C0  : (B, N, N)
        mu_ff: (B, M)
        C_ff : (B, M, M)
        """
        
        mu, C = mu0, C0
        # for t in range(self.T):
        #     mu, C = self.rmnn(mu, C, mu_ff, C_ff)

        # if torch.isnan(mu).any() or torch.isinf(mu).any():
        #     print("NaN in mu at step", t)
        #     return torch.zeros(mu.size(0), 10, device=mu.device), mu, C

        # if torch.isnan(C).any() or torch.isinf(C).any():
        #      print("NaN in C at step", t)
        #      return torch.zeros(mu.size(0), 10, device=mu.device), mu, C
        
        for t in range(self.T):
            mu, C = self.rmnn(mu, C, mu_ff, C_ff)

            # ---- covariance stabilization ----
            C = 0.5 * (C + C.transpose(1, 2))
            C = C + 1e-6 * torch.eye(
                C.size(-1), device=C.device
            )

            # ---- truncated BPTT ----
            if t < self.T - 1:
                mu = mu.detach()
                C = C.detach()

        logits = self.readout(mu)
        return logits, mu, C


def train_one_epoch(model, loader, optimizer, criterion, N, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        mu_ff, C_ff = input_encoder(x)

        B = x.size(0)
        mu0 = torch.zeros(B, N, device=device)
        C0 = (
            torch.eye(N, device=device)
            .unsqueeze(0)
            .repeat(B, 1, 1)
            * 1e-6
        )

        logits, _, _ = model(mu0, C0, mu_ff, C_ff)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        correct += (logits.argmax(1) == y).sum().item()
        total += B

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, N, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        mu_ff, C_ff = input_encoder(x)

        B = x.size(0)
        mu0 = torch.zeros(B, N, device=device)
        C0 = (
            torch.eye(N, device=device)
            .unsqueeze(0)
            .repeat(B, 1, 1)
            * 1e-6
        )

        logits, _, _ = model(mu0, C0, mu_ff, C_ff)
        loss = criterion(logits, y)

        total_loss += loss.item() * B
        correct += (logits.argmax(1) == y).sum().item()
        total += B

    return total_loss / total, correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    N = 128
    M = 784
    T = 1
    batch_size = 32
    epochs = 2

    train_loader = DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    from RMNN_model import RMNN

    rmnn = RMNN(N, M).to(device)
    model = RMNNClassifier(rmnn, N, T=T).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, N, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, N, device
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f}, acc {test_acc:.4f}"
        )

if __name__ == "__main__":        
        main()