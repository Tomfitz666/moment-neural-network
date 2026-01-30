import torch
import torch.nn as nn
import numpy as np
from mnn.mnn_core.mnn_utils import Mnn_Core_Func
from momentactivationtrio import MnnActivateTrio



class RMNN(nn.Module):
    


    def __init__(self, N, M):
        super().__init__()
        self.N = N
        self.M = M
        

        
       #self.W = nn.Parameter(torch.randn(N, N) * (1.0 / N) ** 0.5)
        W = torch.randn(N, N) * (1.0 / N) ** 0.5
        W.fill_diagonal_(2.0)
        self.W = nn.Parameter(W)
        self.V = nn.Parameter(torch.randn(N, M) * (1.0 / M) ** 0.5)

     
    def forward(self, mu, C, mu_ff, C_ff):


    # ---- mean ----
        mu_bar = (
        torch.einsum("ij,bj->bi", self.W, mu)
      + torch.einsum("ik,bk->bi", self.V, mu_ff)
    )
        

    # ---- covariance propagation ----
        A = torch.einsum("ik,bkj->bij", self.W, C)
        
        Cff_bar = torch.einsum("ik,bkl,jl->bij", self.V, C_ff, self.V)

        AW = A * self.W.unsqueeze(0)
    
        sigma2_bar = (
        AW.sum(dim=2)
      + torch.diagonal(Cff_bar, dim1=1, dim2=2)
    )
       
        sigma2_bar = sigma2_bar.clamp_min(1e-1)
        
        
        
        mu_out, cov_out, chi = MnnActivateTrio.apply(mu_bar, sigma2_bar)

    
        s_bar = torch.sqrt(sigma2_bar)
        s_out = torch.sqrt(cov_out.clamp_min(1e-12))
        ratio = s_out / (s_bar + 1e-4)
        ratio = torch.clamp(ratio, max=1e4)

        chi_eff =ratio * chi
        chi_i = chi_eff            # (B, N)
        chi_j = chi_eff            # (B, N)

        chi_outer = chi_i.unsqueeze(2) * chi_j.unsqueeze(1)  # (B, N, N)
        #chi_vec = chi_eff.unsqueeze(-1)  # (B, N, 1)

    # ---- output covariance ----
        B = chi_eff.unsqueeze(2) * A

        C_out = (
        B
      + B.transpose(1, 2)
      + chi_outer * Cff_bar 
    )



        return mu_out, C_out






  

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# dimensions
# -----------------------------
B = 1    # batch
N = 128  # recurrent units
M = 784  # feedforward units


model = RMNN(N, M).to(device)

# -----------------------------
# inputs
# -----------------------------
mu = torch.randn(B, N, device=device, requires_grad=True)
C = torch.randn(B, N, N, device=device)
C = 0.5 * (C + C.transpose(-1, -2))   # make symmetric
C.requires_grad_(True)

mu_ff = torch.randn(B, M, device=device, requires_grad=True)
C_ff = torch.randn(B, M, M, device=device)
C_ff = 0.5 * (C_ff + C_ff.transpose(-1, -2))
C_ff.requires_grad_(True)

# -----------------------------
# forward
# -----------------------------
mu_out, C_out = model(mu, C, mu_ff, C_ff)

print("\n=== Forward outputs ===")
print("mu_out shape:", mu_out.shape)
print("mu_out sample:\n", mu_out[0])

print("\nC_out shape:", C_out.shape)
print("C_out[0] symmetry check:",
      torch.allclose(C_out[0], C_out[0].T, atol=1e-6))
print("C_out[0] sample:\n", C_out[0])

# -----------------------------
# loss
# -----------------------------
loss = mu_out.mean() + C_out.mean()
loss.backward()

# -----------------------------
# gradients
# -----------------------------
print("\n=== Gradients ===")

print("mu.grad sample:\n", mu.grad[0])
print("C.grad sample:\n", C.grad[0])

print("\nW.grad shape:", model.W.grad.shape)
print("W.grad sample:\n", model.W.grad)

print("\nV.grad shape:", model.V.grad.shape)
print("V.grad sample:\n", model.V.grad)

# -----------------------------
# sanity checks
# -----------------------------
def finite(x):
    return torch.isfinite(x).all().item()

print("\n=== Finite checks ===")
print("mu_out finite:", finite(mu_out))
print("C_out finite:", finite(C_out))
print("mu.grad finite:", finite(mu.grad))
print("C.grad finite:", finite(C.grad))
print("W.grad finite:", finite(model.W.grad))
print("V.grad finite:", finite(model.V.grad))

print("\nDemo finished.")
