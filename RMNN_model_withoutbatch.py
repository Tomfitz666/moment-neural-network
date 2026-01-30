
import torch
import torch.nn as nn
import numpy as np
from mnn.mnn_core.mnn_utils import Mnn_Core_Func
from momentactivationtrio_withoutbatch import MnnActivateTrio

class RMNN(nn.Module):
    


    def __init__(self, N, M):
        super().__init__()
        self.N = N
        self.M = M
        

        
       #self.W = nn.Parameter(torch.randn(N, N) * (1.0 / N) ** 0.5)
        W = torch.randn(N, N) * (1.0 / N) ** 0.01
        W.fill_diagonal_(0.5)
        self.W = nn.Parameter(W)
        self.V = nn.Parameter(torch.randn(N, M) * (1.0 / M) ** 0.01)

 

    def forward(self, mu, C, mu_ff, C_ff):
        """
        mu:    (N,)
        C:     (N,N)
        mu_ff: (M,)
        C_ff:  (M,M)
        """

       
        mu_bar = self.W @ mu + self.V @ mu_ff

        
        A = self.W @ C               
        Cff_bar = self.V @ C_ff @ self.V.t()

        
        AW = A * self.W

        sigma2_bar = AW.sum(dim=1) + torch.diagonal(Cff_bar)        
        sigma2_bar=torch.clamp(sigma2_bar,min=1e-4)
        mu_out, cov_out, chi = MnnActivateTrio.apply(mu_bar, sigma2_bar)

        s_bar=torch.sqrt(sigma2_bar)
        s_out=torch.sqrt(cov_out)
        eps=1e-8
        chi_eff=(s_out/(s_bar+1e-8))*chi
        #chi_eff= chi
        
        chi_vec = chi_eff.view(self.N, 1)  # (N,1)

       
        B = chi_vec * A
        C_out = B + B.t() + chi_vec * Cff_bar * chi_vec.t()

        return mu_out, C_out


#mcf = MnnActivateTrio
#activation = MomentActivation(mcf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




# N = 8
# M = 8
# rmnn = RMNN(N, M).to(device)

# mu = torch.randn(N, device=device)
# C = torch.eye(N, device=device)

# mu_ff = torch.randn(M, device=device)
# C_ff = torch.eye(M, device=device)

# mu_out, C_out = rmnn(mu, C, mu_ff, C_ff)
# print(mu_out)
# print(C_out)


# loss = mu_out.sum() + C_out.sum()
# loss.backward()

# print("grad W:", rmnn.W.grad)
# print("grad V:", rmnn.V.grad)

N = 10
M = 784
rmnn = RMNN(N, M).to(device)

mu = torch.randn(N, device=device, requires_grad=True) * 0.5
C = torch.eye(N, device=device, requires_grad=True) * 0.1
mu_ff = torch.randn(M, device=device, requires_grad=True) * 0.5
C_ff = torch.eye(M, device=device, requires_grad=True) * 0.1

mu_out, C_out = rmnn(mu, C, mu_ff, C_ff)


print("\n=== Diagnostic Info ===")
print(f"mu_out min: {mu_out.min():.2e}, max: {mu_out.max():.2e}")
print(f"Number of near-zero activations (< 1e-10): {(mu_out < 1e-10).sum().item()}")
print(f"Indices of near-zero activations: {torch.where(mu_out < 1e-10)[0].tolist()}")


dead_neurons = torch.where(mu_out < 1e-20)[0]
if len(dead_neurons) > 0:
    print(f"\n WARNING: {len(dead_neurons)} 'dead' neurons detected!")
    print(f"Dead neuron indices: {dead_neurons.tolist()}")

loss = mu_out.sum() + C_out.sum()

try:
    loss.backward()
    print("\n=== Gradients ===")
    
    
    nan_rows_W = torch.isnan(rmnn.W.grad).any(dim=1)
    nan_rows_V = torch.isnan(rmnn.V.grad).any(dim=1)
    
    print(f"W.grad: {nan_rows_W.sum().item()} rows contain NaN")
    print(f"NaN row indices in W: {torch.where(nan_rows_W)[0].tolist()}")
    print(f"V.grad: {nan_rows_V.sum().item()} rows contain NaN")
    print(f"NaN row indices in V: {torch.where(nan_rows_V)[0].tolist()}")
    
    
    W_grad_finite = rmnn.W.grad[~torch.isnan(rmnn.W.grad)]
    if len(W_grad_finite) > 0:
        print(f"\nW.grad (finite values) range: [{W_grad_finite.min():.2e}, {W_grad_finite.max():.2e}]")
        print(f"W.grad (finite values) mean: {W_grad_finite.mean():.2e}")
    
except RuntimeError as e:
    print(f"\nBackward failed: {e}")
