import torch
import torch.nn as nn
import numpy as np
from mnn.mnn_core.mnn_utils import Mnn_Core_Func
from momentactivationtrio import MnnActivateTrio



# class RMNN(nn.Module):
    


#     def __init__(self, N, M):
#         super().__init__()
#         self.N = N
#         self.M = M
        

        
#        #self.W = nn.Parameter(torch.randn(N, N) * (1.0 / N) ** 0.5)
#         W = torch.randn(N, N) * (1.0 / N) ** 0.5
#         W.fill_diagonal_(2.0)
#         self.W = nn.Parameter(W)
#         self.V = nn.Parameter(torch.randn(N, M) * (1.0 / M) ** 0.5)

     
#     def forward(self, mu, C, mu_ff, C_ff):


#     # ---- mean ----
#         mu_bar = (
#         torch.einsum("ij,bj->bi", self.W, mu)
#       + torch.einsum("ik,bk->bi", self.V, mu_ff)
#     )
        

#     # ---- covariance propagation ----
#         A = torch.einsum("ik,bkj->bij", self.W, C)
        
#         Cff_bar = torch.einsum("ik,bkl,jl->bij", self.V, C_ff, self.V)

#         AW = A * self.W.unsqueeze(0)
    
#         sigma2_bar = (
#         AW.sum(dim=2)
#       + torch.diagonal(Cff_bar, dim1=1, dim2=2)
#     )
       
#         sigma2_bar = sigma2_bar.clamp_min(1e-1)
        
        
        
#         mu_out, cov_out, chi = MnnActivateTrio.apply(mu_bar, sigma2_bar)

    
#         s_bar = torch.sqrt(sigma2_bar)
#         s_out = torch.sqrt(cov_out.clamp_min(1e-12))
#         ratio = s_out / (s_bar + 1e-4)
#         ratio = torch.clamp(ratio, max=1e4)

#         chi_eff =ratio * chi
#         chi_i = chi_eff            # (B, N)
#         chi_j = chi_eff            # (B, N)

#         chi_outer = chi_i.unsqueeze(2) * chi_j.unsqueeze(1)  # (B, N, N)
#         #chi_vec = chi_eff.unsqueeze(-1)  # (B, N, 1)

#     # ---- output covariance ----
#         B = chi_eff.unsqueeze(2) * A

#         C_out = (
#         B
#       + B.transpose(1, 2)
#       + chi_outer * Cff_bar 
#     )



#         return mu_out, C_out


class RMNN(nn.Module):
    def __init__(self, N, M, eps=1e-5, momentum=0.1):
        super().__init__()
        self.N = N
        self.M = M
        self.eps = eps
        self.momentum = momentum
        
        
        # self.W = nn.Parameter(torch.empty(N, N))
        # self.V = nn.Parameter(torch.empty(N, M))
        
       
        # nn.init.kaiming_uniform_(self.W, a=0, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.V, a=0, mode='fan_in', nonlinearity='relu')
        
        
        # self.gamma = nn.Parameter(torch.ones(N))
        # self.beta = nn.Parameter(torch.zeros(N))

        self.W = nn.Parameter(torch.empty(N, N))
        self.V = nn.Parameter(torch.empty(N, M))
        
        
        # 递归权重需要谱半径 < 1 保证稳定
        nn.init.normal_(self.W, mean=0, std=1.0 / N)
        nn.init.xavier_uniform_(self.V)
        
        # beta 正偏置，让 BN 输出偏正，减少死神经元
        self.gamma = nn.Parameter(torch.ones(N))
        self.beta = nn.Parameter(torch.ones(N) * 1.0)  


        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(N))
        self.register_buffer('running_var', torch.ones(N))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, mu, C, mu_ff, C_ff):
    # ===== Step 1: Synaptic summation =====
      mu_hat = (
          torch.einsum("ij,bj->bi", self.W, mu) +
          torch.einsum("ik,bk->bi", self.V, mu_ff)
      )
      
      A_hat = torch.einsum("ik,bkj->bij", self.W, C)
      C_ff_hat = torch.einsum("ik,bkl,jl->bij", self.V, C_ff, self.V)
      
      AW = A_hat * self.W.unsqueeze(0)
      sigma2_hat = AW.sum(dim=2) + torch.diagonal(C_ff_hat, dim1=1, dim2=2)

      # print(f"[DEBUG1] sigma2_hat range: [{sigma2_hat.min():.4e}, {sigma2_hat.max():.4e}]")
      # print(f"[DEBUG1] sigma2_hat 负值数量: {(sigma2_hat < 0).sum().item()}")


      sigma2_hat = torch.nn.functional.softplus(sigma2_hat) + 1e-6  
      
      # ===== Step 2: Moment batchnorm =====
      if self.training:
          mu_hat_mean = mu_hat.mean(dim=0)
          mu_hat_var = mu_hat.var(dim=0, unbiased=False)
          sigma2_hat_mean = sigma2_hat.mean(dim=0)
          nu = mu_hat_var + sigma2_hat_mean
          
          with torch.no_grad():
              self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu_hat_mean
              self.running_var = (1 - self.momentum) * self.running_var + self.momentum * nu  # 
              self.num_batches_tracked += 1
      else:
          mu_hat_mean = self.running_mean
          nu = self.running_var
      

      # print(f"[DEBUG2] nu range: [{nu.min():.4e}, {nu.max():.4e}]")
      # print(f"[DEBUG2] nu 接近零数量: {(nu < 1e-4).sum().item()}")


      inv_sqrt_nu = torch.rsqrt(nu + self.eps)  
      
      mu_bar = (mu_hat - mu_hat_mean) * inv_sqrt_nu * self.gamma + self.beta
      sigma2_bar = sigma2_hat * (inv_sqrt_nu ** 2) * (self.gamma ** 2)

      # print(f"[DEBUG3] mu_bar range: [{mu_bar.min():.4e}, {mu_bar.max():.4e}]")
      # print(f"[DEBUG3] sigma2_bar range: [{sigma2_bar.min():.4e}, {sigma2_bar.max():.4e}]")


      A_bar = A_hat * (inv_sqrt_nu * self.gamma).view(1, -1, 1)
      
      nu_i = nu.view(1, -1, 1)
      nu_j = nu.view(1, 1, -1)
      gamma_i = self.gamma.view(1, -1, 1)
      gamma_j = self.gamma.view(1, 1, -1)
      C_ff_bar = C_ff_hat / torch.sqrt((nu_i + self.eps) * (nu_j + self.eps)) * (gamma_i * gamma_j)
      
      # ===== Step 3: Moment activation =====
    #   mu_out, sigma2_out, chi_out = MnnActivateTrio.apply(mu_bar, sigma2_bar) 

    #   # print(f"[DEBUG4] mu_out: finite={torch.isfinite(mu_out).all()}, range=[{mu_out.min():.4e}, {mu_out.max():.4e}]")
    #   # print(f"[DEBUG4] sigma2_out: finite={torch.isfinite(sigma2_out).all()}, range=[{sigma2_out.min():.4e}, {sigma2_out.max():.4e}]")
    #   # print(f"[DEBUG4] chi_out: finite={torch.isfinite(chi_out).all()}, range=[{chi_out.min():.4e}, {chi_out.max():.4e}]")
      
    #   # ===== Step 4: Covariance mapping =====
    #   B_bar = chi_out.unsqueeze(2) * A_bar
    #   chi_i = chi_out.unsqueeze(2)
    #   chi_j = chi_out.unsqueeze(1)
    #   chi_outer = chi_i * chi_j
      
    #   C_out = B_bar + B_bar.transpose(1, 2) + chi_outer * C_ff_bar
      
    #   # 对角线设为 sigma2_out
    #   diag_idx = torch.arange(self.N, device=C_out.device)
    #   C_out[:, diag_idx, diag_idx] = sigma2_out
      
    #   # 对称
    #   C_out = (C_out + C_out.transpose(1, 2)) / 2

             # ===== Step 3: Moment activation =====
               # ===== Step 3: Moment activation =====
              # ===== Step 3: Moment activation =====
      mu_out, sigma2_out, chi_out = MnnActivateTrio.apply(mu_bar, sigma2_bar)
        
        # ✅ 修正增益系数
      sigma_bar = torch.sqrt(sigma2_bar.clamp_min(1e-8))   # (B, N)
      sigma_out = torch.sqrt(sigma2_out.clamp_min(1e-8))   # (B, N)
      gain = chi_out * (sigma_out / sigma_bar)              # (B, N)
        
        # ===== Step 4: Covariance mapping =====
      B_bar = gain.unsqueeze(2) * A_bar                     # (B, N, N)
        
      gain_i = gain.unsqueeze(2)                            # (B, N, 1)
      gain_j = gain.unsqueeze(1)                            # (B, 1, N)
      gain_outer = gain_i * gain_j                          # (B, N, N)
        
      C_out = B_bar + B_bar.transpose(1, 2) + gain_outer * C_ff_bar
        
        # 对角线设为 sigma2_out
      diag_idx = torch.arange(self.N, device=C_out.device)
      C_out[:, diag_idx, diag_idx] = sigma2_out
        
        # 强制对称
      C_out = (C_out + C_out.transpose(1, 2)) / 2  
    
    



      
      return mu_out, C_out
    










if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B = 32   # batch size
    N = 128
    M = 128
    
    rmnn = RMNN(N, M).to(device)
    
    # 输入 batch
    mu = torch.randn(B, N, device=device, requires_grad=True) * 1.0
    C = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1) * 0.5
    C.requires_grad = True
    mu_ff = torch.randn(B, M, device=device, requires_grad=True) * 1.0
    C_ff = torch.eye(M, device=device).unsqueeze(0).repeat(B, 1, 1) * 0.5
    C_ff.requires_grad = True
    
    # Forward
    rmnn.train()  # 训练模式
    mu_out, C_out = rmnn(mu, C, mu_ff, C_ff)
    
    print(f"mu_out shape: {mu_out.shape}")
    print(f"C_out shape: {C_out.shape}")
    print(f"mu_out range: [{mu_out.min():.2e}, {mu_out.max():.2e}]")
    print(f"C_out range: [{C_out.min():.2e}, {C_out.max():.2e}]")
    
    # Backward
    loss = mu_out.sum() + C_out.sum()
    loss.backward()
    
    print(f"\nGradients:")
    print(f"W.grad finite: {torch.isfinite(rmnn.W.grad).all().item()}")
    print(f"V.grad finite: {torch.isfinite(rmnn.V.grad).all().item()}")
    print(f"gamma.grad finite: {torch.isfinite(rmnn.gamma.grad).all().item()}")
    print(f"beta.grad finite: {torch.isfinite(rmnn.beta.grad).all().item()}")
    
    if (torch.isfinite(rmnn.W.grad).all() and 
        torch.isfinite(rmnn.V.grad).all() and
        torch.isfinite(rmnn.gamma.grad).all() and
        torch.isfinite(rmnn.beta.grad).all()):
        print("✅ All gradients are finite! Ready to train.")
    else:
        print("⚠️  Some gradients are NaN/Inf")
        
    # 检查死神经元
    dead_neurons = (mu_out < 1e-20).sum(dim=0)
    print(f"\nDead neurons per position: {dead_neurons.sum().item()} / {B * N}")








  

# torch.manual_seed(0)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)

# # -----------------------------
# # dimensions
# # -----------------------------
# B = 1    # batch
# N = 128  # recurrent units
# M = 784  # feedforward units


# model = RMNN(N, M).to(device)

# # -----------------------------
# # inputs
# # -----------------------------
# mu = torch.randn(B, N, device=device, requires_grad=True)
# C = torch.randn(B, N, N, device=device)
# C = 0.5 * (C + C.transpose(-1, -2))   # make symmetric
# C.requires_grad_(True)

# mu_ff = torch.randn(B, M, device=device, requires_grad=True)
# C_ff = torch.randn(B, M, M, device=device)
# C_ff = 0.5 * (C_ff + C_ff.transpose(-1, -2))
# C_ff.requires_grad_(True)

# # -----------------------------
# # forward
# # -----------------------------
# mu_out, C_out = model(mu, C, mu_ff, C_ff)

# print("\n=== Forward outputs ===")
# print("mu_out shape:", mu_out.shape)
# print("mu_out sample:\n", mu_out[0])

# print("\nC_out shape:", C_out.shape)
# print("C_out[0] symmetry check:",
#       torch.allclose(C_out[0], C_out[0].T, atol=1e-6))
# print("C_out[0] sample:\n", C_out[0])

# # -----------------------------
# # loss
# # -----------------------------
# loss = mu_out.mean() + C_out.mean()
# loss.backward()

# # -----------------------------
# # gradients
# # -----------------------------
# print("\n=== Gradients ===")

# print("mu.grad sample:\n", mu.grad[0])
# print("C.grad sample:\n", C.grad[0])

# print("\nW.grad shape:", model.W.grad.shape)
# print("W.grad sample:\n", model.W.grad)

# print("\nV.grad shape:", model.V.grad.shape)
# print("V.grad sample:\n", model.V.grad)

# # -----------------------------
# # sanity checks
# # -----------------------------
# def finite(x):
#     return torch.isfinite(x).all().item()

# print("\n=== Finite checks ===")
# print("mu_out finite:", finite(mu_out))
# print("C_out finite:", finite(C_out))
# print("mu.grad finite:", finite(mu.grad))
# print("C.grad finite:", finite(C.grad))
# print("W.grad finite:", finite(model.W.grad))
# print("V.grad finite:", finite(model.V.grad))

# print("\nDemo finished.")
