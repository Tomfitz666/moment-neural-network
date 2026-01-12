import torch
import torch.nn as nn
from mnn.mnn_core.nn.activation import OriginMnnActivation

class RMNN(nn.Module):
  

    def __init__(self, N: int, M: int, activation: OriginMnnActivation):
        super().__init__()
        self.N = N
        self.M = M
        self.activation = activation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.W = nn.Parameter(torch.randn(N, N,device=self.device) * (1.0 / N) ** 0.5)
        self.V = nn.Parameter(torch.randn(N, M,device=self.device) * (1.0 / M) ** 0.5)

        self.to(self.device)
   
    def compute_sigma_out(self, cov_out):
        
        if cov_out.dim() == 2:
            return torch.sqrt(torch.diag(cov_out))
        else:
            return torch.sqrt(cov_out)

    # def compute_chi(self, mu_bar, sigma2_bar, sigma_out):
    #     """
    #     Computes:
    #         chi_i = (sigma_bar[i] / sigma_out[i]) * ∂mu_out_i / ∂mu_bar_i
    #     """
    #     N = self.N

    #     sigma_bar = torch.sqrt(sigma2_bar)  
    #     chi = torch.zeros(N, device=mu_bar.device)

    #     # mu_bar must require gradient
    #     mu_bar_for_grad = mu_bar.clone().detach().requires_grad_(True)

    #     # Run activation to build graph
    #     mu_out2, _ = self.activation(mu_bar_for_grad, sigma2_bar)

    #     # Compute chi_i for each neuron i
    #     for i in range(N):
    #         grad_i = torch.autograd.grad(
    #             mu_out2[i],              # scalar
    #             mu_bar_for_grad,         # vector
    #             retain_graph=True,
    #             create_graph=False,
    #             allow_unused=True,
    #         )[0]

    #         if grad_i is None:
    #             dphi = 0.0
    #         else:
    #             dphi = grad_i[i]

    #         chi[i] = (sigma_bar[i] / sigma_out[i]) * dphi

    #     return chi
    # 


    
    def forward(self, mu, C, mu_ff, C_ff):
        mu     = mu.to(self.device)
        C = C.to(self.device)
        mu_ff  = mu_ff.to(self.device)
        C_ff   = C_ff.to(self.device)
        N = self.N

     
        mu_bar = self.W @ mu + self.V @ mu_ff               
        A = self.W @ C                                    
        Cff_bar = self.V @ C_ff @ self.V.t()                 
        AW = A * self.W
        sigma2_bar = AW.sum(dim=1) + torch.diag(Cff_bar)     
        mu_out, cov_out = self.activation(mu_bar, sigma2_bar)
       
        sigma_out = self.compute_sigma_out(cov_out)
        # chi = self.compute_chi(mu_bar, sigma2_bar, sigma_out)
        # chi = chi.detach() 
        # chi_vec = chi.view(N, 1)

        with torch.no_grad():
         chi = torch.sqrt(sigma2_bar) / (sigma_out + 1e-8)
        chi = chi.unsqueeze(-1)
        B = chi * A
        C_out = B + B.t() + chi * Cff_bar * chi.t()
        print("mu_out.grad_fn =", mu_out.grad_fn)
        return mu_out, C_out


class RMNN_VarOnly(nn.Module):
    """
    Variance-only Recurrent Moment Neural Network
    """

    def __init__(self, N: int, M: int):
        super().__init__()
        self.N = N
        self.M = M

        self.activation = OriginMnnActivation()

        # Trainable weights
        self.W = nn.Parameter(torch.randn(N, N) / N**0.5)
        self.V = nn.Parameter(torch.randn(N, M) / M**0.5)

    def forward(self, mu, var, mu_ff, var_ff):
        """
        mu     : (B, N)
        var    : (B, N)        # variance only
        mu_ff  : (B, M)
        var_ff : (B, M)
        """

        # ---------- mean propagation ----------
        mu_bar = (
            torch.einsum('ij,bj->bi', self.W, mu) +
            torch.einsum('ij,bj->bi', self.V, mu_ff)
        )  # (B, N)

        # ---------- variance propagation ----------
        # Var(Wx) = sum_j W_ij^2 Var(x_j)
        var_rec = torch.einsum(
            'ij,bj->bi', self.W**2, var
        )  # (B, N)

        var_ff_bar = torch.einsum(
            'ij,bj->bi', self.V**2, var_ff
        )  # (B, N)

        sigma2_bar = var_rec + var_ff_bar + 1e-8

        # ---------- moment activation ----------
        mu_out, sigma2_out = self.activation(mu_bar, sigma2_bar)

        return mu_out, sigma2_out
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# N=4
# M=6    
# activation=OriginMnnActivation()
# rmnn = RMNN(N, M, activation).to(device)
# mu = torch.randn(N)
# C = torch.randn(N,N); C = C@C.t() + 1e-3*torch.eye(N)
# mu_ff = torch.randn(M)
# C_ff = torch.randn(M,M); C_ff = C_ff@C_ff.t() + 1e-3*torch.eye(M)

# mu_out, C_out = rmnn(mu, C, mu_ff, C_ff)
# print(mu_out)
# print(C_out)
