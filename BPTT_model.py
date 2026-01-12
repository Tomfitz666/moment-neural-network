import torch
import torch.nn as nn
from RMNN_model import RMNN_VarOnly


class RMNN_BPTT(nn.Module):
    def __init__(self, rmnn, T):
        super().__init__()
        self.rmnn = rmnn
        self.T = T

    def forward(self, mu0, C0):
        """
        mu0: (B, N)
        C0 : (B, N, N)
        """
        B, N = mu0.shape
        device = mu0.device

        mu = mu0
        C  = C0

        mu_ff = torch.zeros(B, self.rmnn.N, device=device)
        C_ff  = torch.eye(self.rmnn.N, device=device).unsqueeze(0).repeat(B,1,1) * 1e-6

        for _ in range(self.T):
            mu, C = self.rmnn(mu, C, mu_ff, C_ff)
            mu_ff = mu
            C_ff  = C

        return mu, C
    

class RMNN_BPTT_NEW(nn.Module):
    def __init__(self, rmnn: RMNN_VarOnly, T: int):
        super().__init__()
        self.rmnn = rmnn
        self.T = T

    def forward(self, mu0, var0, mu_ff, var_ff):
        mu, var = mu0, var0

        for _ in range(self.T):
            mu, var = self.rmnn(mu, var, mu_ff, var_ff)

        return mu, var