import torch
import torch.nn as nn



class RMNN_BPTT(nn.Module):
    """
    Unroll RMNN for T steps.
    (mu_t, C_t) from previous step are fed into the next step.
    """

    def __init__(self, rmnn, T):
        super().__init__()
        self.rmnn = rmnn   # RMNN module
        self.T = T

    def forward(self, mu0, C0, mu_ff, C_ff):
        """
        mu0  : (B, N)
        C0   : (B, N, N)
        mu_ff: (B, M)
        C_ff : (B, M, M) or None

        returns
        -------
        mu_T : (B, N)
        C_T  : (B, N, N)
        """

        mu_t = mu0
        C_t  = C0

        for _ in range(self.T):
            mu_t, C_t = self.rmnn(mu_t, C_t, mu_ff, C_ff)

        return mu_t, C_t
    


# class RMNN_BPTT(nn.Module):
#     def __init__(self, rmnn, T):
#         super().__init__()
#         self.rmnn = rmnn
#         self.T = T

#     def forward(self, mu0, C0):
#         """
#         mu0: (B, N)
#         C0 : (B, N, N)
#         """
#         B, N = mu0.shape
#         device = mu0.device

#         mu = mu0
#         C  = C0

#         mu_ff = torch.zeros(B, self.rmnn.N, device=device)
#         C_ff  = torch.eye(self.rmnn.N, device=device).unsqueeze(0).repeat(B,1,1) * 1e-6

#         for _ in range(self.T):
#             mu, C = self.rmnn(mu, C, mu_ff, C_ff)
#             mu_ff = mu
#             C_ff  = C

#         return mu, C
    
# 
# class RMNN_BPTT_NEW(nn.Module):
#     def __init__(self, rmnn: RMNN_VarOnly, T: int):
#         super().__init__()
#         self.rmnn = rmnn
#         self.T = T

#     def forward(self, mu0, var0, mu_ff, var_ff):
#         mu, var = mu0, var0

#         for _ in range(self.T):
#             mu, var = self.rmnn(mu, var, mu_ff, var_ff)

#         return mu, var