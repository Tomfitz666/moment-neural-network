
import torch
import torch.nn as nn
import numpy as np
from mnn.mnn_core.mnn_utils import Mnn_Core_Func
mcf=Mnn_Core_Func()

class MnnActivateTrio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ubar, sigma2_bar):
        device, dtype = ubar.device, ubar.dtype

        # ---- numpy forward ----
        u_np = ubar.detach().cpu().numpy()
        s_np = np.sqrt(sigma2_bar.detach().cpu().numpy())

        mu_out_np, s_out_np, chi_np = mcf.fast_forward(u_np, s_np)

        # ---- back to torch ----
        mu_out = torch.tensor(mu_out_np, device=device, dtype=dtype)
        sigma2_out = torch.tensor(s_out_np ** 2, device=device, dtype=dtype)
        chi = torch.tensor(chi_np, device=device, dtype=dtype)

        # save for backward
        ctx.save_for_backward(
            ubar, sigma2_bar,
            mu_out, sigma2_out, chi
        )

        return mu_out, sigma2_out, chi



    @staticmethod
    def backward(ctx, grad_mu, grad_sigma2, grad_chi):
        device, dtype = grad_mu.device, grad_mu.dtype
        ubar, sigma2_bar, mu_out, sigma2_out, chi = ctx.saved_tensors

        # if grad_mu is None:
        #     grad_mu = torch.zeros_like(mu_out)
        # if grad_sigma2 is None:
        #     grad_sigma2 = torch.zeros_like(sigma2_out)
        # if grad_chi is None:
        #     grad_chi = torch.zeros_like(chi)

        eps = 1e-10
        

        mu_out_safe = torch.where(
            torch.abs(mu_out) < eps,
            torch.sign(mu_out + eps) * eps,  
            mu_out
        )
        
        
        sigma2_bar_safe = torch.clamp(sigma2_bar, min=eps)
        sigma2_out_safe = torch.clamp(sigma2_out, min=eps)
        
        
        chi_safe = torch.clamp(torch.abs(chi), min=eps) * torch.sign(chi + eps)

        # numpy backward 
        u_np = ubar.detach().cpu().numpy()
        s_np = np.sqrt(sigma2_bar_safe.detach().cpu().numpy())
        mu_np = mu_out_safe.detach().cpu().numpy()  
        s_out_np = np.sqrt(sigma2_out_safe.detach().cpu().numpy())
        chi_np = chi_safe.detach().cpu().numpy()   

        (
            dmu_dubar, dmu_dsbar,
            dsout_dubar, dsout_dsbar,
            dchi_dubar, dchi_dsbar
        ) = mcf.fast_backward(u_np, s_np, mu_np, s_out_np, chi_np)

        # to torch
        dmu_dubar = torch.tensor(dmu_dubar, device=device, dtype=dtype)
        dmu_dsbar = torch.tensor(dmu_dsbar, device=device, dtype=dtype)
        dsout_dubar = torch.tensor(dsout_dubar, device=device, dtype=dtype)
        dsout_dsbar = torch.tensor(dsout_dsbar, device=device, dtype=dtype)
        dchi_dubar = torch.tensor(dchi_dubar, device=device, dtype=dtype)
        dchi_dsbar = torch.tensor(dchi_dsbar, device=device, dtype=dtype)
        
    
        if torch.isnan(dmu_dubar).any():
            print(f"Warning: NaN in dmu_dubar after fast_backward")
            dmu_dubar = torch.nan_to_num(dmu_dubar, nan=0.0)
        
        

        s_bar = torch.sqrt(sigma2_bar_safe)
        s_out = torch.sqrt(sigma2_out_safe)
        
        grad_s_out = grad_sigma2 * 2.0 * s_out

        grad_ubar = (
            grad_mu * dmu_dubar 
            + grad_s_out * dsout_dubar
            + grad_chi * dchi_dubar
        )

        grad_sbar = (
            grad_mu * dmu_dsbar 
            + grad_s_out * dsout_dsbar
            + grad_chi * dchi_dsbar
        )

        grad_sigma2_bar = grad_sbar * (0.5 / s_bar)
        
        # clip
        grad_ubar = torch.clamp(grad_ubar, -1e3, 1e3)
        grad_sigma2_bar = torch.clamp(grad_sigma2_bar, -1e3, 1e3)

        return grad_ubar, grad_sigma2_bar







    # @staticmethod
    # def backward(ctx, grad_mu, grad_sigma2, grad_chi):
    #     device, dtype = grad_mu.device, grad_mu.dtype

    #     ubar, sigma2_bar, mu_out, sigma2_out, chi = ctx.saved_tensors


        


    #     # numpy backward
    #     u_np = ubar.detach().cpu().numpy()
    #     s_np = np.sqrt(sigma2_bar.detach().cpu().numpy())
    #     mu_np = mu_out.detach().cpu().numpy()
    #     s_out_np = np.sqrt(sigma2_out.detach().cpu().numpy())
    #     chi_np = chi.detach().cpu().numpy()

    #     (
    #         dmu_dubar, dmu_dsbar,
    #         ds_dubar, ds_dsbar,
    #         dchi_dubar, dchi_dsbar
    #     ) = mcf.fast_backward(
    #         u_np, s_np, mu_np, s_out_np, chi_np
    #     )

    #     # to torch
    #     dmu_dubar = torch.tensor(dmu_dubar, device=device, dtype=dtype)
    #     dmu_dsbar = torch.tensor(dmu_dsbar, device=device, dtype=dtype)
    #     ds_dubar = torch.tensor(ds_dubar, device=device, dtype=dtype)
    #     ds_dsbar = torch.tensor(ds_dsbar, device=device, dtype=dtype)
    #     dchi_dubar = torch.tensor(dchi_dubar, device=device, dtype=dtype)
    #     dchi_dsbar = torch.tensor(dchi_dsbar, device=device, dtype=dtype)

       

    #     s_bar = torch.sqrt(sigma2_bar+1e-10)
    #     s_out = torch.sqrt(sigma2_out+1e-10)
        
        

       

    #     grad_ubar = (
    #         grad_mu * dmu_dubar
    #         + grad_sigma2 * (2 * s_out * ds_dubar)
    #         + grad_chi * dchi_dubar
    #     )

    #     grad_sbar = (
    #         grad_mu * dmu_dsbar
    #         + grad_sigma2 * (2 * s_out * ds_dsbar)
    #         + grad_chi * dchi_dsbar
    #       )

    #     grad_sigma2_bar = grad_sbar * (0.5 / s_bar)

       

    #     return grad_ubar, grad_sigma2_bar



# ubar = torch.tensor(0.4, requires_grad=True)
# sbar = torch.tensor(0.7, requires_grad=True)

# sigma2_bar = sbar ** 2   # 

# mu, sigma2_out, chi = MnnActivateTrio.apply(ubar, sigma2_bar)
# loss = mu
# loss.backward()

# print("ubar.grad =", ubar.grad)



# def numerical_dmu_dubar(ubar, sigma2_bar, eps=1e-5):
#     with torch.no_grad():
#         mu_p, _, _ = MnnActivateTrio.apply(
#             ubar + eps, sigma2_bar
#         )
#         mu_m, _, _ = MnnActivateTrio.apply(
#             ubar - eps , sigma2_bar
#         )
#     return ((mu_p - mu_m) / (2*eps)).item()

#print("Numerical dmu/dubar =", numerical_dmu_dubar(ubar, sigma2_bar))
