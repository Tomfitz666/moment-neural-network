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
        #    grad_mu = torch.zeros_like(mu_out)
        # if grad_sigma2 is None:
        #    grad_sigma2 = torch.zeros_like(sigma2_out)
        # if grad_chi is None:
        #    grad_chi = torch.zeros_like(chi)


        # numpy backward
        u_np = ubar.detach().cpu().numpy()
        s_np = np.sqrt(sigma2_bar.detach().cpu().numpy())
        mu_np = mu_out.detach().cpu().numpy()
        s_out_np = np.sqrt(sigma2_out.detach().cpu().numpy())
        chi_np = chi.detach().cpu().numpy()

        (
            dmu_dubar, dmu_dsbar,
            ds_dubar, ds_dsbar,
            dchi_dubar, dchi_dsbar
        ) = mcf.fast_backward(
            u_np, s_np, mu_np, s_out_np, chi_np
        )

        # to torch
        dmu_dubar = torch.tensor(dmu_dubar, device=device, dtype=dtype)
        dmu_dsbar = torch.tensor(dmu_dsbar, device=device, dtype=dtype)
        ds_dubar = torch.tensor(ds_dubar, device=device, dtype=dtype)
        ds_dsbar = torch.tensor(ds_dsbar, device=device, dtype=dtype)
        dchi_dubar = torch.tensor(dchi_dubar, device=device, dtype=dtype)
        dchi_dsbar = torch.tensor(dchi_dsbar, device=device, dtype=dtype)

        # chain rule 
        # grad_ubar = (
        #     grad_mu * dmu_dubar
        #     + grad_sigma2 * (2 * torch.sqrt(sigma2_bar) * ds_dubar)
        #     + grad_chi * dchi_dubar
        # )

        # grad_sigma2_bar = (
        #     grad_mu * dmu_dsbar
        #     + grad_sigma2 * (2 * torch.sqrt(sigma2_bar) * ds_dsbar)
        #     + grad_chi * dchi_dsbar
        # ) * (0.5 / torch.sqrt(sigma2_bar))

        s_bar = torch.sqrt(sigma2_bar)
        s_out = torch.sqrt(sigma2_out)
        
        



        grad_ubar = (
            grad_mu * dmu_dubar
            + grad_sigma2 * (2 * s_out * ds_dubar)
            + grad_chi * dchi_dubar
        )

        grad_sbar = (
            grad_mu * dmu_dsbar
            + grad_sigma2 * (2 * s_out * ds_dsbar)
            + grad_chi * dchi_dsbar
          )
        inv_sbar = torch.clamp(1.0 / s_bar, max=1e4)
        grad_sigma2_bar = grad_sbar * 0.5 * inv_sbar
        #grad_sigma2_bar = grad_sbar * (0.5 / s_bar)

       

        return grad_ubar, grad_sigma2_bar



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
