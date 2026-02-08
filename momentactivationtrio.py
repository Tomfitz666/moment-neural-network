import torch
import torch.nn as nn
import numpy as np
from mnn.mnn_core.mnn_utils import Mnn_Core_Func
import matplotlib.pyplot as plt
mcf=Mnn_Core_Func()

class MnnActivateTrio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ubar, sigma2_bar):
        device, dtype = ubar.device, ubar.dtype
        EPS = 1e-7

        # ---- numpy forward ----
        u_np = ubar.detach().cpu().numpy()
        # 防止 sqrt(0)
        sigma2_bar_safe = sigma2_bar.clamp_min(EPS)
        s_np = np.sqrt(sigma2_bar_safe.detach().cpu().numpy())

        mu_out_np, s_out_np, chi_np = mcf.fast_forward(u_np, s_np)

        # ---- back to torch ----
        mu_out = torch.tensor(mu_out_np, device=device, dtype=dtype)
        s_out_t = torch.tensor(s_out_np, device=device, dtype=dtype)
        sigma2_out = s_out_t ** 2
        chi = torch.tensor(chi_np, device=device, dtype=dtype)

        # save for backward（保存 safe 版本）
        ctx.save_for_backward(
            ubar, sigma2_bar_safe,
            mu_out, s_out_t, chi
        )
        ctx.eps = EPS

        return mu_out, sigma2_out, chi

    @staticmethod
    def backward(ctx, grad_mu, grad_sigma2, grad_chi):
        device, dtype = grad_mu.device, grad_mu.dtype
        EPS = ctx.eps

        ubar, sigma2_bar, mu_out, s_out_t, chi = ctx.saved_tensors

        # 处理 None 梯度
        if grad_mu is None:
            grad_mu = torch.zeros_like(mu_out)
        if grad_sigma2 is None:
            grad_sigma2 = torch.zeros_like(mu_out)
        if grad_chi is None:
            grad_chi = torch.zeros_like(chi)

        # safe sqrt
        s_bar = torch.sqrt(sigma2_bar)  # 已经 clamp 过，>= sqrt(EPS)
        s_out = s_out_t.clamp_min(EPS)  # 防止 s_out=0

        # numpy backward
        u_np = ubar.detach().cpu().numpy()
        s_np = s_bar.detach().cpu().numpy()
        mu_np = mu_out.detach().cpu().numpy()
        s_out_np = s_out.detach().cpu().numpy()
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

        # ============================================
        # 链式法则
        # ============================================
        # fast_forward:  (ubar, sbar) -> (mu_out, s_out, chi)
        # 我们需要:      (ubar, sigma2_bar) -> (mu_out, sigma2_out, chi)
        #
        # 其中: sbar = sqrt(sigma2_bar)
        #        sigma2_out = s_out^2
        #
        # 所以:
        #   d(sigma2_out)/d(ubar)      = 2 * s_out * ds_dubar
        #   d(sigma2_out)/d(sigma2_bar) = 2 * s_out * ds_dsbar * d(sbar)/d(sigma2_bar)
        #                                = 2 * s_out * ds_dsbar * 0.5 / sbar
        #                                = s_out * ds_dsbar / sbar
        #
        #   d(mu_out)/d(sigma2_bar)     = dmu_dsbar * 0.5 / sbar
        #   d(chi)/d(sigma2_bar)        = dchi_dsbar * 0.5 / sbar
        # ============================================

        inv_2sbar = 0.5 / s_bar  # s_bar >= sqrt(EPS), 安全

        # --- grad w.r.t. ubar ---
        # L → mu_out → ubar, L → sigma2_out → ubar, L → chi → ubar
        grad_ubar = (
            grad_mu * dmu_dubar
            + grad_sigma2 * (2.0 * s_out * ds_dubar)
            + grad_chi * dchi_dubar
        )

        # --- grad w.r.t. sigma2_bar ---
        # 每条路径都要乘 d(sbar)/d(sigma2_bar) = 0.5/sbar
        

        grad_sigma2_bar = (
            grad_mu * dmu_dsbar * inv_2sbar
            + grad_sigma2 * (s_out / s_bar) * ds_dsbar  # = 2*s_out*ds_dsbar * 0.5/sbar
            + grad_chi * dchi_dsbar * inv_2sbar
        )

        # clamp 防止极端值
        grad_ubar = torch.clamp(grad_ubar, -1e4, 1e4)
        grad_sigma2_bar = torch.clamp(grad_sigma2_bar, -1e4, 1e4)

        # NaN 安全网
        grad_ubar = torch.nan_to_num(grad_ubar, nan=0.0, posinf=1e4, neginf=-1e4)
        grad_sigma2_bar = torch.nan_to_num(grad_sigma2_bar, nan=0.0, posinf=1e4, neginf=-1e4)

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

# print("Numerical dmu/dubar =", numerical_dmu_dubar(ubar, sigma2_bar))


if __name__ == "__main__":
    ubar = torch.tensor([0.05], requires_grad=True)
    sbar = torch.tensor([0.7], requires_grad=True)
    sigma2_bar = sbar ** 2

    # ---- 解析梯度 ----
    mu, sigma2_out, chi = MnnActivateTrio.apply(ubar, sigma2_bar)
    mu.backward()
    analytic = ubar.grad.item()
    print(f"解析梯度 (backward): {analytic:.12f}")

    # ---- 多 eps 数值梯度 ----
    eps_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,1e-8 ]
    numerical = []
    abs_errors = []
    rel_errors = []

    print(f"\n{'eps':>12s}  {'数值梯度':>16s}  {'绝对误差':>14s}  {'相对误差':>14s}  状态")
    print("-" * 75)

    for eps in eps_list:
        with torch.no_grad():
            mu_p = MnnActivateTrio.apply(ubar + eps, sigma2_bar)[0]
            mu_m = MnnActivateTrio.apply(ubar - eps, sigma2_bar)[0]
        num = ((mu_p - mu_m) / (2 * eps)).item()

        abs_err = abs(num - analytic)
        rel_err = abs_err / (abs(analytic) + 1e-15)

        numerical.append(num)
        abs_errors.append(abs_err)
        rel_errors.append(rel_err)

        status = "✅" if rel_err < 0.01 else ("⚠️" if rel_err < 0.1 else "❌")
        print(f"{eps:>12.0e}  {num:>16.12f}  {abs_err:>14.2e}  {rel_err:>14.2e}  {status}")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1: 数值梯度 vs eps
    ax = axes[0]
    ax.semilogx(eps_list, numerical, 'bo-', markersize=8, linewidth=2, label='Numerical')
    ax.axhline(y=analytic, color='r', linestyle='--', linewidth=2, label=f'Analytic = {analytic:.8f}')
    ax.set_xlabel('eps', fontsize=13)
    ax.set_ylabel('d(mu)/d(ubar)', fontsize=13)
    ax.set_title('Numerical Gradient vs eps', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # 图2: 误差 vs eps (log-log)
    ax = axes[1]
    ax.loglog(eps_list, abs_errors, 'rs-', markersize=8, linewidth=2, label='Absolute Error')
    ax.loglog(eps_list, rel_errors, 'g^-', markersize=8, linewidth=2, label='Relative Error')
    ax.axhline(y=0.01, color='gray', linestyle=':', alpha=0.7, label='1% threshold')
    best_idx = np.argmin(abs_errors)
    ax.axvline(x=eps_list[best_idx], color='blue', linestyle=':', alpha=0.7,
               label=f'Best eps = {eps_list[best_idx]:.0e}')
    ax.set_xlabel('eps', fontsize=13)
    ax.set_ylabel('Error', fontsize=13)
    ax.set_title('Error vs eps', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # 图3: 柱状图对比
    ax = axes[2]
    selected = [0, 2, 3, 4, 6]  # 选几个代表性的 eps
    x_pos = np.arange(len(selected))
    width = 0.35

    ax.bar(x_pos - width/2, [analytic] * len(selected), width,
           color='#2196F3', alpha=0.8, label='Analytic')
    ax.bar(x_pos + width/2, [numerical[i] for i in selected], width,
           color='#FF5722', alpha=0.8, label='Numerical')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{eps_list[i]:.0e}' for i in selected], fontsize=10)
    ax.set_xlabel('eps', fontsize=13)
    ax.set_ylabel('Gradient', fontsize=13)
    ax.set_title('Analytic vs Numerical', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(f'ubar={0.4}, sbar={0.7}, sigma2_bar={0.49:.2f}', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('gradient_verification.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ 最佳 eps = {eps_list[best_idx]:.0e}, 最小绝对误差 = {abs_errors[best_idx]:.2e}")