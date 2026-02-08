
import torch
import torch.nn as nn
import numpy as np
from mnn.mnn_core.mnn_utils import Mnn_Core_Func
from momentactivationtrio import MnnActivateTrio
import matplotlib.pyplot as plt



def numerical_grad_ubar(ubar, sigma2_bar, eps):
    """中心差分法: d(mu_out)/d(ubar)"""
    with torch.no_grad():
        mu_p = MnnActivateTrio.apply(ubar + eps, sigma2_bar)[0]
        mu_m = MnnActivateTrio.apply(ubar - eps, sigma2_bar)[0]
    return ((mu_p - mu_m) / (2 * eps)).item()


def analytic_grad_ubar(ubar, sigma2_bar):
    """backward(1,0,0) 对 ubar 的梯度"""
    u = ubar.clone().detach().requires_grad_(True)
    s2 = sigma2_bar.clone().detach().requires_grad_(True)
    mu, sigma2_out, chi = MnnActivateTrio.apply(u, s2)
    torch.autograd.backward(
        [mu, sigma2_out, chi],
        [torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0)]
    )
    return u.grad.item()


if __name__ == "__main__":
    # ============================================================
    # 测试点
    # ============================================================
    test_points = [
        (0.4, 0.7, "正常值"),
        (2.0, 0.3, "大均值小方差"),
        (-1.0, 1.0, "负均值"),
        (0.01, 0.5, "接近零均值"),
        (0.5, 0.01, "极小方差"),
    ]

    eps_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    # ============================================================
    # 计算数据
    # ============================================================
    results = {}
    for ubar_val, sbar_val, desc in test_points:
        ubar = torch.tensor(ubar_val)
        sigma2_bar = torch.tensor(sbar_val ** 2)

        grad_analytic = analytic_grad_ubar(ubar, sigma2_bar)

        abs_errors = []
        rel_errors = []
        num_grads = []

        for eps in eps_list:
            num_grad = numerical_grad_ubar(ubar, sigma2_bar, eps)
            abs_err = abs(num_grad - grad_analytic)
            rel_err = abs_err / (abs(grad_analytic) + 1e-15)
            abs_errors.append(abs_err)
            rel_errors.append(rel_err)
            num_grads.append(num_grad)

        results[desc] = {
            'analytic': grad_analytic,
            'num_grads': num_grads,
            'abs_errors': abs_errors,
            'rel_errors': rel_errors,
            'ubar': ubar_val,
            'sbar': sbar_val,
        }

    # ============================================================
    # 打印表格
    # ============================================================
    print("=" * 85)
    print("实验1: d(mu)/d(ubar)  — backward(1,0,0) vs 数值梯度")
    print("=" * 85)

    for desc, r in results.items():
        print(f"\n--- {desc}: ubar={r['ubar']}, sbar={r['sbar']} ---")
        print(f"  解析梯度: {r['analytic']:.10f}")
        print(f"  {'eps':>12s}  {'数值梯度':>14s}  {'绝对误差':>14s}  {'相对误差':>14s}  状态")
        print(f"  {'-'*68}")
        for i, eps in enumerate(eps_list):
            rel = r['rel_errors'][i]
            status = "✅" if rel < 0.01 else ("⚠️" if rel < 0.1 else "❌")
            print(f"  {eps:>12.0e}  {r['num_grads'][i]:>14.10f}  "
                  f"{r['abs_errors'][i]:>14.2e}  {rel:>14.2e}  {status}")

    # ============================================================
    # 可视化
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    # ---- 图1: 所有测试点的相对误差 vs eps ----
    ax = axes[0]
    for i, (desc, r) in enumerate(results.items()):
        ax.loglog(eps_list, r['rel_errors'], 'o-', color=colors[i],
                  label=f"{desc}", markersize=6, linewidth=2)
    ax.set_xlabel('eps', fontsize=12)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title('Relative Error vs eps (all test points)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='1% threshold')

    # ---- 图2: 所有测试点的绝对误差 vs eps ----
    ax = axes[1]
    for i, (desc, r) in enumerate(results.items()):
        ax.loglog(eps_list, r['abs_errors'], 's-', color=colors[i],
                  label=f"{desc}", markersize=6, linewidth=2)
    ax.set_xlabel('eps', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Absolute Error vs eps (all test points)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 图3: 解析 vs 数值梯度 (最佳eps) ----
    ax = axes[2]
    best_eps_idx = 3  # 1e-5
    analytic_vals = [r['analytic'] for r in results.values()]
    numeric_vals = [r['num_grads'][best_eps_idx] for r in results.values()]
    descs = list(results.keys())

    x_pos = np.arange(len(descs))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, analytic_vals, width, label='Analytic (backward)',
                   color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, numeric_vals, width, label=f'Numerical (eps=1e-5)',
                   color='#FF5722', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(descs, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Gradient Value', fontsize=12)
    ax.set_title('Analytic vs Numerical Gradient', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # ---- 图4-5: 每个测试点单独的收敛曲线 ----
    for idx, (desc, r) in enumerate(results.items()):
        if idx >= 2:
            break
        ax = axes[3 + idx]

        ax.loglog(eps_list, r['abs_errors'], 'o-', color='#2196F3',
                  markersize=8, linewidth=2, label='Absolute Error')
        ax.loglog(eps_list, r['rel_errors'], 's-', color='#FF5722',
                  markersize=8, linewidth=2, label='Relative Error')

        # 标注最优 eps
        best_idx = np.argmin(r['abs_errors'])
        ax.axvline(x=eps_list[best_idx], color='green', linestyle='--',
                   alpha=0.7, label=f'Best eps={eps_list[best_idx]:.0e}')

        ax.set_xlabel('eps', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title(f'{desc}\n(analytic={r["analytic"]:.6f})', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ---- 图6: 误差热力图 ----
    ax = axes[5]
    error_matrix = np.array([r['rel_errors'] for r in results.values()])
    error_matrix_log = np.log10(error_matrix + 1e-16)

    im = ax.imshow(error_matrix_log, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(len(eps_list)))
    ax.set_xticklabels([f'{e:.0e}' for e in eps_list], rotation=45, fontsize=8)
    ax.set_yticks(range(len(descs)))
    ax.set_yticklabels(descs, fontsize=9)
    ax.set_xlabel('eps', fontsize=12)
    ax.set_title('log10(Relative Error) Heatmap', fontsize=13)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(rel error)', fontsize=10)

    # 在格子里标数值
    for i in range(len(descs)):
        for j in range(len(eps_list)):
            val = error_matrix[i, j]
            color = 'white' if error_matrix_log[i, j] > -2 else 'black'
            ax.text(j, i, f'{val:.1e}', ha='center', va='center',
                    fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig('gradient_verification.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ 图片已保存为 gradient_verification.png")