"""
exp_margin_hist.py
---------------------
Thực nghiệm: Kiểm chứng Margin Theory (Histogram Margin)

Yêu cầu:
  - Theo dõi phân bố Margin tại T = 10, 50, 100
  - Vẽ Histogram của Margin
  - Nhận xét: Margin có thực sự tăng khi T tăng không?
    (Nếu khớp lý thuyết, cần nhấn mạnh trong báo cáo)

Nội dung:
  - Dùng cùng dataset (make_moons, seed=42)
  - Tại mỗi mốc T = 10, 50, 100, tính margin của từng điểm train:
        ρ(x_i) = y_i · f(x_i) / Σ|α_t|
    Dùng model.decision_function(X, up_to=T) và model.alphas_[:T]
  - Vẽ 4 subplot histogram (T=1 thêm để thấy rõ sự thay đổi từ đầu)
  - Đánh dấu đường margin = 0, vẽ đường mean margin
  - Xuất hình vào figures/margin_histogram.pdf
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from src.adaboost import AdaBoost

# ── Hằng số ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_SAMPLES    = 400
NOISE        = 0.25
TEST_SIZE    = 0.3
N_ESTIMATORS = 100
T_CHECKPOINTS = [1, 10, 50, 100]   # các mốc T cần vẽ histogram
N_BINS       = 30
OUT_PATH     = os.path.join(os.path.dirname(__file__),
                             'figures', 'margin_histogram.pdf')


def generate_data() -> tuple:
    """Sinh dataset make_moons, nhãn {-1, +1}."""
    X, y_raw = make_moons(n_samples=N_SAMPLES,
                          noise=NOISE,
                          random_state=RANDOM_STATE)
    y = np.where(y_raw == 0, -1, 1)
    return train_test_split(X, y,
                            test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)


def compute_margins(model    : AdaBoost,
                    X        : np.ndarray,
                    y        : np.ndarray,
                    up_to    : int) -> np.ndarray:
    """
    Tính margin chuẩn hóa tại mốc T = up_to:
        ρ(x_i) = y_i · f(x_i) / Σ_{t=1}^{up_to} |α_t|

    Parameters
    ----------
    model  : AdaBoost đã fit
    X      : (n, d)
    y      : (n,) nhãn {-1, +1}
    up_to  : dùng bao nhiêu weak learner đầu

    Returns
    -------
    margins : (n,) float ∈ [-1, +1]
    """
    f    = model.decision_function(X, up_to=up_to)          # (n,)
    norm = np.sum(np.abs(model.alphas_[:up_to])) + 1e-12    # scalar
    return (y * f) / norm


def plot_margin_histograms(margins_dict : dict[int, np.ndarray],
                           out_path     : str) -> None:
    """
    Vẽ 4 subplot histogram phân bố margin tại các mốc T.

    Parameters
    ----------
    margins_dict : {T: margins array}
    out_path     : đường dẫn xuất PDF
    """
    fig, axes = plt.subplots(1, len(T_CHECKPOINTS),
                             figsize=(14, 4), sharey=True)
    fig.suptitle(
        'Hình 2.4 – Phân bố Margin tại các mốc $T$ khác nhau\n'
        r'(AdaBoost + Decision Stump, make\_moons, noise=0.25)',
        fontsize=12
    )

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    for ax, T, color in zip(axes, T_CHECKPOINTS, colors):
        margins = margins_dict[T]

        # ── Histogram ──
        ax.hist(margins, bins=N_BINS,
                color=color, alpha=0.75, edgecolor='white',
                linewidth=0.5)

        # ── Đường margin = 0 ──
        ax.axvline(x=0, color='black', linewidth=1.8,
                   linestyle='--', label='Margin = 0')

        # ── Đường mean margin ──
        mean_m = float(np.mean(margins))
        ax.axvline(x=mean_m, color='black', linewidth=1.2,
                   linestyle=':', label=f'Mean = {mean_m:.2f}')

        # ── Tỉ lệ điểm margin > 0 ──
        pct_correct = float(np.mean(margins > 0)) * 100

        # ── Nhãn ──
        ax.set_title(f'$T = {T}$\n'
                     f'Mean margin = {mean_m:.3f}\n'
                     f'{pct_correct:.1f}% điểm margin > 0',
                     fontsize=10)
        ax.set_xlabel('Margin $\\rho_f(x_i)$', fontsize=10)
        ax.set_xlim(-1.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Số lượng điểm', fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f'[2.4] Đã xuất hình → {out_path}')
    plt.close()


def print_summary(margins_dict: dict[int, np.ndarray]) -> None:
    """In bảng tóm tắt để nhận xét trong báo cáo."""
    print('\n[2.4] Tóm tắt phân bố Margin:')
    print(f'{"T":>6} | {"Mean":>8} | {"Min":>8} | {"Max":>8} | {"% > 0":>8}')
    print('-' * 50)
    for T, margins in margins_dict.items():
        print(f'{T:>6} | {np.mean(margins):>8.4f} | '
              f'{np.min(margins):>8.4f} | '
              f'{np.max(margins):>8.4f} | '
              f'{np.mean(margins > 0)*100:>7.1f}%')


def main() -> None:
    # 1. Sinh data
    X_tr, X_te, y_tr, y_te = generate_data()
    print(f'[2.4] Train: {X_tr.shape[0]} mẫu | Test: {X_te.shape[0]} mẫu')

    # 2. Train AdaBoost đến T = max(T_CHECKPOINTS)
    model = AdaBoost(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    print(f'[2.4] Đã train {N_ESTIMATORS} vòng lặp')

    # 3. Tính margin tại từng mốc T
    #    Dùng up_to parameter → chỉ train 1 lần, không cần train lại
    margins_dict = {}
    for T in T_CHECKPOINTS:
        margins_dict[T] = compute_margins(model, X_tr, y_tr, up_to=T)
    print(f'[2.4] Đã tính margin tại T = {T_CHECKPOINTS}')

    # 4. In tóm tắt
    print_summary(margins_dict)

    # 5. Vẽ
    plot_margin_histograms(margins_dict, OUT_PATH)


if __name__ == '__main__':
    main()
