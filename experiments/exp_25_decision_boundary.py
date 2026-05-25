"""
exp_25_decision_boundary.py
---------------------------
Thực nghiệm 2.5 – Ranh giới quyết định (Decision Boundary)

Yêu cầu đồ án:
  - Plot dữ liệu 2D, vẽ Decision Boundary tại T = 1, 5, 20, 100
  - Minh họa cách mô hình dần thích nghi với dữ liệu phức tạp
  - Nhúng code sinh đồ thị vào báo cáo LaTeX bằng lstlisting

Nội dung:
  - Dùng cùng dataset với 2.3 và 2.4 (make_moons, seed=42)
  - Train AdaBoost T=100 vòng (1 lần duy nhất)
  - Tại T = 1, 5, 20, 100 dùng model.predict(X, up_to=T) để vẽ boundary
    bằng kỹ thuật meshgrid → contourf
  - Mỗi subplot ghi rõ: T, train error, test error tại mốc đó
  - Xuất PDF vào figures/decision_boundary.pdf
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from src.adaboost import AdaBoost

# ── Hằng số ──────────────────────────────────────────────────────────
RANDOM_STATE  = 42
N_SAMPLES     = 400
NOISE         = 0.25
TEST_SIZE     = 0.3
N_ESTIMATORS  = 100
T_CHECKPOINTS = [1, 5, 20, 100]    # các mốc T cần vẽ boundary
MESH_STEP     = 0.02                # độ phân giải meshgrid (nhỏ = chính xác hơn)
OUT_PATH      = os.path.join(os.path.dirname(__file__),
                              'figures', 'decision_boundary.pdf')


def generate_data() -> tuple:
    """Sinh dataset make_moons, nhãn {-1, +1}."""
    X, y_raw = make_moons(n_samples=N_SAMPLES,
                          noise=NOISE,
                          random_state=RANDOM_STATE)
    y = np.where(y_raw == 0, -1, 1)
    return train_test_split(X, y,
                            test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)


def make_meshgrid(X: np.ndarray,
                  step: float = MESH_STEP,
                  margin: float = 0.5) -> tuple:
    """
    Tạo meshgrid bao phủ toàn bộ vùng dữ liệu.

    Parameters
    ----------
    X      : (n, 2) - dữ liệu 2D
    step   : độ phân giải lưới
    margin : lề mở rộng quanh dữ liệu

    Returns
    -------
    xx, yy : meshgrid arrays
    """
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    return xx, yy


def compute_errors_at(model   : AdaBoost,
                      X_tr    : np.ndarray,
                      y_tr    : np.ndarray,
                      X_te    : np.ndarray,
                      y_te    : np.ndarray,
                      up_to   : int) -> tuple[float, float]:
    """
    Tính train error và test error tại mốc T = up_to.

    Returns
    -------
    train_err, test_err : float
    """
    train_err = float(np.mean(model.predict(X_tr, up_to=up_to) != y_tr))
    test_err  = float(np.mean(model.predict(X_te, up_to=up_to) != y_te))
    return train_err, test_err


def plot_decision_boundaries(model        : AdaBoost,
                              X_tr        : np.ndarray,
                              y_tr        : np.ndarray,
                              X_te        : np.ndarray,
                              y_te        : np.ndarray,
                              out_path    : str) -> None:
    """
    Vẽ 4 subplot decision boundary tại các mốc T.

    Parameters
    ----------
    model    : AdaBoost đã fit
    X_tr, y_tr : tập train
    X_te, y_te : tập test
    out_path   : đường dẫn xuất PDF
    """
    # Màu sắc
    cmap_bg     = mcolors.ListedColormap(['#FFAAAA', '#AAAAFF'])  # nền 2 vùng
    cmap_points = mcolors.ListedColormap(['#CC0000', '#0000CC'])  # điểm dữ liệu

    X_all = np.vstack([X_tr, X_te])
    xx, yy = make_meshgrid(X_all)
    grid   = np.c_[xx.ravel(), yy.ravel()]   # (N_grid, 2)

    fig, axes = plt.subplots(1, len(T_CHECKPOINTS),
                             figsize=(16, 4), sharey=True)
    fig.suptitle(
        'Hình 2.5 – Ranh giới quyết định của AdaBoost tại các mốc $T$\n'
        r'(Decision Stump, make\_moons, noise=0.25)',
        fontsize=12
    )

    for ax, T in zip(axes, T_CHECKPOINTS):
        # ── Dự đoán trên meshgrid ──
        Z = model.predict(grid, up_to=T).reshape(xx.shape)   # (H, W)

        # ── Vùng màu nền (decision region) ──
        ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)

        # ── Ranh giới quyết định (đường contour Z=0) ──
        ax.contour(xx, yy, Z, levels=[0],
                   colors='black', linewidths=1.5)

        # ── Scatter điểm train ──
        ax.scatter(X_tr[:, 0], X_tr[:, 1],
                   c=y_tr, cmap=cmap_points,
                   s=25, edgecolors='white', linewidths=0.4,
                   label='Train', zorder=3)

        # ── Scatter điểm test (hình vuông để phân biệt) ──
        ax.scatter(X_te[:, 0], X_te[:, 1],
                   c=y_te, cmap=cmap_points,
                   s=30, marker='s',
                   edgecolors='black', linewidths=0.4,
                   alpha=0.6, label='Test', zorder=3)

        # ── Tính và hiển thị error ──
        tr_err, te_err = compute_errors_at(model,
                                           X_tr, y_tr,
                                           X_te, y_te,
                                           up_to=T)
        ax.set_title(f'$T = {T}$\n'
                     f'Train err = {tr_err*100:.1f}%  |  '
                     f'Test err = {te_err*100:.1f}%',
                     fontsize=10)
        ax.set_xlabel('$x_1$', fontsize=10)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('$x_2$', fontsize=10)
    axes[0].legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f'[2.5] Đã xuất hình → {out_path}')
    plt.close()


def main() -> None:
    # 1. Sinh data
    X_tr, X_te, y_tr, y_te = generate_data()
    print(f'[2.5] Train: {X_tr.shape[0]} mẫu | Test: {X_te.shape[0]} mẫu')

    # 2. Train AdaBoost 1 lần đến T = max(T_CHECKPOINTS)
    #    Dùng up_to để vẽ boundary tại T nhỏ hơn, không cần train lại
    model = AdaBoost(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    print(f'[2.5] Đã train {N_ESTIMATORS} vòng lặp')

    # 3. In error tại từng mốc T
    print(f'\n[2.5] Error tại từng mốc T:')
    print(f'{"T":>6} | {"Train Error":>12} | {"Test Error":>12}')
    print('-' * 36)
    for T in T_CHECKPOINTS:
        tr, te = compute_errors_at(model, X_tr, y_tr, X_te, y_te, up_to=T)
        print(f'{T:>6} | {tr*100:>11.1f}% | {te*100:>11.1f}%')

    # 4. Vẽ
    plot_decision_boundaries(model, X_tr, y_tr, X_te, y_te, OUT_PATH)


if __name__ == '__main__':
    main()
