"""
exp_23_error_curve.py
---------------------
Thực nghiệm 2.3 – Sự hội tụ của hàm lỗi (Error Convergence)

Yêu cầu đồ án:
  - Tự sinh tập dữ liệu make_moons hoặc vòng tròn nhiễu
  - Vẽ Line chart thể hiện Training Error và Test Error giảm dần qua T vòng lặp
  - Đồ thị phải có nhãn trục, tiêu đề và caption rõ ràng

Nội dung:
  - Sinh dataset make_moons (n=400, noise=0.25)
  - Train AdaBoost T=300 vòng
  - Thu thập train_errors_ (có sẵn trong model) và tính test_errors
    bằng staged_predict() generator
  - Vẽ thêm theory_bounds_ (chặn lý thuyết Π Z_t) để kiểm chứng
  - Xuất PDF vào figures/error_curve.pdf
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from src.adaboost import AdaBoost

# ── Hằng số ──────────────────────────────────────────────────────────
RANDOM_STATE  = 42
N_SAMPLES     = 400
NOISE         = 0.25
TEST_SIZE     = 0.3
N_ESTIMATORS  = 300
OUT_PATH      = os.path.join(os.path.dirname(__file__), 'figures', 'error_curve.pdf')


def generate_data() -> tuple:
    """Sinh dataset make_moons và chia train/test."""
    X, y_raw = make_moons(n_samples=N_SAMPLES,
                          noise=NOISE,
                          random_state=RANDOM_STATE)
    # AdaBoost yêu cầu nhãn {-1, +1}
    y = np.where(y_raw == 0, -1, 1)
    return train_test_split(X, y,
                            test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)


def compute_errors(model: AdaBoost,
                   X_te: np.ndarray,
                   y_te: np.ndarray) -> list[float]:
    """
    Tính test error tại từng vòng T bằng staged_predict().
    Hiệu quả hơn gọi predict(up_to=t) lặp T lần.

    Returns
    -------
    test_errors : list[float] độ dài T
    """
    test_errors = []
    for y_hat in model.staged_predict(X_te):
        err = float(np.mean(y_hat != y_te))
        test_errors.append(err)
    return test_errors


def plot_error_curve(train_errors : list[float],
                     test_errors  : list[float],
                     theory_bounds: list[float],
                     out_path     : str) -> None:
    """
    Vẽ line chart Training Error, Test Error và Theory Bound theo T.

    Parameters
    ----------
    train_errors  : training error tại mỗi vòng T
    test_errors   : test error tại mỗi vòng T
    theory_bounds : chặn lý thuyết Π Z_t tích lũy
    out_path      : đường dẫn xuất PDF
    """
    T = np.arange(1, len(train_errors) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    # ── Đường training error ──
    ax.plot(T, train_errors,
            color='steelblue', linewidth=2,
            label='Training Error')

    # ── Đường test error ──
    ax.plot(T, test_errors,
            color='tomato', linewidth=2,
            label='Test Error')

    # ── Chặn lý thuyết Π Z_t ──
    ax.plot(T, theory_bounds,
            color='forestgreen', linewidth=1.5,
            linestyle='--', label=r'Chặn lý thuyết $\prod_t Z_t$')

    # ── Đánh dấu điểm training error chạm 0 ──
    zero_idx = next((i for i, e in enumerate(train_errors) if e == 0.0), None)
    if zero_idx is not None:
        ax.axvline(x=zero_idx + 1,
                   color='steelblue', linestyle=':',
                   linewidth=1.2, alpha=0.7)
        ax.annotate(f'Train error = 0\n(T = {zero_idx + 1})',
                    xy=(zero_idx + 1, 0),
                    xytext=(zero_idx + 20, 0.08),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='steelblue'),
                    color='steelblue')

    # ── Trục và nhãn ──
    ax.set_xlabel('Số vòng lặp $T$', fontsize=13)
    ax.set_ylabel('Error', fontsize=13)
    ax.set_title('Hình 2.3 – Sự hội tụ của Training Error và Test Error\n'
                 r'(AdaBoost + Decision Stump, make\_moons, noise=0.25)',
                 fontsize=12)
    ax.set_xlim(1, len(train_errors))
    # ax.set_ylim(-0.01, max(max(test_errors), max(theory_bounds)) + 0.05)
    ax.set_ylim(-0.01, 0.25)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f'[2.3] Đã xuất hình → {out_path}')
    plt.close()


def main() -> None:
    # 1. Sinh data
    X_tr, X_te, y_tr, y_te = generate_data()
    print(f'[2.3] Train: {X_tr.shape[0]} mẫu | Test: {X_te.shape[0]} mẫu')

    # 2. Train AdaBoost
    model = AdaBoost(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    print(f'[2.3] Đã train {N_ESTIMATORS} vòng lặp')

    # 3. Thu thập errors
    train_errors  = model.train_errors_          # có sẵn trong model
    theory_bounds = model.theory_bounds_         # có sẵn trong model
    test_errors   = compute_errors(model, X_te, y_te)

    # 4. In tóm tắt
    final_train = train_errors[-1]
    final_test  = test_errors[-1]
    print(f'[2.3] Train Error cuối: {final_train:.4f} | '
          f'Test Error cuối: {final_test:.4f}')

    # 5. Vẽ
    plot_error_curve(train_errors, test_errors, theory_bounds, OUT_PATH)


if __name__ == '__main__':
    main()
