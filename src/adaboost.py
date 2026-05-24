"""
AdaBoost (Freund & Schapire, 1997) cài đặt from scratch.
Nhận weak learner kế thừa BaseWeakLearner.

Công thức cốt lõi (mỗi vòng t = 1 … T)
---------------------------------------
  Khởi tạo : D_1(i) = 1/n

  1. Huấn luyện h_t với D_t     (gọi weak_learner.fit)
  2. ε_t = Σ_i D_t(i) · 1[h_t(x_i) ≠ y_i]      (weighted error)
  3. α_t = ½ · ln((1 - ε_t) / ε_t)              (learner weight)
  4. Z_t = 2 · sqrt(ε_t · (1 - ε_t))               (hệ số chuẩn hoá)
  5. D_{t+1}(i) = D_t(i) · exp(-α_t · y_i · h_t(x_i)) / Z_t

  Dự đoán : h(x) = sign(Σ_t α_t · h_t(x))

Chặn lý thuyết
---------------------------------------------
  Training error ≤  Π_{t=1}^{T} Z_t  = Π_t 2·√(ε_t·(1-ε_t))

  Tích luỹ : upper_bound_T = Π_{t=1}^{T} Z_t
"""

from __future__ import annotations
from typing import Type
import numpy as np

try:
    from .base import BaseWeakLearner
    from .decision_stump import DecisionStump
except ImportError:
    from src.base import BaseWeakLearner
    from src.decision_stump import DecisionStump


class AdaBoost:
    """
    AdaBoost Classifier.

    Parameters
    ----------
    n_estimators     : số vòng lặp T (mặc định 50)
    weak_learner_cls : class weak learner; phải kế thừa BaseWeakLearner
                       (mặc định DecisionStump)
    random_state     : seed numpy để tái tạo kết quả

    Attributes (sau fit)
    --------------------
    learners_        : list[BaseWeakLearner] - T weak learner đã huấn luyện
    alphas_          : list[float]  - trọng số α_t
    epsilons_        : list[float]  - weighted error ε_t mỗi vòng
    Zs_              : list[float]  - hệ số chuẩn hoá Z_t mỗi vòng
    theory_bounds_   : list[float]  - chặn trên lý thuyết tích luỹ Π Z_t
    train_errors_    : list[float]  - training error thực nghiệm tích luỹ
    """

    def __init__(self,
                 n_estimators: int = 50,
                 weak_learner_cls: Type[BaseWeakLearner] = DecisionStump,
                 random_state: int = 42) -> None:
        if not issubclass(weak_learner_cls, BaseWeakLearner):
            raise TypeError(
                f"{weak_learner_cls} phải kế thừa BaseWeakLearner."
            )
        self.n_estimators     = n_estimators
        self.weak_learner_cls = weak_learner_cls
        self.random_state     = random_state

        # --- Kết quả sau fit ---
        self.learners_      : list[BaseWeakLearner] = []
        self.alphas_        : list[float]           = []
        self.epsilons_      : list[float]           = []
        self.Zs_            : list[float]           = []
        self.theory_bounds_ : list[float]           = []   
        self.train_errors_  : list[float]           = []  

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoost":
        """
        Huấn luyện AdaBoost T vòng.

        Parameters
        ----------
        X : (n, d)  - ma trận đặc trưng
        y : (n,)    - nhãn {-1, +1}

        Returns
        -------
        self
        """
        np.random.seed(self.random_state)
        n = X.shape[0]

        # Reset tất cả trạng thái
        self.learners_.clear()
        self.alphas_.clear()
        self.epsilons_.clear()
        self.Zs_.clear()
        self.theory_bounds_.clear()
        self.train_errors_.clear()

        # --- Bước 0: Khởi tạo phân bố đều D_1 = 1/n ---
        D = np.full(n, 1.0 / n)   # (n,)

        # Tích luỹ chặn lý thuyết Π Z_t (bắt đầu = 1)
        bound_product = 1.0

        # Dùng để tính training error nhanh bằng accumulation
        # f_accum[i] = Σ_{k=1}^{t} α_k · h_k(x_i)
        f_accum = np.zeros(n)

        for t in range(self.n_estimators):

            # ── Bước 1: Huấn luyện weak learner h_t với phân bố D_t ──
            learner = self.weak_learner_cls()
            learner.fit(X, y, sample_weight=D)

            # ── Bước 2: Lấy ε_t, clamp tránh log(0) ──
            eps_t = float(np.clip(learner.error, 1e-10, 1.0 - 1e-10))

            # ── Bước 3: Tính α_t = ½ · ln((1-ε_t)/ε_t) ──
            alpha_t = 0.5 * np.log((1.0 - eps_t) / eps_t)

            # ── Bước 4: Hệ số chuẩn hoá Z_t = 2·√(ε_t·(1-ε_t)) ──
            Z_t = 2.0 * np.sqrt(eps_t * (1.0 - eps_t))

            # ── Bước 5: Cập nhật D_{t+1} (vectorized, không for-loop) ──
            #   D_{t+1}(i) = D_t(i) · exp(-α_t · y_i · h_t(x_i)) / Z_t
            h_t = learner.predict(X)                       # (n,)
            D   = D * np.exp(-alpha_t * y * h_t) / Z_t    # (n,) vectorized

            # Tích luỹ chặn lý thuyết: bound_T = Π_{k=1}^{t+1} Z_k
            bound_product *= Z_t

            # Training error tại vòng t+1 (dùng f_accum để tránh recompute)
            f_accum += alpha_t * h_t
            train_err_t = float(np.mean(np.sign(f_accum) != y))

            # ── Lưu kết quả vòng t ──
            self.learners_.append(learner)
            self.alphas_.append(alpha_t)
            self.epsilons_.append(eps_t)
            self.Zs_.append(Z_t)
            self.theory_bounds_.append(bound_product)
            self.train_errors_.append(train_err_t)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def decision_function(self,
                          X: np.ndarray,
                          up_to: int | None = None) -> np.ndarray:
        """
        Tính f(x) = Σ_{t=1}^{up_to} α_t · h_t(x)  (chưa lấy sign).

        Dùng matrix multiply để tránh for-loop trên samples:
          preds : (k, n) - stack predictions của k learners
          alphas: (k,)
          → alphas @ preds = (n,)

        Parameters
        ----------
        X     : (n, d)
        up_to : dùng bao nhiêu learner đầu (None = tất cả)

        Returns
        -------
        f : (n,) - giá trị thực (chưa sign)
        """
        if not self.learners_:
            raise RuntimeError("AdaBoost chưa được fit().")
        k      = up_to if up_to is not None else len(self.learners_)
        # Stack (k, n): mỗi hàng là dự đoán của một learner
        preds  = np.array([l.predict(X) for l in self.learners_[:k]])  # (k, n)
        alphas = np.array(self.alphas_[:k])                             # (k,)
        return alphas @ preds                                           # (n,)

    def predict(self,
                X: np.ndarray,
                up_to: int | None = None) -> np.ndarray:
        """
        Trả về nhãn  {-1, +1} = sign(f(x)).

        Parameters
        ----------
        X     : (n, d)
        up_to : dùng bao nhiêu weak learner đầu (None = tất cả)

        Returns
        -------
        y_hat : (n,) float array ∈ {-1.0, +1.0}
        """
        return np.sign(self.decision_function(X, up_to=up_to))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy = tỉ lệ phân loại đúng trên tập (X, y)."""
        return float(np.mean(self.predict(X) == y))


    def staged_predict(self, X: np.ndarray):
        """
        Generator: yield y_hat tại từng vòng t = 1 … T.

        Hiệu quả hơn gọi predict(up_to=t) lặp T lần vì chỉ cần
        cộng dồn thay vì recompute từ đầu.

        Yields
        ------
        y_hat : (n,) float array ∈ {-1.0, +1.0}
        """
        f = np.zeros(X.shape[0])
        for learner, alpha in zip(self.learners_, self.alphas_):
            f += alpha * learner.predict(X)
            yield np.sign(f)

    def margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Lề (margin) chuẩn hoá của từng điểm dữ liệu:

            ρ(x_i) = y_i · f(x_i) / Σ_t |α_t|

        Trong đó f(x) = Σ_t α_t · h_t(x).
        Chuẩn hoá bởi Σ|α_t| để margin ∈ [-1, +1].

        ρ > 0 → phân loại đúng;  ρ < 0 → phân loại sai.

        Parameters
        ----------
        X : (n, d)
        y : (n,)  nhãn ∈ {-1, +1}

        Returns
        -------
        margins : (n,) float array ∈ [-1, +1]
        """
        f    = self.decision_function(X)
        norm = np.sum(np.abs(self.alphas_)) + 1e-12
        return (y * f) / norm