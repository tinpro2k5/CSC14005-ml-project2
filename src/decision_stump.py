"""
decision_stump.py
-----------------
Decision Stump - weak learner mặc định cho AdaBoost.
Cài đặt from scratch, không dùng scikit-learn.

Vectorization:
  Với mỗi feature j, toàn bộ m threshold ứng viên được tính cùng lúc
  bằng broadcasting (n x m matrix) thay vì nested for-loop.
"""

import numpy as np
from .base import BaseWeakLearner
from typing import Optional

class DecisionStump(BaseWeakLearner):
    """
    Decision Stump: cây quyết định sâu 1 mức (depth-1 tree).

    Với mỗi feature j và ngưỡng θ, stump dự đoán:
        polarity=+1 :  h(x) = +1 nếu x_j < θ,  ngược lại -1
        polarity=-1 :  h(x) = -1 nếu x_j < θ,  ngược lại +1

    Thuật toán fit() tìm bộ (j*, θ*, p*) tối thiểu hoá:
        ε = Σ_i  D(i) · 1[h(x_i) ≠ y_i]

    Attributes (sau fit)
    --------------------
    feature_idx : chỉ số đặc trưng được chọn
    threshold   : ngưỡng phân loại
    polarity    : +1 hoặc -1 (hướng bất đẳng thức)
    _error      : weighted error (float [0, 1])
    """

    def __init__(self) -> None:
        self.feature_idx: Optional[int] = None
        self.threshold:   Optional[float] = None
        self.polarity:    Optional[int] = None 
        self._error:      Optional[float] = None   

    # Property error - expose cho AdaBoost
    @property
    def error(self) -> float:
        """Weighted error. Chỉ hợp lệ sau khi fit() được gọi."""
        if self._error is None:
            raise RuntimeError("DecisionStump chưa được fit().")
        return self._error


    # Dự đoán nội bộ (vectorized)
    @staticmethod
    def _apply_rule(X_col: np.ndarray,
                    threshold: float,
                    polarity: int) -> np.ndarray:
        """
        Trả về nhãn  {-1, +1} cho một cột đặc trưng và một ngưỡng.
        Parameters
        ----------
        X_col     : (n,) - giá trị đặc trưng j của n điểm
        threshold : float
        polarity  : +1 → h = +1 khi x < θ; -1 → h = -1 khi x < θ

        Returns
        -------
        preds : (n,) float array in {-1.0, +1.0}
        """
        pred = np.where(X_col < threshold, 1.0, -1.0)
        return pred if polarity == 1 else -pred


    # fit  
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: np.ndarray) -> "DecisionStump":
        """
        Tìm decision stump tối ưu bằng cách duyệt qua toàn bộ features
        và thresholds ứng viên.

        Với mỗi feature j:
          - Xây dựng tập threshold = midpoints giữa các giá trị phân biệt
            + hai biên ngoài (classify-all-one-class).
          - Vectorize: preds_pos (n x m) = X_col[:, None] < thresholds[None, :]
            → sai số có trọng số cho tất cả m threshold cùng lúc.
          - Chọn polarity=+1 hay -1 bằng cách lấy min(err, 1 - err).

        Parameters
        ----------
        X             : (n, d) - ma trận đặc trưng
        y             : (n,)   - nhãn ∈ {-1, +1}
        sample_weight : (n,)   - phân bố D_t, Σ = 1

        Returns
        -------
        self
        """
        n, d = X.shape
        best_error = np.inf

        # y và D cần shape (n,) cho broadcast
        D = sample_weight  # alias

        for j in range(d):
            col = X[:, j]   # (n,)

            # --- Thresholds ứng viên -----------------------------------
            uniq = np.unique(col)                      # tăng dần
            if len(uniq) == 1:
                # Feature hằng → bỏ qua (không phân biệt được)
                continue
            mids = (uniq[:-1] + uniq[1:]) / 2.0       # (m-1,) midpoints
            thresholds = np.concatenate([
                [uniq[0]  - 1e-9],   # classify-all-positive
                mids,
                [uniq[-1] + 1e-9],   # classify-all-negative
            ])                                          # shape (m,)

            # --- Vectorize trên thresholds ----------------------------
            # col[:, None] so sánh broadcast với thresholds[None, :]
            # preds_pos[i, k] = +1 nếu col[i] < thresholds[k]
            preds_pos = np.where(
                col[:, None] < thresholds[None, :],    # (n, m) bool
                1.0, -1.0
            )
            # mỗi cột k của preds_pos là dự đoán của stump với threshold k và polarity=+1                                         


            # Sai số polarity=+1 : ε_k = Σ_i D_i · 1[preds_pos[i,k] ≠ y_i]
            wrong_pos = (preds_pos != y[:, None]).astype(np.float64)  # (n, m)
            errors_pos = D @ wrong_pos                  # (m,)  dot product

            # polarity=-1 là flip → sai số = 1 - sai số polarity=+1
            errors_neg = 1.0 - errors_pos               # (m,)

            # Chọn polarity tốt nhất cho mỗi threshold
            errors_best = np.minimum(errors_pos, errors_neg)  # (m,)
            best_k = int(np.argmin(errors_best))

            if errors_best[best_k] < best_error:
                best_error       = errors_best[best_k]
                self.feature_idx = j
                self.threshold   = float(thresholds[best_k])
                # polarity: +1 nếu errors_pos thấp hơn, -1 nếu ngược lại
                self.polarity    = (1
                                    if errors_pos[best_k] <= errors_neg[best_k]
                                    else -1)

        self._error = float(best_error)
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Áp dụng rule đã học lên tập X.

        Parameters
        ----------
        X : (n, d)

        Returns
        -------
        y_hat : (n,) float array in {-1.0, +1.0}
        """
        if self.feature_idx is None:
            raise RuntimeError("DecisionStump chưa được fit().")
        return self._apply_rule(
            X[:, self.feature_idx], self.threshold, self.polarity
        )