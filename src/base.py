"""
adaboost/base.py
----------------
Abstract interface cho weak learner.
base learner (DecisionStump, ShallowTree, ...) đều kế thừa BaseWeakLearner và implement fit() + predict().
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseWeakLearner(ABC):
    """
    Subclass implement:
        fit(X, y, sample_weight)  → self
        predict(X)                → np.ndarray of {-1, +1}
    Subclass expose:
        self.error  : weighted error sau khi fit() (float)
    """

    @abstractmethod
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: np.ndarray) -> "BaseWeakLearner":
        """
        Huấn luyện weak learner với phân bố trọng số D_t.

        Parameters
        ----------
        X             : (n, d) - ma trận đặc trưng
        y             : (n,)   - nhãn thuộc {-1, +1}
        sample_weight : (n,)   - D_t, Σ = 1

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Trả về nhãn dự đoán {-1, +1} cho tập X.

        Parameters
        ----------
        X : (n, d)

        Returns
        -------
        y_hat : (n,) - array of {-1.0, +1.0}
        """

    @property
    def error(self) -> float:
        """Weighted error phải được set trong fit()."""
        raise NotImplementedError(
            f"{self.__class__.__name__} chưa expose thuộc tính `error`."
        )