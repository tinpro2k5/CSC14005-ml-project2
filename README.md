# AdaBoost - From Scratch

**Môn:** Nhập môn Học Máy (CSC14005) - Đồ Án 2  
**Chương:** Boosting  
---

## Cấu trúc thư mục


---


## Cài đặt & Sử dụng

```bash
# (Optional) Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Cài thư viện
pip install -r requirements.txt
```

### Import trong code

```python
from src.base import BaseWeakLearner
from src.decision_stump import DecisionStump
from src.adaboost import AdaBoost

# Khởi tạo và huấn luyện
model = AdaBoost(n_estimators=100, weak_learner_cls=DecisionStump)
model.fit(X, y)

# Dự đoán
y_pred = model.predict(X_test)
```

---

## 2.1 – Base Learner: Decision Stump

### `base.py`
Abstract class `BaseWeakLearner` định nghĩa interface chung:
- `fit(X, y, sample_weight) → self` – Huấn luyện với phân bố trọng số D_t
- `predict(X) → np.ndarray ∈ {-1, +1}` – Dự đoán nhãn
- property `error` → weighted error ε_t

### `decision_stump.py`
`DecisionStump` kế thừa `BaseWeakLearner` và cài đặt cây quyết định 1 mức (depth-1):

- `fit()` duyệt qua tất cả features `j = 0…d-1`
- Với mỗi feature, tìm threshold ứng viên (midpoints của giá trị phân biệt)
- Duyệt polarity (+1/-1) để tìm rule có **weighted error nhỏ nhất**
- Lưu `self.error` để AdaBoost dùng tính α_t

**Từng bước:**
```
min_err = Inf
for j in features:
  for thresh in thresholds:
    for polarity in (+1, -1):
      preds = (X[:, j] < thresh) ? 1 : -1
      if polarity == -1: preds = -preds
      err = sum(D * (preds != y))
      if err < min_err:
        min_err = err
        best_j, best_thresh, best_polarity = j, thresh, polarity
```

## 2.2 – AdaBoost Core

### `adaboost.py`
`AdaBoost` cài đặt vòng lặp T bước chuẩn:

| Bước | Công thức |
|---|---|
| Khởi tạo | $D_1(i) = 1/n$ |
| Huấn luyện | $h_t \sim D_t$ (gọi `weak_learner.fit`) |
| Weighted error | $\varepsilon_t = \sum_i D_t(i) \cdot \mathbf{1}[h_t(x_i) \neq y_i]$ |
| Learner weight | $\alpha_t = \frac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}$ |
| Normaliser | $Z_t = 2\sqrt{\varepsilon_t(1-\varepsilon_t)}$ |
| Update D | $D_{t+1}(i) = D_t(i)\cdot e^{-\alpha_t y_i h_t(x_i)} / Z_t$ |

- Cập nhật D **vectorized** (không for-loop trên samples)
- Training error tính bằng `f_accum` (cộng dồn) - không recompute từ đầu
- Lưu `theory_bounds_` = $\prod Z_t$ tích luỹ cho kiểm chứng lý thuyết (sẽ dùng trong 2.3)
- `decision_function()` dùng `alphas @ preds` (matrix multiply) thay for-loop
- `staged_predict()` là generator cộng dồn, hiệu quả hơn gọi `predict(up_to=t)` lặp T lần

---

