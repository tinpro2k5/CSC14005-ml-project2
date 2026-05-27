# Hướng dẫn chạy Mã nguồn (Code) - Đồ án 2: Boosting

Thư mục này chứa toàn bộ mã nguồn Python cài đặt thuật toán **AdaBoost** (from scratch) và các scripts chạy thực nghiệm để minh hoạ và kiểm chứng lý thuyết.

## 1. Môi trường cài đặt (Requirements)

Phiên bản Python khuyến nghị: **Python 3.10+**

Các thư viện cần thiết được liệt kê trong `requirements.txt`:
- `numpy`: Dùng cho phép toán ma trận và cài đặt thuật toán cốt lõi.
- `matplotlib`: Dùng để vẽ các biểu đồ minh hoạ trực quan.
- `scikit-learn`: **CHỈ** được dùng để gọi hàm `make_moons` nhằm tự sinh dữ liệu cho thực nghiệm.

**Lệnh cài đặt:**
```bash
# Optional: Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Trên Linux/macOS
venv\Scripts\activate           # Trên Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

## 2. Mô tả cấu trúc và chức năng từng file

```text
code/
├── requirements.txt                # Danh sách thư viện
├── README.md
│
├── src/                            # Chứa mã nguồn cài đặt thuật toán
│   ├── base.py                     # Abstract class `BaseWeakLearner` định nghĩa interface chung.
│   ├── decision_stump.py           # Cài đặt thuật toán Cây quyết định 1 mức từ đầu.
│   └── adaboost.py                 # Cài đặt vòng lặp Boosting.
│
└── experiments/                    # Chứa các scripts chạy thực nghiệm
    ├── exp_error_curve.py          # Kiểm chứng sự hội tụ của Training Error và Theory Bound.
    ├── exp_margin_hist.py          # Vẽ Histogram phân bố Margin tại các mốc T khác nhau.
    ├── exp_decision_boundary.py    # Vẽ decision boundary.
    └── figures/                    # Chứa các file biểu đồ kết quả.
```

## 3. Hướng dẫn chạy code thực nghiệm

Tất cả các file thực nghiệm đều được thiết kế độc lập và tự động sinh dữ liệu ảo (make_moons), huấn luyện mô hình, và xuất biểu đồ ra định dạng `.pdf` lưu vào thư mục `experiments/figures/`.

Có thể đứng ở thư mục gốc (thư mục chứa file README này) và chạy lần lượt các lệnh sau:

**Thực nghiệm 1: Sự hội tụ của Training Error**
```bash
python experiments/exp_error_curve.py
```
> Kiểm chứng lý thuyết rằng AdaBoost tối thiểu hoá upper bound cho Training Error.

**Thực nghiệm 2: Phân bố Margin**
```bash
python experiments/exp_margin_hist.py
```
> Histogram để minh hoạ Margin Theory: phân bố margin của các điểm huấn luyện khi T tăng.

**Thực nghiệm 3: Ranh giới quyết định**
```bash
python experiments/exp_decision_boundary.py
```
>Vẽ ranh giới phân lớp tại T = 1, 5, 20, 100 để thấy rõ cách AdaBoost kết hợp nhiều Decision Stump lại thành một ranh giới phức tạp.

Sau khi chạy xong, toàn bộ hình vẽ sẽ nằm sẵn trong `experiments/figures/`.
