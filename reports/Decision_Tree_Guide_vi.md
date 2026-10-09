# Phần 3.6 — Decision Tree: chọn độ phức tạp và giải thích một dự đoán

Nhóm 14 — DS66B: Trần Quang Huy, Bùi Ngọc Thuấn, Nguyễn Hoàng Thành.

## 1. Mục đích

Decision Tree trả lời bằng chuỗi câu hỏi dạng “giá trị này có nhỏ hơn ngưỡng không?”. Một mẫu đi qua các nhánh đến một lá. Dự đoán ở lá là trung bình các đáp án của những mẫu train ở lá đó.

Ta thử cây để kiểm tra giả thuyết: các quy tắc ngưỡng và quan hệ không tuyến tính có dự đoán điện trước 30 phút tốt hơn công thức Linear không? Đây là mô hình đối chiếu thông thường của chương 9, không phải đóng góp mới của nhóm.

## 2. Mở, chạy và vị trí tệp

Mở thư mục sau bằng VS Code:

```text
D:\Kỳ I 2026-2027\Học thống kê\Midterm_Tetouan_GridWatch
```

```text
src/gridwatch/tree_tuning.py          Chọn độ sâu và số mẫu mỗi lá
tests/test_tree_tuning_contract.py    Kiểm tra chọn cấu hình và phép tính lá
run_tree.ps1                          Chạy hoặc chỉ hiện kết quả đã lưu
reports/tree_complexity_search.csv    Kết quả của 12 cấu hình
reports/tree_comparison.csv           So sánh Linear, cây cũ, cây đã chọn
reports/tree_real_examples.csv        Hai dự đoán thật của từng mô hình
reports/tree_selected_path.csv        Các điều kiện mẫu minh họa đi qua
reports/tree_selected_leaf_answers.csv  20 đáp án train thuộc lá minh họa
reports/figures/tree_complexity_search.png  Biểu đồ chọn cấu hình
artifacts/tree_tuning_summary.json    Tóm tắt có thể kiểm tra lại
```

Trong PowerShell, tại thư mục dự án:

```powershell
.\run_tree.ps1 -ShowResults
```

Lệnh này chỉ hiện bảng đã lưu. Muốn chạy lại toàn bộ quá trình:

```powershell
.\run_tree.ps1
```

## 3. Cây ban đầu đã dùng gì?

Trong `src/gridwatch/experiment.py`, dòng 33, cấu hình benchmark ban đầu là:

```python
"Decision Tree": DecisionTreeRegressor(
    max_depth=12, min_samples_leaf=20, random_state=42
),
```

- `max_depth=12`: cây được chia tối đa 12 tầng.
- `min_samples_leaf=20`: mỗi lá phải có tối thiểu 20 mẫu train; điều này làm dự đoán bớt phụ thuộc một nhóm quá nhỏ.
- `random_state=42`: cố định ngẫu nhiên để chạy lại nhận cùng kết quả.

Cây không cần chuẩn hóa như KNN, vì mỗi lần chia chỉ so sánh một cột với một ngưỡng. Cấu hình mặc định dùng `squared_error`: cây chọn phép chia giảm sai số bình phương nhiều nhất; ở lá, dự đoán là trung bình. MAE vẫn là chỉ số nhóm dùng để so sánh dự đoán cuối cùng.

## 4. Tìm tham số: làm thế nào?

Trong `src/gridwatch/tree_tuning.py`, dòng 21:

```python
DEPTHS = [4, 8, 12, 16]
LEAF_SIZES = [10, 20, 50]
```

Tổng cộng 12 cấu hình. Độ sâu lớn giúp cây học quy tắc chi tiết hơn nhưng dễ bám sát train. Lá lớn làm dự đoán ổn định hơn nhưng có thể bỏ mất khác biệt thật. Không có cấu hình luôn tốt cho mọi bài toán.

Code thật tại dòng 35:

```python
search = GridSearchCV(
    make_tree(),
    param_grid={"max_depth": DEPTHS, "min_samples_leaf": LEAF_SIZES},
    scoring=SCORER, cv=folds, refit=True, n_jobs=1, error_score="raise",
)
search.fit(X_train, y_train)
```

`folds` là ba lượt thời gian nằm **bên trong train**. Mỗi lượt chỉ học quá khứ và chấm điểm trên đoạn thời gian tiếp sau, có chừa ba dòng ở ranh giới vì nhãn cần dự đoán trước 30 phút. `SCORER` là MAE sau khi chặn dự đoán âm. `refit=True` học lại cấu hình thắng bằng toàn bộ 35.680 mẫu train. Validation ngoài và test không được đưa vào lệnh chọn tham số.

## 5. Kết quả tìm độ phức tạp

| Độ sâu tối đa | Mẫu tối thiểu/lá | MAE trung bình ba lượt train |
| ---: | ---: | ---: |
| 16 | 20 | **1.278,70** |
| 12 | 10 | 1.292,25 |
| 16 | 10 | 1.293,33 |
| 12 | 20 | 1.296,85 |
| 4 | 20 | 1.798,88 |

Các cấu hình còn lại nằm trong `tree_complexity_search.csv`. Cấu hình đã chọn là **độ sâu 16, tối thiểu 20 mẫu/lá**. Đây là tốt nhất trong 12 cấu hình đã thử với cách chia thời gian này; không gọi nó là tốt nhất tuyệt đối cho mọi tham số có thể có.

![Biểu đồ chọn cấu hình cây](Tree_Complexity_Search.png)

Mỗi đường giữ cố định độ sâu và thay đổi số mẫu tối thiểu trong lá. Chấm cam là MAE thấp nhất trong bảng. Biểu đồ minh họa kết quả; số MAE là quy tắc quyết định, không chọn bằng mắt.

## 6. Kết quả validation: có nên dùng cây không?

| Mô hình | Độ sâu | Mẫu/lá | MAE validation |
| --- | ---: | ---: | ---: |
| Linear | — | — | **318,99** |
| Cây ban đầu | 12 | 20 | 1.543,15 |
| Cây đã chọn | 16 | 20 | 1.542,96 |

Cây đã chọn giảm MAE khoảng 0,19 so với cấu hình ban đầu, một thay đổi rất nhỏ. Nó vẫn kém Linear rất nhiều. Kết luận đúng là: nhóm đã thử mô hình ngưỡng phi tuyến và đã chọn độ phức tạp theo thời gian, nhưng bằng chứng hiện tại không ủng hộ dùng cây thay Linear.

Điểm chọn tham số của cây là 1.278,70 trên ba lượt nằm trong train, trong khi validation phía sau là 1.542,96. Đây là lý do không lấy điểm dùng để chọn tham số thay cho kết quả validation. Sự khác nhau có thể phản ánh thay đổi theo giai đoạn thời gian, không đủ để tự kết luận nguyên nhân duy nhất là overfitting.

## 7. Một dự đoán thật đi qua cây

Dùng dữ liệu **12/09/2017 lúc 19:10**, dự đoán mức điện lúc **19:40**. Cây đã chọn có độ sâu thực tế 16 và 1.360 lá. Mẫu này đi qua bảy điều kiện rồi đến lá số 1.759:

| Câu hỏi | Giá trị mẫu | Ngưỡng | Hướng đi |
| --- | ---: | ---: | --- |
| Khu vực 3 ở cùng thời điểm ngày trước ≤ 21.817,01? | 21.798,57 | 21.817,01 | Trái |
| Khu vực 1 hiện tại ≤ 28.423,05? | 47.322,48 | 28.423,05 | Phải |
| Khu vực 1 hiện tại ≤ 33.376,92? | 47.322,48 | 33.376,92 | Phải |
| Khu vực 3 hiện tại ≤ 18.570,64? | 21.869,17 | 18.570,64 | Phải |
| Khu vực 1 ngày trước ≤ 36.413,59? | 46.704,42 | 36.413,59 | Phải |
| Khu vực 1 hiện tại ≤ 41.864,49? | 47.322,48 | 41.864,49 | Phải |
| Khu vực 1 ngày trước ≤ 42.852,74? | 46.704,42 | 42.852,74 | Phải |

Lá cuối có đúng 20 mẫu train. Code tại dòng 65 lấy các đáp án thực của 20 mẫu này, tính trung bình cho từng khu vực và kiểm tra trung bình đó khớp `model.predict`.

| Khu vực | Dự đoán cây | Thực tế lúc 19:40 | Sai số tuyệt đối |
| --- | ---: | ---: | ---: |
| 1 | 45.065,95 | 47.679,29 | 2.613,35 |
| 2 | 26.791,75 | 28.444,49 | 1.652,74 |
| 3 | 21.462,32 | 21.763,27 | 300,94 |

Ví dụ khu vực 1:

```text
sai số = abs(47.679,29204 − 45.065,9469025) ≈ 2.613,35
```

Sai số trung bình của ba khu vực tại riêng thời điểm này là 1.522,34. Nó khác MAE validation 1.542,96 vì MAE validation lấy trung bình trên 7.859 thời điểm × 3 khu vực, không chỉ một thời điểm.

## 8. Cách báo cáo trước lớp

“Nhóm em thử Decision Tree để kiểm tra các quy tắc ngưỡng phi tuyến. Nhóm không giữ nguyên một cấu hình tùy ý mà thử 12 mức độ phức tạp trên ba lượt theo thời gian trong train. Cấu hình tốt nhất là sâu 16, mỗi lá ít nhất 20 mẫu. Đây là biểu đồ kết quả. Sau khi học lại trên toàn bộ train, nhóm kiểm tra validation riêng: MAE của cây là 1.542,96, còn Linear là 318,99, nên hiện tại nhóm vẫn chọn Linear. Đây là một mẫu thật: cây đi qua bảy điều kiện, đến một lá có 20 mẫu train và lấy trung bình đáp án của lá để ra dự đoán.”

Nếu thầy hỏi “cây có overfit không?”, trả lời: “Nhóm đã giới hạn độ sâu và số mẫu mỗi lá, rồi chọn bằng lượt thời gian chưa dùng để học trong từng lượt. Tuy nhiên, cây vẫn có sai số validation cao. Nhóm không khẳng định nguyên nhân duy nhất là overfitting; kết quả chỉ cho thấy cây chưa phù hợp bằng Linear trong thí nghiệm hiện tại.”

## 9. Kiểm tra đã làm

Chín kiểm tra tự động của dự án đã đạt. Hai kiểm tra mới cho cây xác nhận: cấu hình được ghi nhận có MAE nhỏ nhất trong bảng; và trung bình đáp án của lá khớp với dự đoán cây. Bảng 20 mẫu của lá, đường đi và hai ví dụ thật được lưu riêng để kiểm tra lại.

Nguồn phương pháp: [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
