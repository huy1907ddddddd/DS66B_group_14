# Phần 3.7 — Random Forest: kiểm tra nhiều cây và kết luận đối chiếu

Nhóm 14 — DS66B: Trần Quang Huy, Bùi Ngọc Thuấn, Nguyễn Hoàng Thành.

## 1. Phần này làm gì?

Random Forest là nhiều Decision Tree cùng dự đoán, rồi lấy trung bình kết quả của chúng. Ý tưởng là một cây đơn lẻ có thể phụ thuộc mạnh vào các lần chia dữ liệu; trung bình nhiều cây có thể ổn định hơn.

Nhóm dùng Random Forest để kiểm tra: khi giảm phụ thuộc vào một cây, dự đoán điện trước 30 phút có tốt hơn Linear hoặc cây đơn lẻ không? Đây là phương pháp phổ biến trong chương 9, không phải một thuật toán mới do nhóm tạo ra.

## 2. Mở và chạy ở đâu?

Mở dự án bằng VS Code:

```text
D:\Kỳ I 2026-2027\Học thống kê\Midterm_Tetouan_GridWatch
```

```text
src/gridwatch/forest_tuning.py       Code tìm cấu trúc và giải thích rừng
tests/test_forest_tuning_contract.py  Kiểm tra cách chọn và phép lấy trung bình
run_forest.ps1                       Lệnh chạy phần Random Forest
reports/forest_structure_search.csv  Bảng sáu cấu hình đã thử
reports/forest_comparison.csv        So sánh Linear, rừng cũ và rừng chọn
reports/forest_real_examples.csv     Hai dự đoán thật của từng mô hình
reports/forest_selected_tree_predictions.csv  Dự đoán của đủ 60 cây
reports/forest_selected_prediction_summary.csv  Min, max, độ lệch chuẩn và trung bình
reports/forest_feature_importance.csv  Điểm quan trọng đặc trưng
reports/figures/forest_structure_search.png  Biểu đồ chọn cấu trúc
artifacts/forest_tuning_summary.json  Tóm tắt cách chạy
```

Trong PowerShell, từ thư mục dự án:

```powershell
.\run_forest.ps1 -ShowResults
```

Lệnh này chỉ hiện kết quả đã lưu. Muốn chạy lại đầy đủ:

```powershell
.\run_forest.ps1
```

## 3. Code thật và ý nghĩa tham số

`src/gridwatch/forest_tuning.py`, dòng 21:

```python
DEPTHS = [8, 14]
LEAF_SIZES = [5, 20, 50]
N_TREES = 60
```

Tổng cộng có 2 × 3 = 6 cấu trúc. Mỗi cấu trúc giữ 60 cây để đối chiếu công bằng với benchmark cũ. Chúng ta không đồng thời thử quá nhiều tham số vì mỗi cấu hình phải học 60 cây qua ba lượt thời gian; mở rộng quá mức sẽ tăng nguy cơ “chọn may mắn” và tốn thời gian không cần thiết.

`max_depth` là độ sâu tối đa của từng cây. `min_samples_leaf` là số mẫu train tối thiểu trong một lá. `max_features=.8` nghĩa mỗi lần tìm phép chia, mỗi cây xét một tập con 80% trong 48 đặc trưng. Tập con khác nhau giữa các cây giúp các cây bớt giống nhau. `random_state=42` cố định ngẫu nhiên để chạy lại được.

Code tạo một rừng, dòng 25:

```python
return RandomForestRegressor(
    n_estimators=n_trees,
    max_depth=depth,
    min_samples_leaf=leaf_size,
    max_features=.8,
    n_jobs=2,
    random_state=42,
)
```

`n_estimators=60` là 60 cây. `n_jobs=2` cho hai tác vụ tính toán song song; nó không thay đổi kết quả dự đoán.

## 4. Chọn cấu trúc không dùng validation hoặc test

Code tại dòng 35:

```python
search = GridSearchCV(
    make_forest(n_trees=n_trees),
    param_grid={"max_depth": DEPTHS, "min_samples_leaf": LEAF_SIZES},
    scoring=SCORER, cv=folds, refit=True, n_jobs=1, error_score="raise",
)
search.fit(X_train, y_train)
```

`folds` là ba lượt chia theo thời gian bên trong train: học từ quá khứ, kiểm tra ở đoạn tiếp theo, và chừa ba mẫu ở ranh giới vì bài toán dự đoán 30 phút. `SCORER` là MAE; MAE thấp hơn tốt hơn. `search.fit(X_train, y_train)` không nhận validation ngoài hay test, nên hai phần này không quyết định cấu trúc rừng.

Sau khi chọn, nhóm học lại rừng thắng bằng toàn bộ 35.680 mẫu train, rồi đo MAE trên 7.859 thời điểm validation. Test không được đánh giá lại trong bước này.

## 5. Kết quả thử sáu cấu trúc

| Độ sâu tối đa | Mẫu tối thiểu/lá | Số cây | MAE trung bình ba lượt train |
| ---: | ---: | ---: | ---: |
| 14 | 5 | 60 | **1.000,94** |
| 14 | 20 | 60 | 1.069,92 |
| 8 | 5 | 60 | 1.099,73 |
| 8 | 20 | 60 | 1.122,35 |
| 14 | 50 | 60 | 1.164,69 |
| 8 | 50 | 60 | 1.192,41 |

Cấu hình tốt nhất đã trùng cấu hình benchmark ban đầu: sâu 14, ít nhất 5 mẫu/lá. Điều này là kết quả có ích: nhóm đã kiểm tra, thay vì mặc định con số ban đầu là đúng.

![Biểu đồ chọn cấu trúc Random Forest](Forest_Structure_Search.png)

Mỗi đường cố định độ sâu và đổi số mẫu tối thiểu mỗi lá. Chấm cam là cấu hình có MAE thấp nhất trong bảng. Số MAE quyết định lựa chọn; biểu đồ chỉ giúp nhìn xu hướng.

## 6. So sánh validation và quyết định mô hình

| Mô hình | Độ sâu | Mẫu/lá | Số cây | MAE validation |
| --- | ---: | ---: | ---: | ---: |
| Linear | — | — | — | **318,99** |
| Random Forest ban đầu | 14 | 5 | 60 | 1.223,02 |
| Random Forest đã chọn | 14 | 5 | 60 | 1.223,02 |

Random Forest tốt hơn Decision Tree đơn lẻ (1.543,15), phù hợp với mục đích lấy trung bình nhiều cây. Nhưng nó vẫn kém Linear trong bài toán này. Kết luận đúng là: nhóm giữ Linear làm mô hình dự báo chính; Random Forest là đối chứng phi tuyến đã được thử và có bằng chứng cụ thể.

Chênh lệch giữa MAE chọn cấu trúc trong train (1.000,94) và validation ngoài (1.223,02) cho thấy không được thay validation bằng điểm chọn tham số. Không đủ cơ sở để khẳng định nguyên nhân duy nhất là overfitting; dữ liệu theo thời gian có thể thay đổi giữa các giai đoạn.

## 7. Một mẫu thật: 60 cây tạo dự đoán ra sao?

Lấy dữ liệu lúc **12/09/2017 19:10**, dự đoán lúc **19:40**. Với khu vực 1:

| Đại lượng | Giá trị |
| --- | ---: |
| Số thực tế lúc 19:40 | 47.679,29 |
| Dự đoán trung bình của rừng | 45.817,67 |
| Sai số tuyệt đối | 1.861,62 |
| Dự đoán nhỏ nhất của một cây | 43.984,99 |
| Dự đoán lớn nhất của một cây | 46.817,84 |
| Độ lệch chuẩn giữa 60 cây | 792,73 |

Quy tắc thật là:

```text
dự đoán rừng = (dự đoán cây 1 + ... + dự đoán cây 60) / 60
```

Code tự kiểm tra trung bình này trùng với `forest.predict`. Bảng `forest_selected_tree_predictions.csv` có đủ 60 con số; không chỉ lấy vài cây đẹp để minh họa. Ví dụ trên là một dự đoán, không phải MAE của cả mô hình.

Sai số vùng 1 tại mẫu này:

```text
abs(47.679,29204 − 45.817,671745) ≈ 1.861,62
```

## 8. Đặc trưng nào được cây dùng nhiều?

Ba điểm quan trọng cao nhất của rừng đang chọn:

| Đặc trưng | Điểm quan trọng |
| --- | ---: |
| Khu vực 3 ở cùng thời điểm ngày trước | 0,5524 |
| Điện khu vực 1 hiện tại | 0,2863 |
| Khu vực 1 ở cùng thời điểm ngày trước | 0,0449 |

Điểm này đo tổng mức giảm độ phân tán mà đặc trưng tạo ra trong các phép chia của 60 cây. Nó **không** chứng minh nguyên nhân vật lý, cũng không cho thấy “nếu đổi đặc trưng này thì điện chắc chắn đổi”. Các đặc trưng tương quan với nhau có thể chia nhau hoặc thay thế nhau trong cây.

## 9. Lời nói khi báo cáo

“Nhóm em dùng Random Forest để kiểm tra liệu trung bình nhiều Decision Tree có ổn định hơn một cây đơn lẻ không. Nhóm thử sáu cấu trúc, giữ 60 cây và chọn bằng ba lượt theo thời gian trong train. Cấu hình ban đầu sâu 14, lá tối thiểu 5 là tốt nhất trong bảng. Random Forest giảm sai số so với một cây đơn lẻ, nhưng MAE validation là 1.223,02, vẫn cao hơn Linear 318,99. Vì vậy nhóm không chọn mô hình phức tạp chỉ vì nó có nhiều cây; nhóm chọn theo kết quả validation.”

Nếu thầy hỏi vì sao không chỉnh tiếp số cây: “Nhóm giữ 60 cây cố định để so sánh công bằng với benchmark. Thử thêm số cây là một thí nghiệm riêng sau này; kết quả hiện tại đã đủ để kết luận cấu hình Random Forest đang xét chưa vượt Linear.”

## 10. Kiểm tra đã làm

Mười một kiểm tra tự động của dự án đã đạt. Hai kiểm tra mới xác nhận cấu hình được ghi nhận có MAE nhỏ nhất trong bảng; và trung bình dự đoán của từng cây bằng dự đoán cuối của Random Forest. Các kiểm tra này không hứa mô hình sẽ luôn tốt trên dữ liệu tương lai.

Nguồn phương pháp: [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
