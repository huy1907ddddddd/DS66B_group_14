# Phần 3.4 — Chọn alpha cho Ridge và Lasso

Nhóm 14 — DS66B: Trần Quang Huy, Bùi Ngọc Thuấn, Nguyễn Hoàng Thành.

## 1. Nhìn tổng thể trước khi xem code

Chúng ta đang dự đoán điện năng tiêu thụ của ba khu vực trước 30 phút. Phần này trả lời câu hỏi: **Ridge và Lasso nên bị phạt mạnh đến mức nào để dự đoán tốt trên những khoảng thời gian chưa dùng để học?** Alpha chính là mức phạt đó.

Alpha nhỏ cho phép mô hình bám dữ liệu nhiều hơn; alpha lớn kéo các hệ số về gần 0 mạnh hơn. Với Lasso, một số hệ số có thể bằng đúng 0. Mức phạt lớn không tự động cho kết quả tốt hơn.

Đây là cách làm thông thường: thử nhiều alpha, đánh giá trên các lượt dữ liệu theo thời gian, chọn giá trị có sai số trung bình thấp nhất. Điểm cần làm cẩn thận trong bài của nhóm là tránh lấy thông tin tương lai để lựa chọn. Không gọi việc tìm alpha là tính năng đột phá của dự án.

Mở thư mục sau bằng VS Code:

```text
D:\Kỳ I 2026-2027\Học thống kê\Midterm_Tetouan_GridWatch
```

Các tệp cần chỉ khi trình bày:

```text
Midterm_Tetouan_GridWatch/
├── src/gridwatch/tuning.py           Code tìm alpha và so sánh kết quả
├── tests/test_tuning_contract.py     Kiểm tra việc chia và chấm điểm
├── run_tuning.ps1                    Lệnh chạy phần này
├── reports/
│   ├── Alpha_tuning.md               Báo cáo kết quả
│   ├── alpha_search.csv              Sai số của từng alpha
│   ├── alpha_folds.csv               Các khoảng thời gian dùng để kiểm tra
│   ├── alpha_comparison.csv          So sánh cấu hình cũ và mới
│   ├── alpha_real_examples.csv       Dự đoán của hai thời điểm thực tế
│   └── figures/alpha_search.png      Biểu đồ chọn alpha
└── artifacts/alpha_tuning_summary.json  Tóm tắt cách chạy và alpha đã chọn
```

Code tìm alpha được đặt riêng trong `tuning.py` để dễ đọc, chạy lại và đối chiếu với kết quả trước đó. Phần chuẩn bị dữ liệu tiếp tục dùng các hàm đã có trong dự án.

## 2. Chạy gì khi chiếu trước lớp?

Trong VS Code, mở Terminal → New Terminal, chọn PowerShell. Đưa terminal về thư mục dự án rồi hiển thị kết quả đã lưu:

```powershell
Set-Location -LiteralPath 'D:\Kỳ I 2026-2027\Học thống kê\Midterm_Tetouan_GridWatch'
.\run_tuning.ps1 -ShowResults
```

Lệnh này đọc bảng kết quả có sẵn, không huấn luyện lại. Nói rõ điều đó khi trình bày. Để thực sự chạy lại kiểm tra và tìm alpha:

```powershell
.\run_tuning.ps1
```

Trình tự chiếu: bảng kết quả → biểu đồ → code. Không cần chạy tìm kiếm lại giữa lúc báo cáo nếu thầy chỉ yêu cầu trình bày tiến độ; khi thầy yêu cầu chạy lại thì dùng lệnh thứ hai.

## 3. Chúng ta thử alpha như thế nào?

Giữ nguyên cách chia train, validation và test đã có. Trong **35.680 mẫu train**, tạo ba lượt học và kiểm tra theo thời gian:

| Lượt | Số mẫu được học | Số mẫu kiểm tra lượt đó |
| --- | ---: | ---: |
| 1 | 8.917 | 8.920 |
| 2 | 17.837 | 8.920 |
| 3 | 26.757 | 8.920 |

Mỗi lượt chỉ học quá khứ rồi kiểm tra khoảng thời gian tiếp theo. Dữ liệu được học mở rộng dần. Ba mẫu sát ranh giới được chừa ra vì đáp án của một mẫu nằm sau thời điểm dự đoán 30 phút. Ví dụ lượt 1: đáp án muộn nhất được học là **10/03/2017 22:30**, thời điểm bắt đầu dự đoán trong lượt kiểm tra là **22:40**. Đáp án được học vì thế vẫn thuộc quá khứ.

Với mỗi alpha, tính MAE ở ba lượt rồi lấy trung bình. MAE là trung bình độ lệch tuyệt đối giữa dự đoán và số thực tế; càng thấp càng tốt. Điểm trong bài này lấy trung bình trên cả ba khu vực.

Sau khi chọn alpha bằng các lượt bên trong train, mô hình học lại toàn bộ train và được đánh giá trên **7.859 mẫu validation** phía sau. **Lần tìm alpha này không đánh giá lại test.** Dự án đã có kết quả test từ lần benchmark trước; vì vậy không nói rằng test chưa từng được xem trong toàn bộ dự án.

## 4. Các đoạn code thật cần hiểu

Các số dòng dưới đây thuộc `src/gridwatch/tuning.py` ở phiên bản hiện tại. Có thể dùng Ctrl+G trong VS Code để đến dòng cần xem.

### Dòng 22 — Danh sách bắt đầu

```python
ALPHAS = {
    "Ridge": [0.01, 0.1, 1, 10, 100, 1000],
    "Lasso": [0.001, 0.01, 0.1, 1, 10, 100],
}
```

Thử theo cấp số nhân để xem được cả mức phạt rất nhẹ lẫn rất mạnh. Nếu tốt nhất nằm ở mép, code mở rộng tối đa hai lần. Nếu nằm giữa, thử thêm các giá trị quanh nó. Đây là tìm thô rồi tìm kỹ hơn, giúp tiết kiệm số lần chạy.

Ridge được thử thêm 0,001 và 0,0001. Lasso được thử thêm khoảng 0,3162 và 3,1623. Mỗi mô hình có tổng cộng tám alpha được kiểm tra. Giá trị 0,3162 xuất phát từ chia khoảng theo thang log, không phải một con số được đoán tùy ý.

### Dòng 37 — Chia theo thời gian

```python
splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon_steps)
folds = list(splitter.split(timing))
for learn, evaluate in folds:
    if timing.iloc[learn].target_time.max() >= timing.iloc[evaluate].origin.min():
        raise ValueError("A fold's training answer crosses its evaluation boundary.")
```

`learn` là vị trí mẫu được học; `evaluate` là vị trí mẫu được kiểm tra. `gap` chừa khoảng cách ở ranh giới. Dòng `if` kiểm tra thêm: nếu đáp án của train chạm vào thời điểm dự đoán trong lượt kiểm tra thì dừng, không tiếp tục chạy một thí nghiệm sai.

### Dòng 47 — Chuẩn hóa rồi học mô hình

```python
return make_pipeline(StandardScaler(), estimator)
```

Pipeline ghép hai bước: chuẩn hóa các cột về thang đo tương đương rồi đưa vào mô hình. Khi tìm alpha, scaler học riêng trên phần train của từng lượt. Nếu chuẩn hóa trước trên toàn bộ dữ liệu, trung bình và độ lệch chuẩn có thể chứa thông tin của phần chưa được học.

Lasso ở bước tìm kiếm có cấu hình thật như sau:

```python
estimator = Lasso(alpha=alpha, max_iter=3000 if initial else 200000, tol=.001,
                  selection="cyclic" if initial else "random", random_state=42,
                  precompute=not initial)
```

`max_iter` là giới hạn số vòng tối đa, không có nghĩa lần nào cũng chạy đủ 200.000 vòng. Alpha nhỏ cần cho bộ giải đủ thời gian hội tụ. `random_state=42` cố định việc chọn ngẫu nhiên để chạy lại được. `precompute` tính trước ma trận tích của các đặc trưng để giảm tính toán lặp; nó không thêm dữ liệu tương lai và không đổi mục tiêu của Lasso. Cấu hình ban đầu vẫn dùng 3.000 vòng và thứ tự cyclic để đối chiếu kết quả cũ. Vì bộ giải cũng được điều chỉnh, không khẳng định toàn bộ mức cải thiện chỉ đến từ alpha.

### Dòng 72 — Tự động thử và chọn

```python
search = GridSearchCV(
    make_model(name),
    param_grid={f"{name.lower()}__alpha": values},
    scoring=SCORER, cv=folds, refit=True, return_train_score=True,
    n_jobs=1, error_score="raise",
)
with warnings.catch_warnings():
    warnings.simplefilter("error", ConvergenceWarning)
    search.fit(X_train, y_train)
```

`param_grid` là danh sách giá trị cần thử. Ví dụ `ridge__alpha`: `ridge` là tên bước trong pipeline, hai dấu gạch dưới nối tên bước với tham số `alpha`. `cv=folds` yêu cầu dùng đúng ba lượt thời gian vừa tạo. `refit=True` học lại cấu hình thắng trên toàn bộ train. `n_jobs=1` chạy tuần tự để tránh làm máy quá tải. Nếu có lỗi hoặc Lasso chưa hội tụ, code dừng thay vì âm thầm xếp hạng kết quả chưa giải xong.

Ở dòng 34:

```python
SCORER = make_scorer(forecast_mae, greater_is_better=False)
```

Thư viện mặc định tìm điểm lớn nhất, nhưng MAE càng nhỏ càng tốt. Cấu hình này dùng điểm âm của MAE; khi ghi báo cáo code đổi lại thành số dương. Thấy điểm âm trong kết quả nội bộ không có nghĩa sai số thật âm.

### Dòng 126 và 211 — So sánh sau khi chọn

`comparison_row` tính sai số trên validation. `main` nối các bước: đọc dữ liệu → lấy train → tạo ba lượt → tìm alpha → so sánh trên validation → lưu bảng và biểu đồ. Hàm `search_alpha` chỉ nhận dữ liệu train cùng các lượt kiểm tra, không nhận validation ngoài hoặc test để chọn alpha.

## 5. Kết quả chạy thật và cách nhận xét

| Mô hình | Alpha trước | MAE validation trước | Alpha đã chọn | MAE validation sau |
| --- | ---: | ---: | ---: | ---: |
| Linear | Không có | 318,99 | Không có | 318,99 |
| Ridge | 100 | 351,45 | 0,0001 | 318,99 |
| Lasso | 10 | 338,73 | 0,316228 | 320,16 |

Tìm alpha giúp cải thiện hai cấu hình ban đầu. Tuy nhiên, Linear vẫn có MAE validation thấp nhất. Ridge gần như trùng Linear: mức phạt được chọn rất nhỏ, chênh lệch MAE chỉ khoảng 0,000022 nên không có ý nghĩa thực tiễn trong so sánh này.

Lasso có MAE trung bình các lượt trong train thấp nhất: khoảng 367,10, so với Linear 370,11. Nhưng khi kiểm tra giai đoạn validation phía sau, Linear tốt hơn. Đây là lý do cần phân biệt điểm dùng để chọn tham số và kết quả ở một giai đoạn tiếp theo.

Ridge vẫn chọn mép dưới của khoảng đã thử. Có thể giảm alpha thêm, nhưng hiện chưa chứng minh tìm được giá trị tốt nhất trên mọi alpha. Kết luận chính xác là **tốt nhất trong các giá trị đã thử theo quy trình này**. Không cần mở rộng vô hạn khi mức phạt nhỏ đã cho kết quả gần Linear.

## 6. Đọc biểu đồ

Mở `reports/figures/alpha_search.png`. Bên trái là Ridge, bên phải là Lasso.

- Trục ngang: alpha, dùng thang log để các mức 0,001; 0,01; 0,1; 1 cùng nhìn rõ.
- Đường xanh: MAE trung bình của ba lượt kiểm tra theo thời gian, dùng để chọn alpha.
- Đường xám: MAE trên phần được học trong các lượt đó, dùng để tham khảo.
- Chấm cam: alpha có MAE kiểm tra trung bình thấp nhất trong các giá trị đã thử.

Có thể nhìn đường cong giống ý tưởng tìm “khuỷu tay”, nhưng quy tắc chọn ở đây là **lấy MAE kiểm tra thấp nhất**. Không chọn bằng cảm giác nhìn biểu đồ. Chênh lệch train và kiểm tra còn có thể do mùa hoặc giai đoạn dữ liệu, không đủ để tự kết luận mô hình overfit.

## 7. Hai dòng thực tế để giải thích MAE

Lấy hai thời điểm đầu trong validation, khu vực 1. Số dưới đây giữ đơn vị của dữ liệu gốc; không tự gắn đơn vị chưa được nguồn xác nhận.

| Thời điểm dùng dữ liệu | Thời điểm cần dự đoán | Thực tế | Lasso trước | Sai số trước | Lasso sau | Sai số sau |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 12/09/2017 19:10 | 19:40 | 47.679,29 | 47.384,55 | 294,74 | 47.283,86 | 395,43 |
| 12/09/2017 19:20 | 19:50 | 47.475,40 | 47.497,26 | 21,86 | 47.513,80 | 38,40 |

Ví dụ dòng đầu: sai số sau là `abs(47679.29204 - 47283.8632141) ≈ 395.43`. Hai dòng này Lasso sau chỉnh còn dự đoán kém hơn cấu hình cũ. Nhưng trên toàn bộ 7.859 mẫu và ba khu vực, MAE giảm từ 338,73 xuống 320,16. Không dùng một vài dự đoán đẹp để kết luận cả mô hình tốt.

## 8. Lời nói khi báo cáo

“Nhóm em đã bổ sung bước chọn mức phạt alpha cho Ridge và Lasso. Em mở bảng kết quả tại đây. Nhóm chỉ dùng phần train để thử tham số, chia thành ba lượt theo thời gian và chừa khoảng cách vì bài toán dự đoán trước 30 phút. Trên biểu đồ, đường xanh là sai số kiểm tra trung bình; chấm cam là giá trị được chọn. Sau đó nhóm học lại trên toàn bộ train và so sánh trên validation. Ridge giảm MAE từ 351,45 xuống 318,99; Lasso giảm từ 338,73 xuống 320,16. Linear vẫn nhỉnh hơn, nên nhóm chưa kết luận mô hình phức tạp hơn là tốt hơn. Đây là code thử tham số và đây là các bảng lưu kết quả để chạy lại, kiểm tra.”

Nếu thầy hỏi “alpha đã tối ưu chưa?”, trả lời: “Đây là giá trị tốt nhất trong các alpha đã thử bằng ba lượt kiểm tra theo thời gian. Ridge còn nằm ở mép khoảng thử nên em không gọi đó là tối ưu tuyệt đối. Kết quả hiện tại cho thấy phạt rất nhẹ gần với Linear là phù hợp hơn phạt mạnh.”

Đã chạy thành công bảy kiểm tra tự động của dự án, gồm ba kiểm tra mới cho phần alpha. Chúng kiểm tra ranh giới thời gian, cách tính điểm và lựa chọn/refit tham số; chúng không chứng minh dự đoán sẽ luôn tốt trên mọi dữ liệu tương lai.

Tài liệu phương pháp: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [ví dụ chọn alpha cho Lasso](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html), [các tham số bộ giải Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html).
