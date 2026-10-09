# Hai phần khác biệt của GridWatch: quyết định cảnh báo và dự báo dự phòng

Nhóm 14, lớp DS66B: Trần Quang Huy, Bùi Ngọc Thuấn, Nguyễn Hoàng Thành.

## 1. Mở tổng thể trước khi giải thích code

Project: `D:\Kỳ I 2026-2027\Học thống kê\Midterm_Tetouan_GridWatch`.

Mở PowerShell trong thư mục đó, chạy `.\run_demo.ps1`, rồi mở `http://localhost:8501`. Hai phần mới là **Decision lab** và **Data quality**. Chọn Zone 1 ở thanh bên. Date/time trong thanh bên chỉ điều khiển tab Forecast replay; mỗi phòng thí nghiệm mới có bộ chọn mẫu riêng.

Các file cần chỉ cho thầy:

| File trong project | Việc file thực hiện |
|---|---|
| `src/gridwatch/decision_support.py` | Đếm sai sót ở các ngưỡng; chọn ngưỡng theo điểm phạt |
| `src/gridwatch/quality.py` | Kiểm tra đầu vào; chọn dự báo chính, dự phòng hoặc không dự báo |
| `src/gridwatch/operations.py` | Chuẩn bị dữ liệu, chạy hai thí nghiệm và lưu bằng chứng |
| `app/operations_panel.py` | Hiển thị hai tab mới; đổi giả định và xem mẫu thực tế |
| `reports/policy_assessment.csv` | Kết quả cảnh báo ở đoạn thời gian sau |
| `reports/quality_assessment.csv` | Độ chính xác và tỷ lệ có dự báo trong các tình huống lỗi |
| `artifacts/policy_replay.csv`, `quality_replay.csv` | Từng mẫu, thời điểm, số dòng nguồn, dự báo và đáp án để kiểm tra |

`.\run_operations.ps1` tạo lại kết quả của hai phần. `.\run_operations.ps1 -ShowResults` chỉ đọc kết quả đã lưu, không huấn luyện lại.

## 2. Câu chuyện để trình bày

“Nhóm em đặt người vận hành điện vào trung tâm. Dự báo trước 30 phút cho họ thời gian chuẩn bị kiểm tra hoặc sử dụng nguồn lực linh hoạt đã được cho phép. Nhưng cảnh báo nhiều quá gây phiền, cảnh báo ít quá có thể bỏ sót. Vì vậy, nhóm đánh giá cả quyết định cảnh báo và khả năng duy trì dự báo khi đầu vào có vấn đề.”

Các nguồn lực như pin hay tải có thể chuyển lịch là điều kiện của một ứng dụng thực tế, chưa phải thông tin có trong bộ dữ liệu này. Demo không điều khiển thiết bị, chưa tính tiền tiết kiệm và chưa tối ưu lịch vận hành pin. Điểm khác biệt là một quy trình có thể giải thích và kiểm chứng bằng dữ liệu, không phải tuyên bố phát minh ra một thuật toán mới.

## 3. Phần A — chọn cảnh báo theo hậu quả của sai sót

### Ý tưởng và kiến thức

Model đưa ra một điểm số. Ngưỡng quyết định biến điểm số đó thành có/không cảnh báo. **FN** là mẫu cao nhưng không cảnh báo; **FP** là mẫu bình thường nhưng cảnh báo. Nhóm cho người xem đặt điểm phạt cho hai sai sót:

`Tổng điểm phạt = điểm phạt bỏ sót × FN + điểm phạt báo nhầm × FP`.

Đây là cách đánh giá thông thường cho phân loại phục vụ quyết định. Nó nối trực tiếp với hồi quy, Logistic, confusion matrix, lựa chọn model và chia dữ liệu theo thời gian đã học. Điểm phạt là giả định do người dùng đặt, không được ước lượng từ chi phí doanh nghiệp thật. Chi phí khi cảnh báo đúng, mức độ cao của tải, công suất thiết bị, phản ứng của người vận hành và tác động đến sản xuất chưa được mô hình hóa.

### Chọn ngưỡng ở đâu, đánh giá ở đâu?

Model và ngưỡng nhãn cao vẫn lấy từ train cũ. Phần validation được chia tiếp:

- 3.926 mẫu đầu: chọn ngưỡng (calibration), từ 12/09/2017 19:10 đến 10/10/2017 01:20.
- Bỏ 3 mẫu ở ranh giới. Đáp án calibration cuối cùng ở 01:50, trước origin assessment đầu tiên ở 02:00.
- 3.930 mẫu sau: kiểm tra ngưỡng cố định (assessment), từ 10/10/2017 02:00 đến 06/11/2017 08:50.

Đoạn sau không được đưa vào hàm chọn ngưỡng. Tuy nhiên, đây vẫn là bằng chứng phát triển trên validation đã từng được xem, không phải tập test mới hoàn toàn. Phần test gốc không được đánh giá lại ở bước này. Mỗi nhãn cao ứng với một mẫu cách nhau 10 phút; nhiều mẫu liên tiếp có thể thuộc cùng một đợt cao điểm.

Zone 1 có 506 mẫu cao trong calibration, 290 trong assessment. Zone 2 tương ứng 274 và 601. Zone 3 không có mẫu cao trong cả hai đoạn; demo không cung cấp chính sách cảnh báo đã được xác nhận cho Zone 3.

### Code thực tế cần chỉ

Trong `decision_support.py`, hàm `choose_policy`, từ dòng 46:

```python
ranked = candidates.copy()
ranked["penalty"] = miss_penalty * ranked.fn + false_alarm_penalty * ranked.fp
return ranked.sort_values(["penalty", "fp", "fn", "threshold"],
                          kind="stable").iloc[0]
```

`candidates` là bảng tất cả các ngưỡng khác nhau và số lỗi trên calibration. Dòng giữa tính điểm phạt cho từng ngưỡng. `sort_values` xếp theo điểm phạt tăng dần; nếu bằng nhau, ưu tiên ít báo nhầm rồi ít bỏ sót. `.iloc[0]` lấy phương án đứng đầu.

Điểm code hơi đặc biệt nằm ở `threshold_candidates`, dòng 22. Code sắp điểm số một lần, dùng tổng tích lũy để đếm lỗi ở tất cả các ngưỡng. Các điểm bằng nhau được xử lý cùng một nhóm. `np.nextafter(score, np.inf)` lấy số máy tính nhỏ nhất nằm ngay trên `score`: với quy tắc `score >= threshold`, nó loại cả nhóm điểm bằng nhau một cách nhất quán. Đây là kỹ thuật số học chuẩn, không phải thuật toán mới. Cách này xét hết ngưỡng quyết định có thể tạo ra trên calibration mà không cần thử từng ngưỡng rồi đếm lại toàn bộ dữ liệu; khoảng O(n log n) thay vì O(n²).

Điểm tối ưu chỉ là tối ưu **điểm phạt đã đặt trên calibration cho điểm số hiện có**; không có nghĩa tối ưu mọi model hay tối ưu cho tương lai. Logistic dùng trọng số lớp cân bằng; điểm `predict_proba` ở đây chưa được xác nhận là xác suất đã hiệu chỉnh.

### Số liệu thật để nói và thao tác

Trong Decision lab: Zone 1 → Logistic → đặt điểm báo nhầm bằng 1.

| Điểm phạt mỗi bỏ sót | Ngưỡng chọn từ đoạn trước | Bỏ sót ở đoạn sau | Báo nhầm ở đoạn sau | Tổng điểm ở đoạn sau |
|---:|---:|---:|---:|---:|
| 1 | 0,949622 | 15 | 7 | 22 |
| 20 | 0,623382 | 1 | 44 | 64 |
| 200 | 0,623382 | 1 | 44 | 244 |

Không so 22 với 64 để bảo phương án đầu tốt hơn, vì hai dòng dùng hai thước đo khác nhau. Tại cùng giả định 20:1, ngưỡng chọn theo điểm phạt được 64 điểm, còn ngưỡng Logistic 0,5 được 82 điểm (1 bỏ sót, 62 báo nhầm). Đây là kết quả ở đoạn sau, không phải lời hứa cho mọi giai đoạn.

Đổi sang Zone 2, điểm phạt 200:1: Linear-score đạt 344 điểm (1 bỏ sót, 144 báo nhầm), Logistic đạt 517 điểm (2 bỏ sót, 117 báo nhầm). Có trường hợp Linear tốt hơn: nhóm không mặc định classifier phức tạp hơn sẽ có giá trị hơn. Không dùng kết quả assessment để tự động chọn lại model rồi gọi nó là bằng chứng độc lập.

### Lời nói khoảng 40 giây

“Nhóm em không cố giảm mọi loại lỗi cùng lúc. Em đặt một tình huống giả định: bỏ sót một mẫu cao đáng ngại gấp 20 lần báo nhầm. Code xét các ngưỡng trên đoạn dữ liệu trước, khóa ngưỡng rồi kiểm tra ở đoạn sau. Với Zone 1, so với ngưỡng 0,5, chúng em giữ một lần bỏ sót nhưng giảm báo nhầm từ 62 xuống 44. Điểm phạt giảm từ 82 xuống 64. Đây là đánh giá phát triển, chưa phải tiền tiết kiệm thật.”

## 4. Phần B — đầu vào có vấn đề thì làm gì?

### Ý tưởng và tradeoff

Model được học bằng dữ liệu hợp lệ. Đưa số bị nhân nhầm 10 lần hoặc giá trị thiếu vào model có thể gây lỗi hoặc dự báo vô lý. Nhóm đặt một cửa kiểm tra trước model:

1. Kiểm tra mọi đầu vào cần thiết có hữu hạn không; theo dõi thời tiết, số điện hiện tại và số điện trước 10 phút.
2. Dùng phạm vi min/max từ train, mở rộng mỗi phía 25% khoảng biến thiên; thêm điều kiện cơ bản như điện không âm, độ ẩm trong 0–100.
3. Nếu gói đầu vào đạt kiểm tra, dùng Linear.
4. Nếu không đạt, với từng khu vực thử số đo hiện tại còn trong phạm vi; nếu số hiện tại cũng không đạt, thử số đo trước 10 phút.
5. Nếu cả hai số đo đều không đạt, trả về chưa có dự báo. Giao diện không đưa ra cảnh báo classifier từ gói dữ liệu lỗi.

25% là giả định kỹ thuật cố định để tránh chặn mọi giá trị chỉ hơi vượt min/max; chưa được tối ưu. Phạm vi từ train không chứng minh số đo đúng/sai ngoài thực tế. Hai số đo dự phòng chỉ là “đạt các kiểm tra cơ bản”, không phải đã được cảm biến khác xác nhận. Đây là quy tắc Python và thống kê mô tả; phần phát hiện bất thường nâng cao của chương 15 vẫn chưa làm.

### Code thực tế cần chỉ

Trong `quality.py`, hàm `fit_quality_bounds`, dòng 11:

```python
minimum, maximum = observed.min(), observed.max()
span = (maximum - minimum).clip(lower=1e-9)
bounds = pd.DataFrame({"lower": minimum - margin * span, "upper": maximum + margin * span})
```

`observed` chỉ là dữ liệu train. `span` là khoảng biến thiên. `.clip(lower=1e-9)` tránh một cột hằng số có khoảng rộng đúng bằng 0. Đây là cửa kiểm tra đầu vào, không phải khoảng tin cậy của dự báo.

Trong `guarded_forecast`, dòng 28, đoạn từ dòng 61:

```python
for source_column, method in [(zone, "Persistence: current"),
                              (f"{zone}_lag_1", "Persistence: 10 minutes earlier")]:
    measurements = features[source_column].to_numpy(dtype=float)
    safe = (np.isfinite(measurements)
            & (measurements >= bounds.loc[source_column, "lower"])
            & (measurements <= bounds.loc[source_column, "upper"]))
    use = blocked & ~np.isfinite(forecasts[:, zone_index]) & safe
    forecasts[use, zone_index] = measurements[use]
    methods[use, zone_index] = method
```

Vòng lặp thử current trước rồi lag_1. `safe` kiểm tra số đo dự phòng; `use` chỉ chọn những dòng bị chặn, chưa có dự báo và có số dự phòng phù hợp. Nhờ vậy, số điện vừa bị nghi nhân nhầm không được tái sử dụng làm dự phòng cho chính nó. Trước vòng lặp, model chỉ được gọi trên các dòng đạt kiểm tra.

### Một dòng dữ liệu thực tế, bốn tình huống

Chọn Zone 1, origin **10/10/2017 02:00**, target **02:30**. Đây là dòng **40.622** trong CSV gốc, tính dòng tiêu đề là dòng 1.

Giá trị gốc: nhiệt độ **20,2**, điện tại 02:00 **26.203,58862**, điện trước 10 phút **26.493,47921**. Điện thực tế tại 02:30 là **26.172,07877**, chỉ được mở khi bấm Reveal. Đơn vị là đơn vị nguồn, chưa xác nhận kWh/kW.

| Tình huống ở đúng origin đó | Không kiểm tra đầu vào | Sau kiểm tra | Vì sao |
|---|---:|---:|---|
| Gói dữ liệu gốc | 26.094,24714 | 26.094,24714 | Dùng Linear |
| Đặt nhiệt độ thành thiếu | Không có dự báo | 26.203,58862 | Dùng số điện hiện tại |
| Nhân số điện hiện tại 10 lần thành 262.035,8862 | 255.936,82403 | 26.493,47921 | Số hiện tại bị chặn; dùng số trước 10 phút |
| Thiếu cả current và lag_1 của các khu vực | Không có dự báo | Không có dự báo | Báo chưa đủ đầu vào |

Các thay đổi lỗi được áp dụng lên **gói đặc trưng đã tạo**, không phải mô phỏng mất cảm biến kéo dài rồi tính lại toàn bộ lịch sử. Trong mỗi kịch bản, cùng một kiểu lỗi được áp dụng cho mọi origin ở assessment để stress-test; không có nghĩa lỗi ngoài đời xảy ra với tần suất đó.

### Kết quả trên toàn bộ 3.930 mẫu Zone 1

| Kịch bản | MAE không có gate | MAE có gate | Tỷ lệ có dự báo sau gate |
|---|---:|---:|---:|
| Dữ liệu gốc | 418,98 | 419,49 | 100% |
| Thiếu nhiệt độ | Không tính được (0 dự báo) | 1.069,96 | 100% |
| Số điện hiện tại bị nhân 10 | 278.785,15 | 1.394,05 | 100% |
| Không có current/lag_1 để dự phòng | Không tính được | Không tính được | 0% |

Gate chặn 2 mẫu trong dữ liệu gốc vì wind_speed vượt phạm vi đã học; chúng chưa được xác nhận là lỗi. Vì chuyển sang dự phòng, MAE tăng nhẹ. Mẫu đầu ở 25/10/2017 08:10, dòng CSV 42.819. Đây là đánh đổi thật cần công bố: bảo vệ trước lỗi mô phỏng nghiêm trọng, nhưng có thể chặn dữ liệu hợp lệ và giảm độ chính xác.

MAE chỉ tính trên những mẫu có dự báo. Luôn đọc MAE cùng tỷ lệ có dự báo; không gán lỗi bằng 0 cho những mẫu hệ thống không trả lời.

### Lời nói khoảng 40 giây

“Em mô phỏng gói đầu vào bị sai để kiểm tra hệ thống. Ở mẫu 02:00, nếu số điện bị nhân nhầm 10 lần, Linear dự báo khoảng 255 nghìn, trong khi đáp án chỉ khoảng 26 nghìn. Bộ kiểm tra chặn số đó và dùng số đo trước 10 phút, khoảng 26,5 nghìn. Khi không có số dự phòng phù hợp, hệ thống báo chưa thể dự đoán. Cơ chế này cũng chặn hai mẫu dữ liệu gốc nên em công bố cả mặt lợi và mặt đánh đổi.”

## 5. Thứ tự trình chiếu khoảng ba phút

1. Forecast replay: giới thiệu đầu vào, dự báo tại t+30 và cách mở đáp án sau đó.
2. Decision lab: Zone 1, Logistic; đổi điểm phạt bỏ sót từ 1 sang 20, báo nhầm giữ 1. Chỉ số bỏ sót và báo nhầm; nhấn ngưỡng chọn ở đoạn trước.
3. Mở `decision_support.py`, chỉ công thức trong `choose_policy`, rồi giải thích FP/FN.
4. Data quality: chọn mẫu 10/10 02:00. Chuyển từ Clean packets sang Temperature missing, rồi Zone 1 current multiplied by 10.
5. Mở `quality.py`, chỉ đoạn lựa chọn current rồi lag_1. Cuối cùng chọn No current or recent backup để cho thấy hệ thống không bịa một con số.
6. Kết thúc bằng giới hạn: dữ liệu lịch sử, chi phí giả định, lỗi mô phỏng; chưa chứng minh tiết kiệm thực tế hoặc phát hiện quá tải.

## 6. Câu hỏi thầy có thể hỏi

**Tại sao cần 30 phút?** Đó là chân trời dự báo ngắn để minh họa thời gian chuẩn bị vận hành. Khả năng thực sự can thiệp trong 30 phút phụ thuộc nguồn lực; nhóm chưa tối ưu chân trời theo doanh nghiệp.

**Ngưỡng tối ưu có luôn tốt nhất không?** Chỉ tốt nhất theo điểm phạt đã đặt trên calibration cho model đang xét. Sang đoạn sau có thể kém đi; demo báo kết quả thật thay vì bảo đảm.

**Tại sao không dùng công thức ngưỡng xác suất Bayes?** Công thức đó cần xác suất hiệu chỉnh và giả định chi phí phù hợp. Logistic ở đây có class_weight balanced và chưa xác nhận hiệu chỉnh xác suất, nên nhóm chọn ngưỡng trực tiếp từ lỗi quan sát trên calibration.

**Đột phá nằm ở đâu?** Ở thiết kế bài toán có quyết định, phép kiểm chứng theo thời gian và xử lý lỗi đầu vào minh bạch. Các thành phần thuật toán là phương pháp đã biết; nhóm chưa tuyên bố mới so với nghiên cứu thế giới.

**Đã sử dụng kiến thức chưa học chưa?** Phần hiện tại dùng Logistic/Linear, ma trận nhầm lẫn, chia tập theo thời gian, thống kê min/max và xử lý Python. Neural, phân cụm, PCA và detector bất thường nâng cao vẫn là phần mở rộng sau này.

## 7. Trạng thái bàn giao

Hai tab mới, code, bảng kết quả và kiểm tra đã có trên máy. GitHub hiện chưa được cập nhật các phần model và hai phần mở rộng này. Báo cáo LaTeX và slide ban đầu vẫn cần ghép nội dung mới trước khi nộp; tài liệu này là hướng dẫn trình bày bổ sung, không thay thế bản nộp chính thức.
