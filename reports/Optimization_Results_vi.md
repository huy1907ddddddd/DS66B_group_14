# Thử hai hướng tối ưu mô hình

Cập nhật 26/10/2026. Dự đoán bước kế tiếp 10 phút, cùng 32.227 dòng train và 7.861 dòng validation.
Giữ Linear 49 biến làm mốc: validation MAE 163.7002, CV MAE 185.9190.
Các phép thử được lựa chọn bằng train/validation, trước khi chạy lại phần chấm test.

## 1. Học mức thay đổi điện — RF và MLP

Đáp án mới = số điện ở bước kế tiếp − số điện hiện tại. Dự đoán mức điện = điện hiện tại + mức thay đổi dự đoán.
Ví dụ dữ liệu thật khu vực 1 ngày 12/9/2017: 19:10 là 47.322,47788; 19:20 là 47.736,63717; mức tăng là 414,15929.

Clone Random Forest/MLP đã tìm tham số ở phần 6 rồi fit lại từ đầu với đáp án delta.
Đây là so sánh giữ cùng tham số để xét tác động đổi đáp án; không tuyên bố tham số cũ là tối ưu cho delta.
MLP học lại bộ chuẩn hóa y trên delta trong từng fold, không tái dùng bộ chuẩn hóa y của đáp án điện gốc.
CV 3 lượt/gap 1 dùng MAE delta; sai số bằng MAE mức điện sau khi cộng lại cùng giá trị hiện tại.
Không cắt delta âm; nó thể hiện mức điện giảm.

| Model | Validation MAE trước | CV MAE delta | CV độ lệch chuẩn | Validation MAE sau | Validation RMSE sau | Validation MAPE (%) sau | Validation R2 sau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest (delta) | 910.6617 | 256.6403 | 60.2178 | 196.3589 | 306.5373 | 0.9482 | 0.9959 |
| MLP (delta) | 413.6242 | 228.3822 | 40.5117 | 172.0831 | 257.3164 | 0.8299 | 0.9970 |

## 2. Bỏ nhóm đầu vào

Thử bỏ từng nhóm và một tập gọn. Cùng Linear, cùng các dòng dữ liệu, cùng CV 3 lượt/gap 1.
Chỉ đưa cách giảm biến vào nhóm ứng viên nếu cả CV và validation cùng tốt hơn bản gốc.
Đây là quy tắc thực dụng; không phải kiểm định ý nghĩa thống kê.

| Phương án | Số biến | CV MAE | CV độ lệch chuẩn | Validation MAE | Validation RMSE | Validation MAPE (%) | Validation R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Giữ nguyên | 49 | 185.9190 | 14.7062 | 163.7002 | 248.1691 | 0.7952 | 0.9972 |
| Bỏ thời tiết | 44 | 186.1875 | 15.3739 | 163.6213 | 248.1589 | 0.7950 | 0.9972 |
| Bỏ độ lệch chuẩn | 43 | 184.4486 | 12.1606 | 163.9602 | 249.1129 | 0.7950 | 0.9971 |
| Bỏ trung bình | 43 | 185.8556 | 14.7267 | 163.8047 | 248.2614 | 0.7955 | 0.9972 |
| Bỏ cả nhóm rolling | 37 | 184.4378 | 12.1992 | 164.1016 | 249.2103 | 0.7954 | 0.9971 |
| Bỏ lag 3/6/18 | 40 | 184.7321 | 13.1491 | 164.1011 | 248.7772 | 0.7972 | 0.9971 |
| Bỏ lịch sử tuần | 43 | 190.5180 | 15.0224 | 166.3398 | 250.6944 | 0.8131 | 0.9971 |
| Tập gọn | 26 | 242.1579 | 41.5777 | 192.7679 | 290.6699 | 0.9285 | 0.9962 |

Phương án có MAE validation thấp nhất: Bỏ thời tiết, 163.6213.
Phương án có CV MAE thấp nhất: Bỏ cả nhóm rolling, 184.4378.
Số phương án đạt cả hai tiêu chí: 0. Bản hiện tại giữ 49 đầu vào.

## 3. Quyết định và bảng đối chiếu test

Mô hình chốt bằng validation trước khi chạy phần test: Linear.
Hai mô hình delta giúp đánh giá việc đổi đáp án; chúng không tự thay thế Linear chỉ vì tốt hơn chính bản RF/MLP cũ.

| Model | Train MAE | Train RMSE | Train MAPE (%) | Train R2 | Test MAE | Test RMSE | Test MAPE (%) | Test R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear | 174.8728 | 280.6289 | 0.7539 | 0.9981 | 173.5309 | 271.3365 | 0.8951 | 0.9967 |
| Random Forest (tuned) | 116.1270 | 191.3044 | 0.5042 | 0.9991 | 1,878.9355 | 2,739.3874 | 10.4515 | 0.6910 |
| Random Forest (delta) | 111.5020 | 179.9495 | 0.4830 | 0.9992 | 252.4308 | 384.1976 | 1.2816 | 0.9939 |
| MLP (tuned) | 805.3658 | 1,500.3064 | 3.0127 | 0.9452 | 865.7364 | 1,073.5361 | 5.5248 | 0.9343 |
| MLP (delta) | 174.4744 | 272.4167 | 0.7524 | 0.9982 | 219.1560 | 324.3047 | 1.0917 | 0.9954 |

Bảng đầy đủ của 21 cách nằm trong train_test_model_comparison.csv; ảnh và sai số theo thời gian vẫn dùng mô hình chốt bằng validation.
Test đã được chấm trước đây. Vòng thử mới không dùng điểm test để đổi biến, tham số hoặc winner; kết quả test là đối chiếu trên tập cũ.
Lần chạy sạch gồm 21 ô code, 751.8 giây; số cảnh báo được ghi nhận: 0.
