# Kết quả dự đoán bước kế tiếp — 10 phút

DS66B, nhóm 14: Trần Quang Huy, Bùi Ngọc Thuấn, Nguyễn Hoàng Thành.
Cập nhật ngày 26/10/2026. Các bảng dưới đây lấy từ lần chạy thật đã kiểm tra.

## 1. Bài toán và cách chia

Dữ liệu cách nhau 10 phút. Mỗi dòng đầu vào dùng số đo hiện tại/quá khứ để dự đoán số điện ở 3 khu vực tại mốc tiếp theo.
Không gộp ba dòng. Trong code: horizon = 1; future_time = frame.time + horizon * time_step; y = frame[zones].shift(-horizon).
Chia theo thời gian ngay sau đọc CSV: train 70%, validation 15%, test 15%. Chừa một dòng ở ranh giới đáp án.
Sau khi tạo đầu vào và lấy cùng các mốc đủ lịch sử tháng để so sánh công bằng: train 32,227, validation 7,861, test 7,862 dòng.

## 2. Thử chu kỳ tuần và tháng

Giữ cùng Linear và cùng tập thời điểm. Tháng dùng tháng lịch trước qua DateOffset, không dùng cố định 30 ngày.

| Nhóm đầu vào | Số biến | Validation MAE |
| --- | --- | --- |
| Ngắn hạn | 40 | 166.4142 |
| + Tuần | 49 | 163.7002 |
| + Tháng | 47 | 168.1104 |
| + Tuần và tháng | 56 | 165.4323 |

Chọn nhóm + Tuần với 49 đầu vào bằng MAE validation.
Kết quả này không có nghĩa tháng không bao giờ hữu ích; chỉ cho biết cấu hình thêm tháng đang thử chưa tốt hơn trên cửa sổ validation này.

## 3. Mô hình và tối ưu

Giữ Linear/Ridge/Lasso/KNN/Decision Tree/Random Forest; thêm MLP nhỏ ở chương 11 và PCA + Ridge ở chương 14.
MLP chuẩn hóa cả X và y. PCA và các bộ chuẩn hóa nằm bên trong CV, chỉ học từ phần train của mỗi lượt.
PCA giữ phương sai X, không bảo đảm giữ cùng tỷ lệ thông tin dự báo.
Không đưa phân cụm hoặc phát hiện ngoại lệ vào bài chỉ để đủ chương. Hai phần cảnh báo và xử lý đầu vào lỗi đã bỏ khỏi notebook chính.

Tìm tham số bằng TimeSeriesSplit 3 lượt, gap=1, chấm MAE. Thử nghiệm có giới hạn cấu hình; không gọi là tối ưu tuyệt đối.

| Model | Tham số chọn | CV MAE | Validation MAE |
| --- | --- | --- | --- |
| Ridge | {'ridge__alpha': 0.001} | 185.9197 | 163.7004 |
| Lasso | {'lasso__alpha': 0.1} | 195.9828 | 183.3297 |
| KNN | {'kneighborsregressor__n_neighbors': 15} | 1,419.2993 | 1,837.1900 |
| Decision Tree | {'max_depth': None, 'min_samples_leaf': 5} | 1,003.3245 | 1,013.9816 |
| Random Forest | {'max_depth': 18, 'min_samples_leaf': 2} | 769.8846 | 910.6617 |
| PCA + Ridge | {'pca__n_components': 0.999, 'ridge__alpha': 0.1} | 361.2397 | 290.6022 |
| MLP | {'regressor__mlpregressor__alpha': 0.1, 'regressor__mlpregressor__hidden_layer_sizes': (32, 16)} | 341.9898 | 413.6242 |

## 4. Chọn trên validation, chấm test sau cùng

Mô hình chốt trước khi mở test: Linear. MAE validation: 163.7002.
Giữ cả bản ban đầu, bản tối ưu và bốn cách đoán đơn giản để đối chiếu.

| Model | Train MAE | Train RMSE | Train MAPE (%) | Train R2 | Test MAE | Test RMSE | Test MAPE (%) | Test R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 348.0153 | 523.9934 | 1.4638 | 0.9934 | 293.1130 | 456.7377 | 1.4608 | 0.9915 |
| Previous day | 1,137.5372 | 1,769.7652 | 4.8791 | 0.9150 | 951.8637 | 1,528.6008 | 4.8648 | 0.8949 |
| Previous week | 1,486.2506 | 2,230.1941 | 6.2850 | 0.8728 | 1,053.2395 | 1,641.1452 | 6.1375 | 0.8249 |
| Previous month | 3,075.5856 | 4,409.2883 | 13.0812 | 0.4976 | 2,215.2345 | 3,010.3732 | 11.7836 | 0.5977 |
| Linear | 174.8728 | 280.6289 | 0.7539 | 0.9981 | 173.5309 | 271.3365 | 0.8951 | 0.9967 |
| Ridge | 212.8458 | 331.6777 | 0.9173 | 0.9973 | 255.5105 | 366.1971 | 1.3066 | 0.9940 |
| Lasso | 238.4464 | 375.9571 | 1.0164 | 0.9965 | 277.6832 | 391.2877 | 1.3941 | 0.9937 |
| KNN | 0.0002 | 0.0004 | 0.0000 | 1.0000 | 3,213.8866 | 3,646.0513 | 18.8797 | 0.2541 |
| Decision Tree | 385.4110 | 549.3070 | 1.6597 | 0.9922 | 2,176.6558 | 3,067.6209 | 12.4191 | 0.5937 |
| Random Forest | 180.8794 | 286.0495 | 0.7846 | 0.9979 | 1,956.8568 | 2,818.7840 | 11.0688 | 0.6654 |
| PCA + Ridge | 720.9936 | 998.5047 | 3.0983 | 0.9758 | 1,111.9870 | 1,391.1437 | 5.6129 | 0.9192 |
| MLP | 552.0772 | 1,127.8258 | 2.0359 | 0.9713 | 536.4603 | 725.9228 | 2.6921 | 0.9783 |
| Ridge (tuned) | 174.8728 | 280.6289 | 0.7539 | 0.9981 | 173.5318 | 271.3373 | 0.8951 | 0.9967 |
| Lasso (tuned) | 194.0764 | 303.8416 | 0.8501 | 0.9978 | 196.1033 | 299.8879 | 1.0700 | 0.9953 |
| KNN (tuned) | 0.0002 | 0.0004 | 0.0000 | 1.0000 | 3,213.8866 | 3,646.0513 | 18.8797 | 0.2541 |
| Decision Tree (tuned) | 223.5394 | 334.1334 | 0.9718 | 0.9971 | 2,045.0757 | 2,952.3067 | 11.0290 | 0.6517 |
| Random Forest (tuned) | 116.1270 | 191.3044 | 0.5042 | 0.9991 | 1,878.9355 | 2,739.3874 | 10.4515 | 0.6910 |
| PCA + Ridge (tuned) | 290.4488 | 437.3137 | 1.2510 | 0.9953 | 348.7596 | 486.7430 | 1.8085 | 0.9891 |
| MLP (tuned) | 805.3658 | 1,500.3064 | 3.0127 | 0.9452 | 865.7364 | 1,073.5361 | 5.5248 | 0.9343 |
| Random Forest (delta) | 111.5020 | 179.9495 | 0.4830 | 0.9992 | 252.4308 | 384.1976 | 1.2816 | 0.9939 |
| MLP (delta) | 174.4744 | 272.4167 | 0.7524 | 0.9982 | 219.1560 | 324.3047 | 1.0917 | 0.9954 |

MAE: lệch trung bình theo đơn vị gốc. RMSE: nhạy với lỗi lớn. MAPE: phần trăm sai số. R² không phải phần trăm chính xác.
MAE/RMSE/MAPE tính chung ba khu vực; R² lấy trung bình điểm từng khu vực. Train KNN có thể bằng 0 vì gặp lại chính mẫu đã học.

## 5. Sai số thay đổi theo thời gian

Mỗi hàng được gắn với thời điểm có đáp án. Các bảng dưới so Persistence và ba cách có MAE validation thấp nhất, không chọn lại bằng test.
Các CSV giữ đủ tất cả mô hình. Tuần đầu/cuối có thể chỉ có một phần số ngày; số dòng được kiểm tra theo dữ liệu thực tế.

### MAE theo tuần

| Thời điểm đáp án | Persistence | Linear | Ridge (tuned) | MLP (delta) |
| --- | --- | --- | --- | --- |
| 2017-11-06/2017-11-12 | 304.2645 | 190.4812 | 190.4822 | 230.3459 |
| 2017-11-13/2017-11-19 | 316.8090 | 194.3530 | 194.3538 | 222.4360 |
| 2017-11-20/2017-11-26 | 295.0123 | 177.8240 | 177.8245 | 216.4102 |
| 2017-11-27/2017-12-03 | 277.2971 | 159.0940 | 159.0949 | 211.5337 |
| 2017-12-04/2017-12-10 | 281.6906 | 158.2579 | 158.2588 | 209.8802 |
| 2017-12-11/2017-12-17 | 301.1143 | 187.4947 | 187.4956 | 234.7789 |
| 2017-12-18/2017-12-24 | 280.9410 | 157.8260 | 157.8272 | 211.7756 |
| 2017-12-25/2017-12-31 | 287.6341 | 162.2850 | 162.2860 | 216.3274 |

### MAE theo tháng

| Thời điểm đáp án | Persistence | Linear | Ridge (tuned) | MLP (delta) |
| --- | --- | --- | --- | --- |
| 2017-11 | 302.4148 | 184.9023 | 184.9030 | 221.7492 |
| 2017-12 | 285.4864 | 164.2074 | 164.2084 | 217.0298 |

### Thực tế và dự đoán

![Thực tế và dự đoán ở 7 ngày đầu test](next_step_forecast.png)

Hai đường dùng cùng thời điểm đáp án. Biểu đồ hiển thị từng dòng 10 phút của bảy ngày đầu test; các bảng điểm dùng toàn bộ test.
File next_step_predictions.csv giữ thời điểm dự đoán, thời điểm đáp án, số thật, số dự đoán và sai số của cả ba khu vực.

## 6. Cách chạy và giới hạn

Mở notebook GridWatch_Colab.ipynb trên Colab, chọn Runtime → Run all. Không cần các file app/code cũ để chạy notebook này.
Lần kiểm tra trên máy: 751.8 giây, 21 ô code. Thời gian Colab có thể khác.
Đánh giá cuốn chiếu: tại mỗi mốc t được dùng số đo đã quan sát tới t, không dùng đáp án t+10 phút và không fit lại bằng test.
Các thay đổi mô hình được chọn bằng train/validation trước khi chấm test. Test đã được chấm trong các bản 30 phút và 10 phút trước. Vòng delta/lọc biến chỉ dùng train/validation để lựa chọn; test hiện tại là đối chiếu trên tập cũ, không phải tập kiểm chứng hoàn toàn mới.
Dữ liệu chỉ có Tetouan năm 2017. Điểm của bài 10 phút không so trực tiếp với điểm của bản 30 phút hoặc nghiên cứu khác cách chia.
Mã và giải thích tiếng Việt để nhóm đọc hiểu; cần theo đúng yêu cầu ngôn ngữ khi chuẩn bị bản nộp môn học.

Nguồn dữ liệu: [UCI Tetouan](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city).
Kiến thức lớp: chương 11 (MLP), chương 13 (metrics), chương 14 (PCA).


## 7. Hai phép thử tối ưu thêm

Đã chạy thêm RF/MLP học mức tăng giảm và tám cách bỏ nhóm biến. Giữ mô hình theo kết quả validation. Xem [bảng đối chiếu](Optimization_Results_vi.md).
