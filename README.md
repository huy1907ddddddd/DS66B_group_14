# Dự đoán điện sau 10 phút

DS66B, nhóm 14: Trần Quang Huy, Bùi Ngọc Thuấn, Nguyễn Hoàng Thành.

[Colab chính — tiếng Anh](https://colab.research.google.com/drive/1sNqD8rA6HrsUzjvDok5J7W1Zng2Qh5jQ) · [Colab giải thích từng dòng — tiếng Việt](https://colab.research.google.com/drive/18pQmIPzSMiz7GdoDSPnbF6hFkfei8XvF)

Hai notebook trong notebooks/ tương ứng với hai bản Colab trên. Chạy từ đầu bằng Runtime → Run all. Dữ liệu được tải từ phiên bản CSV đã ghim trong repository.

## Mạch bài

1. Đọc dữ liệu, chia theo thời gian và kiểm tra dữ liệu train.
2. So Persistence với sáu mô hình dùng tám số đo hiện tại: Linear, Ridge, Lasso, KNN, Decision Tree và Random Forest.
3. Thêm lịch sử điện, thống kê trượt và biến thời gian. Chọn nhóm đầu vào riêng cho từng mô hình bằng MAE validation.
4. Tìm tham số bằng ba lượt CV thời gian trong train, gap bằng một dòng.
5. Thử RF dự báo mức tăng/giảm và thử bỏ nhóm đầu vào của Linear.
6. Chốt mô hình bằng validation, rồi đối chiếu train/test và sai số từng zone.
7. So với AR và VAR ở mục 10.1–10.3: tạo lag, chọn bậc bằng CV trong train và chấm trên cùng các thời điểm validation/test.
8. Vẽ MAE theo tuần/tháng và dự báo Zone 1 trong ba giờ, 24 giờ và toàn bộ test; xuất CSV.

## Kết quả hiện tại

Các mô hình được chấm trên cùng 32.227 dòng train, 7.861 dòng validation và 7.862 dòng test. Linear được chọn với nhóm + Tuần, gồm 49 biến đầu vào.

| Mô hình | MAE validation | MAE test |
|---|---:|---:|
| Persistence | 295,49 | 293,11 |
| AR(12) | 203,35 | 208,07 |
| VAR(12) | 196,25 | 226,52 |
| Linear, 49 biến | 163,70 | 173,53 |

Linear giảm MAE test 40,8% so với Persistence, 16,6% so với AR(12) và 23,4% so với VAR(12). AR và VAR thử các bậc 1, 3, 6, 12, 18, 36; mỗi họ chọn bậc riêng bằng CV và cùng chọn 12. Scaler chỉ fit trên phần học của từng lượt.

[Điểm từng lượt CV](reports/arvar_lag_order_cv.csv) · [Bậc được chọn](reports/arvar_selected_orders.csv) · [Bảng so sánh](reports/arvar_model_comparison.csv)

Hai notebook lưu code và kết quả của bản hiện tại. Thư mục reports/ còn giữ các báo cáo của những giai đoạn trước. Các lựa chọn dựa trên train/CV/validation; tập test đã được xem trong quá trình phát triển.

Nguồn: [UCI Tetouan](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city), dữ liệu năm 2017, giấy phép CC BY 4.0; [AR](https://otexts.com/fpp3/arima.html#autoregressive-models) và [VAR](https://otexts.com/fpp3/advanced.html#vector-autoregressions) theo Forecasting: Principles and Practice.
