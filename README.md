# Dự đoán điện sau 10 phút

DS66B, nhóm 14: Trần Quang Huy, Bùi Ngọc Thuấn, Nguyễn Hoàng Thành.

[Colab chính](https://colab.research.google.com/drive/1sNqD8rA6HrsUzjvDok5J7W1Zng2Qh5jQ) · [Colab giải thích từng dòng](https://colab.research.google.com/drive/18pQmIPzSMiz7GdoDSPnbF6hFkfei8XvF) · [Kết quả hiện tại](reports/Next_Step_Results_vi.md)

Một notebook chạy toàn bộ trên Colab: Runtime → Run all. Bản giải thích có cùng phép tính/kết quả, thêm comment tiếng Việt từng dòng.

## Mạch bài

1. Đọc và chia ngay theo thời gian, xem dữ liệu thô rồi đổi tên/định dạng giờ; biểu đồ train.
2. Persistence và tám mô hình với tám số đo hiện tại.
3. Thêm quá khứ và giờ thành 40 biến; thử thêm tuần/tháng.
4. Tìm tham số bằng CV thời gian, ba lượt, gap=1.
5. Thử học mức tăng/giảm, thử bỏ nhóm đầu vào.
6. Chốt bằng validation; bảng train/test, MAE từng khu vực/tuần/tháng, biểu đồ hai đường.

Mỗi bước có bảng trước/sau gồm validation và test. Cấu hình chọn bằng train/CV/validation; test chỉ đối chiếu, không dùng chọn hướng thử tiếp. Test đã được chấm từ các bản trước, không phải kiểm chứng hoàn toàn mới.

## Kết quả đã chạy

31 ô code sau khi tách bảng EDA; 42 hàng kết quả đối chiếu. Các bước cùng 32,227/7,861/7,862 dòng train/validation/test sau khi giữ chung mốc đủ lịch sử.
Chọn bằng validation: Linear; nhóm + Tuần, 49 đầu vào. MAE test 173.5309.
Lần chạy đầy đủ trước khi tách ô (cùng phép tính) trên máy mất 1295.0 giây; Colab có thể khác. Scaler/PCA chỉ fit trong train/CV. Giữ tìm Alpha, K và tham số cây/rừng/PCA/MLP.

[Bảng từng bước](reports/optimization_progression.csv) · [Mọi thử nghiệm](reports/all_experiment_results.csv) · [Ghi chú slide/report](docs/Bao_cuc_notebook_slide_report_vi.md)

Chưa thử cây riêng từng khu vực hoặc kiểm tra riêng tác động sin/cos.

Nguồn: [UCI Tetouan](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city), năm 2017 (CC BY 4.0); kiến thức lớp chương 11/13/14. Mã và giải thích tiếng Việt phục vụ nhóm học hiểu.
