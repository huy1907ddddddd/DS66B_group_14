# Dự đoán điện bước kế tiếp — 10 phút

DS66B, nhóm 14: Trần Quang Huy, Bùi Ngọc Thuấn, Nguyễn Hoàng Thành.

[Mở Colab](https://colab.research.google.com/drive/1sNqD8rA6HrsUzjvDok5J7W1Zng2Qh5jQ) · [Notebook](notebooks/GridWatch_Colab.ipynb) · [Kết quả chi tiết](reports/Next_Step_Results_vi.md)

Đọc CSV → tách test ngay → xử lý và chọn mô hình bằng train/validation → chấm test ở cuối. Dự đoán dòng kế tiếp, horizon=1, tương ứng 10 phút.

- So đầu vào ngắn hạn, thêm tuần, thêm tháng lịch, thêm cả hai trên cùng thời điểm.
- 8 mô hình: Linear, Ridge, Lasso, KNN, cây, rừng, PCA + Ridge, MLP; thêm 4 baseline để đối chiếu.
- Giữ tìm Alpha, K, độ sâu/số mẫu ở lá; tìm thêm tham số PCA và MLP bằng CV thời gian 3 lượt, gap=1.
- Bảng MAE/RMSE/MAPE/R² trên train/test, bảng MAE theo tuần/tháng, biểu đồ hai đường thực tế/dự đoán.
- Đã bỏ cảnh báo điện cao và xử lý đầu vào lỗi khỏi notebook chính.

Kết quả chọn bằng validation: nhóm + Tuần, 49 biến, mô hình Linear.
MAE validation 163.7002; MAE test 173.5309.
Các cấu hình thử và bảng đầy đủ nằm trong reports/. Mô hình đã chốt trước khi mở test.

Chạy Runtime → Run all trên Colab. Một notebook chứa toàn bộ quy trình; sklearn/numpy/pandas/matplotlib đã có trên Colab thông thường.
Lần chạy kiểm tra trên máy mất 447.8 giây; Colab có thể khác. Rừng và mạng nơ-ron là các phần tốn thời gian hơn.

Test cuốn chiếu: dùng số đo hiện tại/quá khứ đã có, không fit lại trên test. Dữ liệu chỉ có năm 2017; test từng được xem trong bản 30 phút trước đây.
Mã, app và báo cáo tuần 10 cũ là tài liệu lịch sử; notebook và báo cáo liên kết ở đầu trang là bản hiện hành.
Bản giải thích tiếng Việt phục vụ nhóm học hiểu; báo cáo/slide nộp theo yêu cầu ngôn ngữ môn học.

Nguồn: [UCI Tetouan](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) (CC BY 4.0); kiến thức lớp chương 11/13/14.
