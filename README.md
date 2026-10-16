# Dự đoán điện sau 30 phút

**DS66B — Nhóm 14:** Trần Quang Huy, Bùi Ngọc Thuấn, Nguyễn Hoàng Thành.

[Mở Colab](https://colab.research.google.com/drive/1sNqD8rA6HrsUzjvDok5J7W1Zng2Qh5jQ) · [Xem notebook](notebooks/GridWatch_Colab.ipynb)

**Quy trình:** đọc CSV → tách test ngay → phân tích và chọn cách làm bằng train/validation → chấm test ở phần cuối. Chuẩn hóa và tìm tham số chỉ học từ train; không dùng điểm test để đổi mô hình.

| Phần | Đã làm |
|---|---|
| EDA | Kiểm tra dữ liệu; xem thống kê và điện theo giờ trên train |
| Feature Engineering | Tạo 48 đầu vào từ số đo hiện tại, quá khứ và thời gian |
| Model | So sánh 6 mô hình và 2 cách đoán đơn giản bằng MAE |
| Tối ưu | Tìm Alpha của Ridge/Lasso và K của KNN bằng 3 lượt chia thời gian trong train |
| Đánh giá cuối | So MAE, RMSE và R² trên train/test cho 6 mô hình ban đầu và các bản tối ưu |

`Runtime → Run all`: chạy từ đầu. `RUN_TREE_SEARCH = True`: tìm thêm tham số cây/rừng; chạy lâu hơn. Mọi kết quả được tính trong notebook, không đọc bảng điểm cũ.

Hai ý tưởng ứng dụng: cảnh báo điện cao và dùng số đo dự phòng khi đầu vào sai. Hiện mới là thử nghiệm, chưa chứng minh tiền tiết kiệm hay khả năng vận hành thực tế.

**Còn làm:** chốt cách chia chung, đối chiếu nghiên cứu, hoàn thiện báo cáo/slide và học tiếp chương 11–15.

30 phút là lựa chọn của nhóm; dữ liệu cách nhau 10 phút. Giữ đơn vị gốc. Test từng được xem trong bản cũ; cần ghi rõ lịch sử đó khi chốt đánh giá cuối. Phần 11 đánh giá test cuốn chiếu: dùng số đo hiện tại/quá khứ để dự đoán sau 30 phút, không dự đoán cả giai đoạn từ một mốc. Bảng so sánh giữ cả bản ban đầu và bản tối ưu, không chọn lại tham số theo điểm test.

Các thư mục code, app và báo cáo cũ được giữ để tra cứu; không phải phần phụ thuộc của notebook này.

**Nguồn:** [UCI](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) (CC BY 4.0) · [Yêu cầu môn học](https://docs.google.com/spreadsheets/d/17AhxqqU-uAUvH9RTIqbgO1w13_obfjTln45ukoH7SmM/edit) · [Alsalem 2025](https://www.nature.com/articles/s41598-025-91123-8). So sánh điểm nghiên cứu khác phải cùng cách chia và thời gian dự đoán.
