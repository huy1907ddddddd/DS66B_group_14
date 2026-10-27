# Ý tưởng chỉnh sửa — 27/10/2026

## 1. Dự đoán riêng từng khu vực

Trạng thái: đã lưu để thử sau; chưa huấn luyện thử nghiệm mới.

- X chứa số đo hiện tại/quá khứ, bao gồm điện của cả ba khu vực; y là điện sau 10 phút của ba khu vực. Cùng tên khu vực nhưng khác thời điểm nên có thể xuất hiện ở cả X và y.
- LinearRegression hiện tại đã có hệ số và phương trình riêng cho mỗi khu vực. Tách ba lần fit với cùng X/cùng hàng sẽ cho dự đoán tương đương trong sai số số học; không coi là tối ưu mới.
- DecisionTreeRegressor hiện tại dùng chung cấu trúc cây cho ba đáp án. Có thể thử ba cây riêng, mỗi cây học một cột y, cùng X để so sánh công bằng.
- So MAE từng khu vực và MAE trung bình ba khu vực trên validation, giữ nguyên các mốc chia theo thời gian. Nếu tối ưu tham số riêng từng khu vực thì chỉ dùng train/CV và validation; không chọn bằng test.
- Sau này có thể thử chọn đầu vào hoặc loại mô hình riêng cho từng khu vực; việc dùng riêng không đảm bảo tốt hơn.

## 2. Biểu đồ điện của train theo thời gian

Trạng thái: đã thêm và chạy kiểm tra trong cả notebook chính và bản giải thích từng dòng, ngay sau ô đọc/chia dữ liệu.

- Ba đường cho ba khu vực, chỉ dùng df.iloc[:train_end].
- Trung bình theo ngày để nhìn xu hướng gần chín tháng, không lấy tổng các mức công suất rồi gọi là điện năng.
- Không thay đổi dữ liệu 10 phút dùng để học, không dùng validation/test để vẽ, không huấn luyện lại mô hình.

## 3. Mạch notebook, slide và report

Trạng thái: đã sắp xếp lại và kiểm tra hai notebook theo mạch đơn giản → tối ưu; đã cập nhật báo cáo kết quả. Chưa tạo slide/report nộp cuối.

Theo ý sếp: bắt đầu bằng bức tranh tổng quan, giới thiệu các mô hình, đưa kết quả ban đầu; sau đó lần lượt giải thích mỗi bước tối ưu và so với mốc ban đầu/cấu hình trước đó. Sau mỗi bước có bảng đối chiếu test. Chi tiết nằm trong Bao_cuc_notebook_slide_report_vi.md cùng thư mục.

Mỗi thay đổi làm trên cả notebook chính và notebook giải thích từng dòng. Slide đơn giản, ít chữ; giải thích bằng vấn đề thực tế trước công thức. Không dùng điểm test để chọn bước tối ưu tiếp theo; chọn bằng train/CV/validation rồi ghi điểm test để đối chiếu. Test đã được đánh giá nhiều lần nên phải nói rõ đây là tập đối chiếu cũ.

## 4. Xem dữ liệu trước và sau xử lý

Đã tách hai ô ở cả hai notebook: ô 2.1 đọc/chia ngay rồi hiện hai dòng train với cột/ngày giờ gốc; ô 2.2 đổi tên/định dạng giờ rồi hiện lại hai dòng đó. Biểu đồ train ở ngay sau hai ô.

## 5. Zone 2 và 3 biến động trong giai đoạn tháng 7–9

Trạng thái: ghi nhận ý tưởng của sếp, chưa kiểm tra giả thuyết hoặc thay đổi mô hình.

- Nhận xét từ biểu đồ train: zone 2 và 3 tăng rõ trong mùa hè, đáng chú ý từ tháng 7; cần xem cả nhịp giảm về tháng 9, không mô tả là tăng liên tục suốt tháng 7–9. Train chỉ có tới 12/09 nên tháng 9 chưa đầy đủ.
- Câu hỏi: biến động này có đi cùng nhiệt độ, độ ẩm, gió, bức xạ hay giờ/thứ/tháng không? Có khác giữa các khu vực không?
- Khi tới phần EDA mở rộng, chỉ phân tích train: vẽ mức điện và biến đầu vào theo thời gian trên các ô/trục riêng; xem biểu đồ phân tán và tương quan, kiểm tra thêm theo cùng khung giờ hoặc cùng tháng để tránh nhầm hai biến cùng có xu hướng thời gian là quan hệ trực tiếp.
- Tương quan chỉ là dấu hiệu, chưa chứng minh nguyên nhân và chưa bảo đảm tăng chất lượng dự đoán. Không mặc định do điều hòa hay hoạt động kinh doanh khi dữ liệu chưa có thông tin đó.
- Nếu có dấu hiệu hợp lý, thử đầu vào đơn giản như nhiệt độ kết hợp giờ trong ngày hoặc ảnh hưởng riêng theo khu vực. So với bản hiện tại trên cùng các mốc bằng train/CV/validation; giữ bước mới khi validation tốt hơn, test chỉ đối chiếu sau khi chốt.
- Có thể đưa nhận xét này vào slide/report để nối câu chuyện: quan sát dữ liệu → đặt câu hỏi → kiểm tra → thử tối ưu → giữ hoặc bỏ theo kết quả. Chưa điền kết luận khi chưa chạy.

## 6. Mỗi bảng EDA một ô code

Đã tách ở cả hai bản: một ô kiểm tra dữ liệu, rồi các ô riêng cho thống kê chung, theo giờ, theo thứ và theo tháng; mục theo giờ đã đổi từ bảng sang biểu đồ đường. Giữ nguyên phép tính và các bảng đã chạy; không huấn luyện lại. Notebook hiện có 31 ô code.

## 7. Biểu đồ theo giờ thay bảng

Đã thay bảng theo giờ bằng ba đường cho ba khu vực trên cả hai notebook. Mỗi đường có 24 điểm, là trung bình các số đo cùng khung giờ trong train. Các giá trị khớp bảng cũ; chưa thay đổi mô hình.

## 8. Biểu đồ theo thứ và tháng thay bảng

Đã thay hai bảng bằng hai biểu đồ đường ở cả notebook chính và giải thích chạy trong VS Code. Mỗi khu vực một đường: bảy điểm T2–CN và chín điểm tháng 1–9. Giá trị khớp bảng cũ trên train. Đã ghi rõ tháng 9 chỉ đến 19:00 ngày 12/09, không coi là một tháng đầy đủ.
