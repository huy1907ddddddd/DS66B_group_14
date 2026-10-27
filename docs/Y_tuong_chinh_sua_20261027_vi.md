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
