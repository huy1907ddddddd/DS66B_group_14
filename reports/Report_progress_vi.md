# Báo cáo tiến độ ngày mai

## Thông điệp chính

Nhóm đã có bản chạy được đến chương 10: kiểm tra dữ liệu, chia theo thời gian, so sánh mô hình, đánh giá cảnh báo và demo. Hai hướng khác biệt cho bản cuối là cảnh báo thích nghi và cơ chế tự kiểm tra/dự phòng; chưa tuyên bố chúng đã thành công.

## Bài nói khoảng 2 phút

“Nhóm em làm đề tài dự đoán nhu cầu điện tại ba khu vực của thành phố Tetouan. Mục tiêu hiện tại là dự báo mức điện sau 30 phút và đánh giá cảnh báo nhu cầu cao.

Nhóm đã kiểm tra bộ dữ liệu UCI gồm 52.416 mẫu, cách nhau 10 phút, không có dữ liệu thiếu hoặc trùng thời gian. Nhóm chia dữ liệu theo thời gian và có kiểm tra để tránh đưa thông tin tương lai vào mô hình. Cách chia hiện tại là dự kiến, nhóm sẽ thống nhất với các nhóm dùng cùng dữ liệu.

Về phương pháp, nhóm đã so sánh các cách dự báo đơn giản với hồi quy tuyến tính, Ridge, Lasso, KNN, cây quyết định và random forest. Trên tập chọn mô hình, hồi quy tuyến tính tốt nhất. Trên tập kiểm tra riêng, sai số tuyệt đối trung bình là 357 đơn vị gốc, giảm khoảng 54% so với cách dùng số đo hiện tại làm dự báo. Đây là kết quả cho giai đoạn kiểm tra này, chưa khẳng định cho mọi mùa.

Nhóm cũng thử phân loại trực tiếp nhu cầu cao với các phương pháp đến chương 10. Khu vực 2 phát hiện 1.906 trong 1.913 mẫu nhu cầu cao, nhưng vẫn có báo nhầm. Khu vực 1 chỉ có 9 mẫu nhu cầu cao và báo nhầm nhiều. Khu vực 3 không có mẫu nhu cầu cao trong giai đoạn kiểm tra nên chưa thể đánh giá độ nhạy cảnh báo.

Nhóm đã có demo để xem dự báo rồi mở số đo thực tế xuất hiện sau đó. Giai đoạn tiếp theo sẽ thử mô hình theo kiểu sử dụng sau khi học phân cụm, và kiểm tra dữ liệu bất thường để chuyển sang dự báo dự phòng. Nhóm sẽ dùng các thí nghiệm loại bỏ từng thành phần để chứng minh đóng góp, thay vì chỉ thêm tính năng vào giao diện.”

## Bản tiếng Anh nếu cần trình bày bằng tiếng Anh

“Our project forecasts electricity demand in three Tetouan distribution zones 30 minutes ahead and evaluates high-demand alerts. We audited 52,416 ten-minute observations and built a chronological evaluation pipeline. The split is provisional until we coordinate with the other groups.

We compared simple baselines with classroom regression models. Linear regression had the lowest validation MAE. On the held-out test period, its average MAE was 357 source units, about 54% lower than persistence.

We also compared direct event classification with alerts from forecast scores. Results vary by zone. Zone 2 has useful initial results, zone 1 has many false alarms and only nine positive samples, and zone 3 has no positive validation or test samples. We report these limitations explicitly.

The historical replay demo works. Next, we plan to test past-only regime adaptation and an input-quality gate with fallback. These are proposed contributions, and we will validate them through controlled comparisons.”

## Những câu thầy dễ hỏi

| Thầy hỏi | Trả lời ngắn, dễ hiểu |
|---|---|
| Nhóm đã làm gì cụ thể? | Kiểm tra dữ liệu, tạo đầu vào từ quá khứ, so sánh mô hình, đánh giá trên giai đoạn giữ riêng, làm notebook và demo. |
| Vì sao không chia ngẫu nhiên? | Dự báo thực tế chỉ biết quá khứ. Chia ngẫu nhiên có thể làm mô hình được học từ tương lai của thời điểm cần dự báo. |
| Train, validation, test là gì? | Train để học; validation để chọn mô hình và ngưỡng; test để kiểm tra sau khi đã chốt lựa chọn. Có 35.680 / 7.859 / 7.860 mẫu hợp lệ. |
| Vì sao dùng hồi quy tuyến tính? | Mô hình đó thắng trên tập chọn mô hình. Nhóm không chọn mô hình phức tạp chỉ vì tên nghe hay. |
| Giảm 54% nghĩa là gì? | Sai số MAE giảm từ khoảng 777 xuống 357. Không phải “độ chính xác 54%”, cũng không phải xác suất dự báo đúng. |
| Nhu cầu cao là quá tải à? | Chưa. Nhóm định nghĩa bằng ngưỡng 90% học từ dữ liệu train. Bộ dữ liệu không có công suất thiết kế hoặc nhãn mất điện. |
| SVM dùng ở đâu? | SVM tuyến tính dùng cho bài toán phân loại nhu cầu cao. Nhóm không nói đã học SVR nếu tài liệu chương 10 chỉ dạy phân loại. |
| Điểm khác biệt đã chứng minh chưa? | Chưa hoàn toàn. Nhóm có thí nghiệm đầu tiên cho cảnh báo, còn thích nghi và dự phòng sẽ kiểm chứng sau các chương tương ứng. |
| Vì sao khu vực 3 không có recall? | Không có mẫu dương thì không đo được khả năng phát hiện. Nhóm ghi chưa đủ bằng chứng, không ghi 100%. |
| Phân loại tốt hơn lấy dự báo so ngưỡng không? | Chưa thể kết luận chung. Khu vực 2 giảm báo nhầm nhưng bỏ sót 7 mẫu thay vì 6, và mức báo nhầm thực tế chưa bằng nhau. |
| Thời tiết có giúp không? | Thử bỏ thời tiết làm MAE validation giảm rất nhẹ. Hiện chưa có bằng chứng rõ nó giúp ở mốc 30 phút. |
| Kiến thức tuần 11–15 đã làm chưa? | Chưa. Đã có kế hoạch và giao diện tích hợp cho mạng neural, phân cụm, giảm chiều và phát hiện bất thường. Chương 13 còn thiếu tài liệu. |
| Ai làm phần nào? | Điền công việc thật của từng thành viên trước khi nộp. Có công cụ AI hỗ trợ tạo code; người trình bày phải kiểm tra và hiểu kết quả. |

## Demo 60 giây

1. Mở PowerShell tại folder dự án trên ổ D, chạy `./run_demo.ps1`, mở http://localhost:8501.
2. Chọn **Zone 2**, một ngày trong tập kiểm tra. Chỉ vào thời điểm phát hành và thời điểm dự báo sau 30 phút.
3. Chỉ số đo hiện tại và số dự báo. Bật **Reveal observations recorded later** để hiện số đo thực tế và sai số.
4. Mở **Verified experiment results**, chỉ vào bảng so sánh và số báo nhầm/bỏ sót.
5. Chọn **Zone 3** để giải thích vì sao hệ thống ghi **Not validated**.

Đừng học thuộc số liệu quá nhiều. Nhớ bốn ý: **bài toán – đã làm – kết quả có giới hạn – bước tiếp theo**. Bài nói dùng “nhóm đã làm” chỉ sau khi các thành viên đã xem và xác nhận sản phẩm này.
