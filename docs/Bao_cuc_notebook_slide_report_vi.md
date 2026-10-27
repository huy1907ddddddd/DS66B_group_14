# Ghi chú cho notebook, slide và report

Ghi ngày 27/10/2026 theo yêu cầu của nhóm DS66B, nhóm 14.
Đây là ghi chú thiết kế, chưa phải slide/report nộp cuối. Hai notebook đã tái cấu trúc và chạy kiểm tra theo mạch này ngày 27/10/2026; báo cáo kết quả nằm ở reports/Next_Step_Results_vi.md.

## Mạch chung

1. Bức tranh tổng quan: dữ liệu gì, cần dự đoán gì, X/y là gì, chia dữ liệu thế nào.
2. Cách làm đơn giản ban đầu: mốc Persistence và các mô hình cơ bản, chưa đưa tất cả bước tối ưu vào ngay từ đầu.
3. Kết quả ban đầu và vấn đề quan sát được.
4. Từng bước tối ưu: gặp vấn đề gì → ý tưởng xử lý → thay đổi code gì → so kết quả trước/sau → giữ hay bỏ.
5. Chốt phương án, xem sai số theo thời gian, kết luận và giới hạn.

## Nội dung tổng quan cần có

- Tetouan: 52.416 dòng, 01/01–30/12/2017, 364 ngày, cách nhau 10 phút.
- 9 cột gốc: thời gian, 5 thời tiết/bức xạ, 3 mức điện theo khu vực.
- Đầu vào hiện tại/quá khứ; đáp án là mức điện ba khu vực sau 10 phút.
- Chia ngay sau đọc: train 36.691, validation 7.862, test 7.863 dòng gốc; số dòng thực dùng giảm khi cần lịch sử và đáp án tương lai.
- Một biểu đồ train với ba khu vực; nếu gom ngày để vẽ phải ghi rõ là trung bình ngày, dữ liệu học vẫn 10 phút.

## Các bước tối ưu để tổ chức câu chuyện

Thứ tự dưới đây là dự kiến cho bản trình bày, không được viết thành lịch sử thực nghiệm nếu chưa chạy đúng các cấu hình tương ứng.

1. Tạo đầu vào từ quá khứ và thời gian: số điện trước đó, trung bình gần đây, chu kỳ giờ/tuần. Muốn khẳng định cải thiện cần chạy cấu hình trước/sau trên cùng các mốc đánh giá; chưa có điểm thì để chờ chạy, không điền số suy đoán.
2. So chu kỳ ngắn hạn, thêm tuần, thêm tháng, thêm cả hai. Hiện đã có bảng validation cho bốn nhóm đầu vào trên cùng các mốc đủ lịch sử.
3. Tìm tham số: Ridge/Lasso alpha, KNN K, độ sâu/số mẫu lá của cây/rừng, PCA và MLP. Dùng TimeSeriesSplit ba lượt với gap một bước; không bảo đảm cấu hình sau tìm tham số luôn tốt hơn cấu hình đầu.
4. Thử hai hướng bổ sung đã chạy: dự đoán phần tăng/giảm thay vì mức điện, và bỏ từng nhóm đầu vào. Giải thích vì sao có bước được giữ, có bước bị bỏ.
5. Dự kiến thử riêng từng khu vực, nhất là Decision Tree. Chưa chạy. Linear hiện tại đã có bộ hệ số riêng từng khu vực; tách lệnh fit đơn thuần không phải phương án cải thiện mới.

Không bắt buộc gọi thêm mô hình là tối ưu. Giới thiệu rõ tám mô hình/cấu hình chính đã thử: Linear, Ridge, Lasso, KNN, Decision Tree, Random Forest, PCA + Ridge, MLP; bốn cách đoán đơn giản là mốc đối chiếu.

## Bảng kết quả sau mỗi bước

Theo yêu cầu: có điểm test sau từng bước. Cấu hình/đầu vào/tham số vẫn phải chốt bằng train/CV/validation trước khi tính test; không chọn hướng tiếp theo dựa trên test.

Mỗi hàng dự kiến: bước/cấu hình, số đầu vào, validation MAE, test MAE, test RMSE, test MAPE, test R². Hiện thêm cột chênh lệch MAE với mốc ban đầu và bước trước; MAE thấp hơn là tốt hơn. Có thể để bảng đầy đủ trong report, slide chỉ dùng các cột chính.

So sánh phải cùng các thời điểm đáp án, cùng cách tính điểm và cùng cửa sổ train/validation/test. Không lấy điểm từ các bài 30 phút hoặc khác số dòng rồi gọi là cải thiện của bài 10 phút. Nếu thiếu cấu hình hoặc dự đoán tương ứng thì cần chạy bổ sung, không suy ra từ bảng khác.

Test hiện tại đã được chấm nhiều lần. Trong report gọi đúng là đối chiếu trên tập test đã dùng; không mô tả là kiểm chứng hoàn toàn mới. Nếu sau này quyết định dựa vào điểm test thì cần tập tương lai mới để kiểm chứng độc lập.

## Giải thích sin/cos bằng tình huống đời thường

Bắt đầu bằng câu hỏi: 23 giờ và 0 giờ cách nhau bao lâu? Thực tế một giờ, nhưng nếu chỉ dùng số giờ 23 và 0 thì khoảng cách số học là 23. Mô hình có thể hiểu nhầm hai thời điểm gần nhau thành rất xa nhau.

Ý tưởng: đặt giờ lên mặt đồng hồ tròn thay vì đoạn thẳng. 23 giờ và 0 giờ nằm cạnh nhau. Dùng hai tọa độ sin/cos để mô tả vị trí trên vòng tròn; chỉ dùng một tọa độ có thể khiến hai vị trí khác nhau bị trùng. Tương tự, chủ nhật và thứ hai cũng tiếp nối nhau trên vòng tuần.

Minh họa slide bằng mặt đồng hồ hoặc vòng tròn, nhấn hai điểm sát nửa đêm; không cần dồn các giá trị sin/cos lên slide. Công thức và code để trong notebook/report khi cần.

Đây là cách mã hóa chu kỳ thông dụng, không phải sáng kiến riêng của nhóm. Kể vấn đề dẫn đến ý tưởng như một tình huống minh họa; không bịa danh tính hoặc câu chuyện lịch sử về người đầu tiên nghĩ ra phương pháp. Sin/cos hữu ích để biểu diễn chu kỳ, nhưng không phải mọi mô hình đều bắt buộc dùng hoặc chắc chắn tăng điểm. Muốn nói có cải thiện trong bài này cần thử bỏ/giữ trên validation.

## Slide tối giản dự kiến

1. Bài toán và dữ liệu: sơ đồ hiện tại → sau 10 phút; số dòng, khoảng thời gian, ba khu vực.
2. Hiểu dữ liệu: biểu đồ train và một ví dụ X/y thật.
3. Mốc ban đầu và các mô hình: bảng nhỏ, nhận xét một câu.
4. Thêm đầu vào thời gian/quá khứ: hình đồng hồ sin/cos, kết quả trước/sau nếu đã chạy.
5. Các bước tối ưu: tuần/tháng, tham số, delta, lọc biến; mỗi bước ghi lý do và giữ/bỏ. Nếu quá chật thì tách thành hai slide.
6. Kết quả tổng hợp: bảng các mốc và một biểu đồ hai đường thực tế/dự đoán, chỉ rõ khu vực và cửa sổ hiển thị.
7. Kết luận và giới hạn: phương án chốt, sai số theo tuần/tháng, điều chưa chứng minh; có thể gộp vào slide 6 nếu đủ thoáng.

## Điểm đáng nhắc nếu đủ thời gian

- Persistence là đối thủ thực tế: đoán 10 phút tới bằng mức điện hiện tại. Cho thấy mô hình có hơn cách đoán đơn giản không.
- Linear thắng các phương án phức tạp đã thử: kết luận từ kết quả, không suy rằng mô hình phức tạp luôn kém.
- Học phần tăng/giảm: mức điện sắp tới = mức hiện tại + thay đổi dự đoán. Giải thích tự nhiên trước khi đưa code.
- Thêm tháng hoặc bỏ biến chưa chắc tốt hơn; báo cáo cả thử nghiệm không thành công.
- Ba vùng có xu hướng khác nhau trên train; đây là lý do đề xuất kiểm tra mô hình riêng, chưa phải bằng chứng rằng mô hình riêng tốt hơn.
- Bảng sai số theo tuần/tháng bổ sung cho một điểm trung bình toàn kỳ.
- Giữ đúng khoảng cách thời gian trong CV, không cho đáp án tương lai lọt vào đầu vào.

Report dùng cùng mạch nhưng bổ sung mô tả dữ liệu, công thức ngắn, tham số, bảng đầy đủ, nguồn và giới hạn. Kiểm tra lại yêu cầu ngôn ngữ/template môn học trước khi tạo bản nộp. Không dành nhiều slide cho cấu trúc thư mục/import hoặc đọc lại từng dòng code.

## Tình trạng sau khi sắp xếp lại

Ý tưởng EDA bổ sung của sếp: zone 2 và 3 tăng rõ trong mùa hè, cần đối chiếu giai đoạn tháng 7–9 với thời tiết và các biến thời gian trên train. Biểu đồ còn có nhịp giảm về tháng 9 và train chỉ tới 12/09. Đây là giả thuyết cần kiểm tra, chưa phải nguyên nhân đã xác định hay phương án tối ưu đã có kết quả. Xem mục 5 trong Y_tuong_chinh_sua_20261027_vi.md. Có thể dùng để kể mạch quan sát → đặt câu hỏi → thử nghiệm trong slide/report.

Hai notebook có 31 ô code sau khi tách bảng EDA và 42 hàng thử nghiệm. Mốc đầu chỉ tám số đo hiện tại; các bước thêm lịch sử, tuần/tháng, tìm tham số, delta, bỏ biến có bảng trước/sau. Dùng cùng 32,227/7,861/7,862 dòng train/validation/test. Slide nộp cuối chưa được tạo.
