> Cập nhật mốc tuần 10: đã có kiểm tra dữ liệu, benchmark, cảnh báo, demo, notebook, slide và báo cáo LaTeX đã biên dịch. Kế hoạch bên dưới là bản chốt trước triển khai; các chương sau 10 vẫn chưa thực hiện. Xem `Bao_cao_ngay_mai.md` để dùng số liệu đã chạy.

# Kế hoạch dự án Tetouan GridWatch

Ngày lập: 08/10/2026. Đây là kế hoạch để thảo luận và triển khai; chưa có kết quả mô hình. Thời gian triển khai mốc tuần 10: dự kiến 2 giờ, tính sau khi chốt hướng.

## 1. Mục tiêu và sản phẩm cuối

Tên tiếng Anh dự kiến: **Tetouan GridWatch: Adaptive Electricity Demand Forecasting and Reliable Peak-Risk Alerts**.

Đối tượng sử dụng: người theo dõi nhu cầu điện tại ba khu vực của Tetouan. Hệ thống trả lời: nhu cầu sắp tới là bao nhiêu, có khả năng vào vùng nhu cầu cao không, và có dấu hiệu nào khiến dự báo cần được kiểm tra lại không?

Mốc hôm nay dự báo tại thời điểm cách hiện tại 30 phút. Bản cuối bổ sung 60 phút; 4 giờ là thí nghiệm mở rộng nếu hai mốc đầu ổn định. Không cam kết dự báo cả ngày khi chưa kiểm chứng. Dự báo một thời điểm tương lai khác với dự báo tổng năng lượng trong một khoảng thời gian.

Đầu vào gồm lịch thời gian đã biết, lịch sử tiêu thụ điện của ba khu vực, và các quan sát thời tiết đã có tại thời điểm phát hành dự báo. Không lấy thời tiết thực đo trong tương lai làm đầu vào. Mối liên hệ giữa các khu vực là một giả thuyết cần kiểm chứng, không mặc định giúp tăng độ chính xác.

Đầu ra gồm mức điện dự kiến của từng khu vực, tổng ba khu vực, cảnh báo nhu cầu cao, trạng thái chất lượng dữ liệu và lý do cảnh báo. Tổng được tính từ ba khu vực để các con số khớp nhau.

Dữ liệu không cung cấp công suất thiết kế của lưới, nhãn mất điện hay nhãn hỏng cảm biến. Vì vậy, dùng thuật ngữ **nhu cầu cao**; không kết luận quá tải, mất điện hay tiết kiệm tiền thực tế. Đơn vị đầu ra phải được xác minh từ nguồn trước khi ghi lên báo cáo và giao diện.

## 2. Yêu cầu đã xác minh từ giảng viên

- Vận dụng kiến thức học trên lớp vào bài toán ứng dụng cụ thể.
- Giải thích và chứng minh đóng góp so với các bài đã có trên Kaggle, GitHub hoặc blog.
- Có literature review: bài toán, đầu vào/đầu ra, nhóm phương pháp, thách thức và hướng nghiên cứu.
- Notebook/source code, báo cáo và slide. Nội dung nộp 100% tiếng Anh.
- Có bảng đóng góp thực tế của từng thành viên; không điền công việc chưa làm.
- Dùng nhiều bảng, biểu đồ và sơ đồ khi phù hợp.
- Đưa code lên GitHub và chia sẻ theo yêu cầu môn học. Việc gửi cho giảng viên/trợ giảng cần yêu cầu gửi rõ ràng của người dùng.
- Báo cáo tiến độ hằng tuần.
- Các nhóm dùng cùng dữ liệu phải thống nhất tập train/test. Thiết lập hiện tại của nhóm sẽ ghi là dự kiến cho đến khi thống nhất.

Các mốc ngày trong bảng môn học không đủ thông tin để xác định deadline hiện tại. Lịch tuần 11–15 bên dưới là lịch làm việc đề xuất, không phải lịch công bố của giảng viên.

## 3. Hai đóng góp đề xuất

### A. Cảnh báo nhu cầu cao có xét rủi ro và thích nghi theo kiểu sử dụng

Tách hai câu hỏi: dự báo một con số và dự báo một sự kiện nhu cầu cao. Nhãn nhu cầu cao ban đầu dùng ngưỡng theo từng khu vực, chẳng hạn phân vị 90% học từ tập training. Ngưỡng này là quy ước nghiên cứu, không phải giới hạn kỹ thuật của lưới điện.

So sánh hai cách cảnh báo:

1. Dự báo mức điện rồi so với ngưỡng.
2. Huấn luyện bộ phân loại trực tiếp để dự báo sự kiện nhu cầu cao sau 30 phút.

Cho phép chọn mức ưu tiên giảm bỏ sót hoặc giảm báo động nhầm. Ngưỡng quyết định và trọng số chỉ được chọn trên dữ liệu phát triển mô hình. Khi minh họa chi phí bỏ sót/báo nhầm, ghi rõ là giả định; trình bày nhiều mức giả định thay vì tự đặt một giá tiền thực tế.

Sau khi học phân cụm, thử nhận diện các kiểu sử dụng từ lịch sử đã quan sát, rồi chọn hoặc kết hợp mô hình phù hợp. Không dùng đường điện của cả một ngày chưa kết thúc để xác định kiểu ngày tại thời điểm dự báo. Chỉ giữ cơ chế thích nghi nếu nó đem lại lợi ích kiểm chứng được so với mô hình chung.

**Giả thuyết H1:** bộ phân loại trực tiếp giảm bỏ sót nhu cầu cao so với cảnh báo dựa trên dự báo số, khi so sánh ở cùng mức báo động nhầm hoặc cùng quy tắc chi phí.

**Giả thuyết H2:** mô hình theo kiểu sử dụng cải thiện sai số vào các thời điểm nhu cầu cao mà không làm tốc độ dự báo vượt ngân sách.

### B. Dự báo có cơ chế tự kiểm tra độ tin cậy và dự phòng

Hệ thống kiểm tra dữ liệu thiếu, thời gian gián đoạn, giá trị ngoài vùng quen thuộc và sai lệch dự báo đã có thể quan sát. Khi có dấu hiệu bất thường, hiển thị lý do và có thể chuyển sang phương pháp dự phòng đơn giản, dùng lịch sử hợp lệ gần nhất. Không coi mức điện cao là dữ liệu lỗi và tự xóa khỏi tập dữ liệu.

Mốc hôm nay chỉ thực hiện kiểm tra dữ liệu cơ bản và theo dõi sai số đã quan sát. Sau chương 15, so sánh các phương pháp như LOF hoặc One-class SVM để xây dựng cơ chế kiểm tra. One-class SVM đã xuất hiện trong chương 10, nhưng thí nghiệm phát hiện bất thường đầy đủ được dành cho giai đoạn sau.

Khoảng dự đoán có thể bổ sung như phần mở rộng: dùng một khoảng thời gian hiệu chỉnh riêng và đo tỷ lệ bao phủ thực tế cùng độ rộng. Dữ liệu chuỗi thời gian có phụ thuộc và thay đổi phân phối; không mặc định gọi khoảng đó là bảo đảm xác suất.

**Giả thuyết H3:** cơ chế kiểm tra và dự phòng giúp hệ thống suy giảm ít hơn khi đầu vào bị thiếu/nhiễu, với tỷ lệ cảnh báo nhầm chấp nhận được trên dữ liệu sạch.

Dữ liệu không có nhãn lỗi thực tế. Thử nghiệm tạo mất dữ liệu, nhiễu và đột biến trên bản sao phải ghi rõ là thử nghiệm lỗi mô phỏng. Điểm được thuật toán phát hiện trên dữ liệu thật chỉ là ứng viên cần xem xét, không phải lỗi cảm biến đã được xác nhận.

Các thành phần trên đều dựa trên phương pháp đã được biết đến. Đóng góp của nhóm là cách thiết kế, kết hợp và kiểm chứng trong bài toán cụ thể. Chưa có cơ sở để khẳng định thuật toán mới hoặc vượt nghiên cứu tốt nhất.

## 4. Ánh xạ kiến thức môn học

| Chương | Vai trò trong dự án | Mốc |
|---|---|---|
| 1. Introduction to Machine Learning | Xác định hai nhiệm vụ regression/classification; overfitting, mất cân bằng cảnh báo và thay đổi phân phối | Hôm nay |
| 2. Python for Machine Learning | Xử lý dữ liệu, biểu đồ, pipeline có thể chạy lại | Hôm nay |
| 3. Popular Datasets | Trình bày nguồn, cấu trúc, giới hạn và cách dùng dữ liệu UCI | Hôm nay |
| 4. Descriptive Statistics | Phân phối, phân vị, biến động theo giờ và tương quan giữa khu vực/thời tiết | Hôm nay |
| 5. Machine Learning Process | Tiền xử lý, chia dữ liệu, đánh giá, tối ưu và giám sát | Hôm nay |
| 6. Regression Models | Linear, Ridge, Lasso cho mức điện; Logistic Regression cho cảnh báo; Perceptron là đối chứng bổ sung nếu phù hợp | Hôm nay |
| 7. KNN | KNN regression và KNN classification làm đối chứng | Hôm nay |
| 8. Naive Bayes | Gaussian Naive Bayes cho cảnh báo từ các biến liên tục; không ép classifier dự báo một số liên tục | Hôm nay |
| 9. Decision Tree | Decision Tree/Random Forest cho dự báo và cảnh báo; phân tích vai trò đặc trưng | Hôm nay |
| 10. SVM | SVM cho cảnh báo; nghiên cứu ảnh hưởng chuẩn hóa, C và kernel. SVR là biến thể hồi quy, cần trình bày riêng nếu bổ sung | Hôm nay |
| 11. Deep Neural Networks | MLP/LSTM nhỏ làm ứng viên để kiểm tra lợi ích của học chuỗi; không mặc định tốt hơn Random Forest | Sau khi học |
| 12. Clustering | K-means phân kiểu lịch sử sử dụng; thử mô hình chuyên biệt/đặc trưng cụm | Sau khi học |
| 13. Chưa thấy tài liệu | Giữ mục thiết kế dự phòng; không tự đặt nội dung môn học | Chờ tài liệu |
| 14. Dimensionality Reduction | Chọn đặc trưng/PCA trên các biến lịch sử mở rộng; so sánh tốc độ và sai số trước/sau | Sau khi học |
| 15. Outlier Detection | Kiểm tra dữ liệu bất thường, thử LOF/DBSCAN/One-class SVM và cơ chế dự phòng | Sau khi học |

Dự án tận dụng các nhóm kiến thức theo vai trò phù hợp. Các nội dung như MNIST, text classification hoặc CNN xử lý ảnh là ví dụ môn học, không cần tạo chức năng ảnh/văn bản không liên quan. Các thuật toán không được chọn làm mô hình cuối vẫn có thể xuất hiện trong bảng đối chứng hoặc phần giải thích lựa chọn.

## 5. Thiết kế thực nghiệm

### Quy trình phổ biến cần làm đúng

- Xác minh số dòng, tần suất đo, mốc thời gian, giá trị thiếu/trùng và đơn vị từ file thật.
- Chia theo thời gian thành training, validation và test; ban đầu dự kiến 70/15/15 theo thời gian, điều chỉnh theo thống nhất giữa các nhóm. Không chia ngẫu nhiên cho phép đo dự báo thực tế chính.
- Loại các mẫu có thời điểm nhãn vượt qua ranh giới của từng tập. Các cửa sổ đánh giá phải tôn trọng thời điểm dữ liệu/nhãn trở nên sẵn có.
- Mọi chuẩn hóa, chọn đặc trưng, ngưỡng nhu cầu cao, PCA và phân cụm đều học từ dữ liệu được phép dùng trong training. Test được giữ riêng cho đánh giá cuối.
- Lịch sử gần trước điểm dự báo có thể được dùng nếu lúc dự báo đã thực sự quan sát được. Không dùng rolling window có chứa tương lai, nhãn tương lai hoặc thời tiết tương lai thực đo.
- Chọn mô hình/siêu tham số trên validation. Không xem test rồi liên tục sửa mô hình để tăng điểm.

### Đối chứng

- Persistence: tương lai bằng mức điện hiện tại.
- Seasonal naive: tương lai bằng thời điểm tương ứng của ngày trước, nếu dữ liệu hợp lệ.
- Hồi quy: Linear/Ridge/Lasso, KNN, Decision Tree, Random Forest.
- Cảnh báo: Logistic Regression, Gaussian Naive Bayes, KNN, Decision Tree, Random Forest, SVM. Perceptron là đối chứng gọn bổ sung.
- Mạng nơ-ron và mô hình thích nghi được bổ sung sau khi học.

Giai đoạn hai giờ chỉ chạy tập cấu hình nhỏ để có bảng kết quả ban đầu. Kernel SVM có thể tốn thời gian; nếu phải giảm mẫu, ghi rõ và so sánh các mô hình liên quan trên cùng tập con. Không đặt kết quả huấn luyện trên các tập khác nhau vào một bảng và coi là phép so sánh công bằng. Lượt benchmark đầy đủ và tối ưu rộng dành cho tuần sau.

### Các phép thử đóng góp

1. Bỏ lịch sử điện: thời tiết/lịch có đủ để dự báo không?
2. Bỏ thông tin khu vực khác: mối liên hệ ba khu vực có giúp không?
3. Bỏ bộ phân loại cảnh báo: dự báo con số rồi đặt ngưỡng khác gì?
4. Bỏ phân cụm/thích nghi: thành phần đó có ích thật không?
5. Bỏ cơ chế kiểm tra và dự phòng: khi dữ liệu bị lỗi mô phỏng, hệ thống suy giảm thế nào?
6. Bỏ PCA/chọn đặc trưng: tốc độ cải thiện có đổi lấy sai số lớn hơn không?

### Tiêu chí đánh giá

- Dự báo: MAE, RMSE, R² theo khu vực, theo khoảng dự báo, theo từng giai đoạn thời gian và riêng vùng nhu cầu cao.
- Cảnh báo: precision, recall, F1, ma trận nhầm lẫn, số báo động nhầm và tỷ lệ bỏ sót. Xem thêm cảnh báo theo từng đợt để tránh nhiều cảnh báo liên tiếp làm kết quả trông tốt giả tạo.
- Độ tin cậy: tỷ lệ dữ liệu bị gắn cờ, ảnh hưởng của cơ chế dự phòng, hiệu quả trên lỗi mô phỏng; tỷ lệ bao phủ/độ rộng nếu có khoảng dự đoán.
- Hiệu quả tính toán: thời gian training, thời gian dự báo và tốc độ phản hồi demo trên máy thật.
- Bản cuối đánh giá qua nhiều cửa sổ thời gian tiến dần. Nếu làm khoảng bất định cho mức cải thiện, dùng cách tôn trọng phụ thuộc thời gian, chẳng hạn lấy mẫu theo khối ngày.

Không so trực tiếp điểm số của bài báo với điểm số nhóm khi khác cách chia dữ liệu, mốc dự báo hoặc đầu vào. Ưu tiên tái hiện một đối chứng khả thi dưới cùng thiết lập.

## 6. Mốc hôm nay: kế hoạch hai giờ

| Thời gian | Khối công việc | Đầu ra kiểm chứng được |
|---|---|---|
| 0–20 phút | Tổ chức dự án, lấy dữ liệu, kiểm tra và dựng biểu đồ chính | Data audit, nguồn dữ liệu, biểu đồ và định nghĩa đầu vào/nhãn |
| 20–65 phút | Pipeline thời gian, baseline, benchmark gọn cho dự báo/cảnh báo đến chương 10 | Bảng validation, cấu hình, predictions; mô hình được chọn bằng validation |
| 65–90 phút | Phân tích giờ cao điểm và một thử nghiệm bỏ thành phần; đánh giá test ban đầu đã chốt | Bảng sai số, bỏ sót/báo nhầm, kết luận có giới hạn |
| 90–120 phút | Demo đơn giản, khung báo cáo/slide, hướng dẫn trình bày và GitHub nếu sẵn sàng | Demo chạy được, 5–6 slide tiến độ, lời trình bày và câu hỏi thường gặp |

Ưu tiên thứ nhất là pipeline và kết quả có thể tái chạy. Nếu tính toán hoặc môi trường chậm, giảm trang trí giao diện và phạm vi tối ưu. Không đưa kết quả chưa chạy vào báo cáo. Khi chưa đủ thời gian cho một thuật toán, ghi rõ pending và lý do.

Các phần sau chương 10 chỉ có đầu vào/đầu ra, cấu hình tắt, điều kiện tích hợp và danh sách thí nghiệm. Không có điểm số giả, không có file mô hình giả. Demo hiển thị phần mở rộng là planned.

## 7. Lộ trình năm tuần còn lại

| Tuần dự án | Trọng tâm | Điều kiện hoàn thành |
|---|---|---|
| 11 | Tối ưu benchmark hiện có; thêm MLP/LSTM nhỏ sau khi học chương 11 | Bảng so sánh cùng đầu vào/mốc dự báo; có thời gian chạy và phân tích khi neural network không tốt hơn |
| 12 | Phân cụm kiểu sử dụng và kiểm tra cơ chế thích nghi | So sánh mô hình chung với mô hình theo kiểu; chỉ nhận dạng kiểu từ thông tin sẵn có |
| 13 | Hoàn thiện H1/H2, nhiều cửa sổ thời gian; cập nhật khi nhận tài liệu chương 13 | Bảng thử bỏ thành phần và kết quả ổn định qua thời gian; chương 13 được ánh xạ khi biết nội dung |
| 14 | Chọn đặc trưng/PCA; tối ưu tốc độ và độ gọn | Bảng sai số–thời gian–số đặc trưng và quyết định giữ/bỏ PCA có bằng chứng |
| 15 | Phát hiện bất thường, cơ chế dự phòng, kiểm tra cuối, hoàn thiện report/slides/demo | H1–H3 có kết luận; chạy lại dự án được; mọi kết quả và giới hạn có nguồn rõ ràng |

Các tuần này là lịch triển khai đề xuất. Tiến độ học thực tế của giảng viên có thể khác số chương; khi đó giữ giới hạn kiến thức đã học và đổi thứ tự việc tương ứng.

## 8. Tổ chức code dự kiến trên ổ D

Thư mục dự kiến: `D:\Kỳ I 2026-2027\Học thống kê\Midterm_Tetouan_GridWatch`. Thư mục này chưa được tạo trong bước thảo luận.

```text
Midterm_Tetouan_GridWatch/
  README.md
  requirements.txt
  config.yaml
  data/                 # raw giữ nguyên, processed có thể tạo lại
  src/                  # data, features, forecasting, alerts, evaluation
  notebooks/            # một notebook trình bày kết quả, gọi lại code trong src
  app/                  # demo
  reports/              # kế hoạch, nguồn nghiên cứu, bảng/biểu đồ, LaTeX và slide
  artifacts/            # mô hình, kết quả máy sinh, split manifest
```

Không chia thành hàng chục file rất nhỏ. Một nơi sở hữu mỗi phép xử lý. Dữ liệu/mô hình sinh ra được quản lý bằng `.gitignore`; GitHub chứa code, cấu hình, tài liệu và kết quả gọn cần cho tái hiện. Không đẩy sách, slide môn học hoặc mẫu của người khác lên repository của nhóm.

Tên repository theo lớp/nhóm thực tế và mẫu giảng viên yêu cầu. Chưa tự đặt tên nhóm hoặc suy đoán thành viên. Khi tài khoản GitHub chưa sẵn sàng, vẫn hoàn thành repository Git cục bộ và hướng dẫn chạy.

## 9. Khung báo cáo theo mẫu đã đọc

Mẫu report thực tế gồm bìa, phần đầu, cơ sở bài toán, nghiên cứu liên quan, phương pháp, thực nghiệm, phân tích thành phần, hiệu quả tính toán và kết luận/hướng phát triển. Mẫu slides cũng có related work, proposed method, experimental setup, comparison và nhiều ablation studies. Dùng cấu trúc trình bày; không sao chép kết quả hoặc tuyên bố của đề tài mẫu.

Khung report tiếng Anh của nhóm:

1. Problem Overview and Research Objectives.
2. Related Work and Theoretical Foundations.
3. Proposed Forecasting and Alerting Framework.
4. Experiments, Ablation Studies and Computational Efficiency.
5. Conclusions, Limitations and Future Work.

Có abstract, sơ đồ hệ thống, bảng mô tả dữ liệu, literature review matrix, bảng benchmark, đóng góp thành viên và tài liệu tham khảo. Dựng khung từ hôm nay; chỉ điền kết quả sau khi thực nghiệm. Nội dung nhóm nộp 100% tiếng Anh, dù mẫu report hiện có nhiều đoạn tiếng Việt.

Buổi báo cáo ngày mai: 5–6 slide và lời nói khoảng 2–3 phút, theo thứ tự mục tiêu, dữ liệu, thiết kế, kết quả ban đầu thực tế, đóng góp dự kiến, kế hoạch còn lại. Viết lời trình bày sau khi chạy để mọi câu về tiến độ đều đúng.

## 10. Nguồn và mức độ đã kiểm tra

- Đã đọc lại bảng yêu cầu môn học: https://docs.google.com/spreadsheets/d/17AhxqqU-uAUvH9RTIqbgO1w13_obfjTln45ukoH7SmM/edit?gid=673209130
- Đề tài số 18 và nguồn dữ liệu: https://docs.google.com/spreadsheets/d/17AhxqqU-uAUvH9RTIqbgO1w13_obfjTln45ukoH7SmM/edit?gid=108516761
- UCI: https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city
- Đã mở mẫu report, tải nguồn và rà cấu trúc các file LaTeX: https://www.overleaf.com/read/dvxypbdcsphw#f2176e
- Đã mở mẫu slides, tải nguồn và đọc phần thiết lập/so sánh/ablation: https://www.overleaf.com/read/mfprxpjmtmfc#46c516
- Nghiên cứu cùng dữ liệu đã kiểm tra: Alsalem (2025), fuzzy clustering kết hợp các mô hình ML. Vì hướng clustering + forecasting đã có, không coi riêng tổ hợp đó là tính mới: https://www.nature.com/articles/s41598-025-91123-8
- Đã rà soát bộ slide chương 1–12, 14–15, tổng 366 trang theo các file hiện có. Chưa thấy chương 13. Đọc nội dung trích xuất của slide để lập kế hoạch; công thức/hình ảnh sẽ được kiểm tra trực quan ở phần cần dùng khi triển khai.
- Đã rà cấu trúc và các phương pháp trong notebooks thực hành. Ba sách tham khảo đã kiểm tra danh mục/phần đầu để định hướng tra cứu, chưa đọc toàn bộ hơn 1.400 trang sách.

Tài liệu môn học đối chiếu theo chương, trong `D:\Kỳ I 2026-2027\Học thống kê`: Introduction to Machine Learning; Python for Machine Learning; Polular Datasets in Machine Learning; Descriptive Statistics of Datasets; Machine Learning Process; Regression Models; K-Nearest Neighbor Algorithm; Naive Bayes Classifier; Decision Tree; Support Vector Machine; Deep Neural Networks; Clustering Models; Dimensionality Reduction Methods; Machine Learning for Outlier Detection.

Nguyên tắc làm việc: trước mỗi khối, nói rõ việc định làm, lý do, phương pháp phổ biến hay cải tiến đề xuất, và đầu ra sẽ dùng để kiểm chứng. Chỉ chuyển kết quả đã xác minh sang báo cáo.
