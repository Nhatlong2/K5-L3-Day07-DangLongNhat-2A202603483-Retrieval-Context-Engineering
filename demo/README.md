# Demo Day 07 trên localhost

Demo có ba tab để nạp và cắt tài liệu, so sánh truy xuất, hỏi agent và kiểm tra câu trả lời tại nguồn. Phần truy xuất dùng TF-IDF, Dense cosine trên Qdrant local và Hybrid RRF của Day 07.

## Chạy trên Windows PowerShell

Mở PowerShell tại thư mục gốc của repo Day 07. Nếu chưa có venv Python 3.11, tạo bằng `py -3.11 -m venv .venv`.

```powershell
$demoPython = '.\.venv\Scripts\python.exe'
& $demoPython -m pip install -r demo/requirements.txt
& $demoPython -m streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8501
```

Mở [http://127.0.0.1:8501](http://127.0.0.1:8501). Dừng bằng `Ctrl+C`.

Luồng TXT, Markdown và PDF có văn bản, cắt chunk, TF-IDF và xem kết quả đã lưu chạy offline. Để dùng Dense, Hybrid hoặc hỏi agent, đặt `OPENAI_API_KEY` trong `.env` ở gốc repo theo mẫu `.env.example` hoặc nạp key vào phiên PowerShell theo [README Day 07](../README.md#nạp-api-key). Đặt `OPENAI_AGENT_MODEL` nếu dùng model khác có tool calling và structured outputs. Sau khi đổi key, khởi động lại demo.

**Tạo chỉ mục vector** gửi các chunk lên OpenAI theo lô. **So sánh truy xuất** có thể gửi câu hỏi để tạo embedding. **Hỏi agent** gửi câu hỏi và context đã chọn. Chỉ nạp tài liệu bạn được phép xử lý qua API. Chỉ mục vector của tài liệu nạp nằm trong `.day07/demo/vectors/`, gắn với hash tệp và cấu hình cắt. Chỉ mục TF-IDF của tài liệu nạp và trace phiên nằm trong bộ nhớ. Có thể tải trace UTF-8 từ giao diện.

## Kết nối corpus Day 06

1. Trong **Truy xuất & hỏi đáp**, chọn **Corpus Day 06** và mở **Kết nối Day 06**. Điền đường dẫn checkout. Dùng [commit Day 06 đã ghim](../benchmark/day06_reference.json).
2. Bấm **Chuẩn bị snapshot Lexical** để dựng lại 264 chunk B offline vào `.day07/demo/day06-chunks.json`. Demo cũng đọc được snapshot gốc `.day07/chunks.json` nếu có. Có thể chạy bước chuẩn bị bằng CLI.

```powershell
$day06 = (Resolve-Path '..\day06').Path
& $demoPython -m src.run prepare --day06 $day06 --chunks .day07/demo/day06-chunks.json --offline
```

3. Để dùng Dense hoặc Hybrid, nhập thư mục store có `active.json` và `meta.json`. Mặc định là `starter\.day06\qdrant_openai_large` trong checkout Day 06. Demo đọc collection B từ `active.json` và dùng bộ kiểm tra store của Day 07. Nếu chưa có store, làm theo [hướng dẫn dựng B](../README.md#2-mở-qdrant-store-của-day-06). Nếu store báo khóa, dừng tiến trình đang dùng thư mục đó rồi thử lại.
4. Có thể giới hạn tìm kiếm theo `asyncio`, `concurrent.futures` hoặc `threading`. Chọn preset hoặc chỉnh top-k và ngân sách context. Preset `lexical_k6` và `lexical_k6_budget3000` dùng để quan sát ảnh hưởng của ngân sách.

## Kịch bản trình diễn khoảng 3 phút

| Thời điểm | Thao tác |
|---|---|
| 0:00–0:45 | Dùng tệp mẫu hoặc nạp [phong_delta.txt](phong_delta.txt). Bấm **Cắt và xem trước**, chọn một chunk để xem phần tô sáng. Đổi A/B và cắt lại để so sánh ranh giới. |
| 0:45–1:25 | Với cách cắt B mặc định, lập chỉ mục **TF-IDF** rồi **vector**. Nhập “Phòng Delta mở cửa lúc mấy giờ vào thứ Hai?” và bấm **So sánh truy xuất**. Mở một kết quả và context của mỗi phương pháp để xem thứ hạng và phần được đóng gói. |
| 1:25–2:20 | Chọn Hybrid và bấm **Hỏi agent**. Kiểm tra câu trả lời **08:30** rồi bấm **Mở citation** để đối chiếu với đoạn nguồn. Hỏi tiếp “Phòng Delta mở cửa lúc mấy giờ vào thứ Bảy?” để kiểm tra trạng thái **Thiếu bằng chứng trong tài liệu**. |
| 2:20–3:00 | Sang **Trace & kết quả**, xem truy vấn, các tool call, context, token và chi phí nếu có đơn giá. Mở trace đã nộp `D01` và `DN01` để trình diễn citation và từ chối ngay cả khi offline. |

Khi không có key, **So sánh truy xuất** chạy Lexical và báo trạng thái của hai phương pháp cần embedding. Đổi cách cắt hoặc nạp tệp mới cần cắt lại và lập chỉ mục cho cấu hình tương ứng. Các chỉ số đã nộp được đọc từ kết quả dev/test của repo, với Hit@3 lấy từ `summary.json` và các điểm đã chấm từ `reviewed_metrics.json`.

## Kiểm thử

```powershell
& $demoPython -m unittest discover -s tests -v
& $demoPython -m unittest demo.test_demo -v
```

Đã chạy trên Linux với Python 3.11.18. 25 test gốc và 9 test demo đạt. Kiểm thử demo bao gồm UTF-8 BOM và dấu tiếng Việt, offset hai cách cắt, số trang PDF, đổi cấu hình sau khi lập chỉ mục, phục hồi lần lập chỉ mục lỗi, ba phương pháp trên Qdrant local với vector kiểm thử, agent với phản hồi OpenAI mô phỏng, citation ngoài context và giao diện qua Streamlit AppTest. Lệnh chuẩn bị offline đã dựng đúng 264 chunk B và `validate` đã kiểm tra cả hai split. Streamlit trả HTTP thành công trên localhost. Đã kiểm tra khả năng phân giải dependencies cho Windows x64/Python 3.11. Các bước gọi OpenAI thật, mở store vector Day 06 thật và chạy trên Windows bản địa cần xác minh trong môi trường của bạn.
