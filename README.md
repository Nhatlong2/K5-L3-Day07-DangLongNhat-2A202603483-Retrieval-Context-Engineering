# Day 07 — Agentic RAG trên kho Day 06

Bài cá nhân dùng Python 3.11+, Qdrant local và OpenAI. Agent tra cứu tài liệu Python 3.11.9, trả lời kèm `[chunk-id]` hoặc từ chối khi thiếu bằng chứng. Xem [yêu cầu của lab](GUIDE.md), [cách chấm](BENCHMARK.md), [rubric](RUBRIC.md), [quy tắc](RULES.md) và [danh sách bài nộp](SUBMISSION.md).

## Kết quả đã kiểm chứng

Đã chạy 85 lượt agent trên dev cho 5 cấu hình, sau đó khóa hybrid và chạy 14 lượt test cho dense và hybrid. Các lượt API dùng collection B với 264 chunk từ 45 mục tài liệu Python 3.11.9. Xem [báo cáo cá nhân](report/individual/2A202603483_DangLongNhat.md) và [kết quả agent](results/agent-dev/).

Trên test, dense và hybrid đều đạt Evidence Hit@3 gốc 4/5. Context đủ bằng chứng lần lượt là 4/5 và 5/5. Điểm grounded là 0,00/2 và 0,40/2. Nhãn do AI hỗ trợ và đang chờ học viên xác nhận. [Kết quả TF-IDF offline](results/offline-dev/)

## 1. Chuẩn bị trên Windows PowerShell

Chạy từ gốc repo Day 07. Dùng đường dẫn thật của repo Day 06 đã chạy ingest. Nếu cần clone, lấy đúng bản tham chiếu trong `benchmark/day06_reference.json`.

```powershell
$day07 = (Get-Location).Path
py -3.11 -m venv .venv
$day07Python = Join-Path $day07 '.venv\Scripts\python.exe'
& $day07Python -m pip install -r requirements.txt
& $day07Python -m unittest discover -s tests -v
```

Nếu chưa có repo Day 06 cùng cấp, chạy từ gốc Day 07.

```powershell
git clone https://github.com/Nhatlong2/K5-L3A-DAY06-DANGLONGNHAT-2A202603483-DataIngestion.git ..\day06
git -C ..\day06 switch --detach 7d7c7ea42a620ae88cea16dabcb58f546c778ef5
```

Đặt đường dẫn checkout cho các lệnh bên dưới. Sửa đường dẫn nếu repo Day 06 nằm ở thư mục khác.

```powershell
$day06 = (Resolve-Path '..\day06').Path
```

### Tái lập dev offline

```powershell
& $day07Python -m src.run prepare --day06 $day06 --offline
& $day07Python -m src.run validate
& $day07Python -m src.run benchmark --split dev --mode retrieval --output .day07/offline-run
```

Lệnh `prepare` dùng đúng code chia chunk B của Day 06 và kiểm tra hash. `validate` kiểm tra số lượng, scope và đoạn trích cho hai split. Benchmark offline đo TF-IDF và context đóng gói theo query gốc. Mỗi lượt chạy dùng thư mục output riêng.

### Nạp API key

Cách dùng biến môi trường hoạt động với cả Day 06 lẫn Day 07 và giữ giá trị khỏi màn hình.

```powershell
$day07Key = Read-Host 'OPENAI_API_KEY' -AsSecureString
$env:OPENAI_API_KEY = (New-Object System.Net.NetworkCredential('', $day07Key)).Password
Remove-Variable day07Key
```

Day 07 cũng tự đọc `.env` ở gốc repo. Tạo từ `.env.example` bằng `Copy-Item .env.example .env`, mở `.env` trong trình soạn thảo và điền `OPENAI_API_KEY`. Biến môi trường có ưu tiên cao hơn. Khi dựng lại store Day 06, nạp key vào PowerShell theo đoạn trên. `.env` được Git bỏ qua. Có thể đặt `OPENAI_AGENT_MODEL` nếu cần dùng model khác có tool calling và structured outputs. Bản mặc định dùng snapshot `gpt-5.4-mini-2026-03-17`. Ghi đơn giá cho ID model API thực sự trả về trong `benchmark/prices.json` trước dev nếu cần tính chi phí. [Tài liệu model](https://developers.openai.com/api/docs/models/gpt-5.4-mini) · [Tool calling](https://developers.openai.com/api/docs/guides/function-calling)

## 2. Mở Qdrant store của Day 06

Đóng các tiến trình đang mở cùng store Qdrant local. Đường dẫn mặc định khi đã có store là `$day06\starter\.day06\qdrant_openai_large`. Nếu đã ingest bằng `--store`, truyền đúng thư mục đó với `--store` cho các lệnh Day 07.

```powershell
$store = Join-Path $day06 'starter\.day06\qdrant_openai_large'
& $day07Python -m src.run prepare --day06 $day06 --store $store
& $day07Python -m src.run validate
```

Nếu store chưa có, nạp key vào biến môi trường và dựng lại bằng venv Day 06. Từ gốc Day 07, dùng đoạn sau. Đổi `python` thành đường dẫn Python của venv Day 06 nếu venv đó chưa active.

```powershell
Push-Location (Join-Path $day06 'starter')
try {
    python -m pip install -r requirements-openai.txt
    if ($LASTEXITCODE -ne 0) { throw 'Cài dependencies Day 06 thất bại.' }
    $day06Args = @('-m', 'src.run', '--provider', 'openai', '--openai-model', 'text-embedding-3-large', '--store', $store)
    python @day06Args validate --freeze
    if ($LASTEXITCODE -ne 0) { throw 'Validate Day 06 thất bại.' }
    python @day06Args ingest
    if ($LASTEXITCODE -ne 0) { throw 'Ingest Day 06 thất bại.' }
} finally {
    Pop-Location
}
& $day07Python -m src.run prepare --day06 $day06 --store $store
& $day07Python -m src.run validate
```

`prepare` lấy tên collection động từ `active.json → indexes.B`, kiểm tra model, số chiều, cosine, số point, signatures và đối chiếu toàn bộ ID/payload với bản dựng lại qua `scroll`. Ingest Day 06 ghi thời gian, token, chi phí và ingest ID trong `results/openai_large/ingest.json`.

## 3. Chạy dev, chấm và chốt

Từ gốc Day 07, sau khi chuẩn bị B và nạp key. Mặc định chạy dense, lexical, hybrid và hai biến thể lexical. `--configs` cho phép chạy từng nhóm vào các thư mục cấu hình chưa có trong cùng một lượt dev.

```powershell
$runDir = '.day07/run-01'
$selection = Join-Path $runDir 'selection.json'
& $day07Python -m src.run benchmark --split dev --mode agent --day06 $day06 --store $store --output $runDir --selection $selection
```

Mỗi thư mục `dev/<config>` có `traces.json`, `summary.json` và `reviews.template.json`. Đọc trace cùng [câu hỏi dev](benchmark/dev.json) và nguồn được citation trỏ đến. Sao chép template thành `reviews.json`, điền `reviewer`, `review_state` (`human_reviewed` hoặc `ai_assisted_unverified`) và `reason` cho mỗi câu chạy thành công.

- Câu có đáp án ghi `context_grade` và `answer_grade` theo thang 0–2 trong [BENCHMARK.md](BENCHMARK.md). Một câu từ chối nhầm nhận `answer_grade: 0`.
- Câu thiếu bằng chứng ghi `abstention_correct`. Kiểm tra agent có tìm kiếm, từ chối rõ ràng và tránh thêm khẳng định thiếu nguồn.
- Mỗi `citation_reviews` cần `supports_claim` và `reason` sau khi đối chiếu phần văn bản trong context thực gửi. Giữ ID và hash trace trong template. Lỗi hệ thống đã được ghi riêng trong trace.

```powershell
Get-ChildItem (Join-Path $runDir 'dev') -Directory | ForEach-Object {
    & $day07Python -m src.run grade --run $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "Chấm thất bại tại $($_.Name)." }
}
& $day07Python -m src.run select --output $runDir --selection $selection
```

Tiêu chí đã công bố trong [experiments.json](benchmark/experiments.json). Ưu tiên số câu chạy thành công, điểm grounded answer, citation có nguồn hỗ trợ, context đủ bằng chứng, Hit@3, rồi chi phí và latency. `select` lưu cấu hình, model thực tế, hash của đầu vào và nhãn dev. Nếu dev có lỗi kết nối, xem `setup_error.json` hoặc `traces.json`, xử lý nguyên nhân và chạy lại trong thư mục mới trước khi chốt.

## 4. Chạy test và tổng hợp

```powershell
& $day07Python -m src.run benchmark --split test --day06 $day06 --store $store --output $runDir --selection $selection
```

Lệnh dùng baseline dense và cấu hình đã chốt. Nếu dense được chọn, một lượt chạy dùng chung cho cả hai vai trò. Đọc 5 câu có đáp án và 2 câu thiếu evidence mỗi cấu hình, điền `reviews.json` theo cách ở bước dev rồi chạy `grade`.

```powershell
Get-ChildItem (Join-Path $runDir 'test') -Directory | ForEach-Object {
    & $day07Python -m src.run grade --run $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "Chấm thất bại tại $($_.Name)." }
}
```

Dùng `summary.json` cho ranking query gốc, evidence trong các ứng viên qua mọi lượt search của agent, latency, lỗi và API usage. Dùng `reviewed_metrics.json` cho chất lượng context, câu trả lời, citation, abstention và số lượt search. Các metric dùng `n`, `N` và `ratio`. Khi `N` bằng 0, `ratio` là `null`. `ranking_errors_answerable` cho biết số lỗi bị loại khỏi mẫu số ranking. `usage` lấy từ số token API trả về và ước tính theo [bảng giá](benchmark/prices.json). Khi API báo lỗi, số token và phí của request thất bại có thể không quan sát được.

## 5. Demo và xem trace

```powershell
& $day07Python -m src.run ask --day06 $day06 --store $store --config dense --output .day07/demo-answer.json 'What is required to keep a fire-and-forget task alive after scheduling it?'
& $day07Python -m src.run ask --day06 $day06 --store $store --config lexical --api-family threading --output .day07/demo-abstain.json 'On Windows, what is the default stack size in bytes for a new Python threading.Thread?'
& $day07Python -m src.run ask --day06 $day06 --store $store --config lexical --output .day07/demo-ranking.json 'Which method schedules an event-loop callback safely when I am calling from another OS thread?'
```

Xem ví dụ D01, DN01 và D05 trong [báo cáo](report/individual/2A202603483_DangLongNhat.md) để đối chiếu với trace offline. Sau khi hoàn tất dev/test, đưa `summary.json`, `traces.json`, `reviews.json`, `reviewed_metrics.json` và `selection.json` đã rà soát vào bài nộp cùng báo cáo. Khi dùng key qua môi trường, có thể kết thúc phiên bằng `Remove-Item Env:OPENAI_API_KEY`.
