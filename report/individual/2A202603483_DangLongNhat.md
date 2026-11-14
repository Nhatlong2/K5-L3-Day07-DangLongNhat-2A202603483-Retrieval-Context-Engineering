# Báo cáo Lab Day 07 — Retrieval & Context Engineering

**Đặng Long Nhật · 2A202603483**

Đã chạy và chấm 85 lượt dev của 5 cấu hình cùng 14 lượt test của dense và hybrid. Hybrid được khóa bằng kết quả dev trước khi chạy test. Nhãn do ChatGPT hỗ trợ và đang chờ học viên xác nhận (`ai_assisted_unverified`).

## Dữ liệu và benchmark

| Thành phần | Dữ liệu và môi trường đã dùng |
|---|---|
| Day 07 | Mốc `c67ec6292cd6d15e6e6cbaa2c7e7260398bb5a19`, bản làm việc Lab Day 07. Các trace được kiểm tra theo SHA-256 trong summary. |
| Day 06 | Commit `7d7c7ea42a620ae88cea16dabcb58f546c778ef5`, 45 mục tài liệu Python 3.11.9, 264 chunk B. |
| Corpus và payload | SHA-256 corpus `7507144bf56c9e6c7cf89ff9deac60fe5ba7786e46a68017a6f81ceecb09a840`. SHA-256 payload B `e4a7e029da185337e2b85ed3f598e990d83c5f10013da978f03751e9d3ed510f`. [Corpus](../../data/corpus.jsonl) · [Giấy phép](../../data/PYTHON_DOC_LICENSE.txt) · [Đối chiếu trace Day 06](../../results/offline-dev/day06_trace_check.json) |
| Qdrant local | Client `1.15.1`, cosine, `text-embedding-3-large`, 3.072 chiều. Ingest ID `ef88203d5d4e4aef9caf78a77aacf394`, collection `day06_b_b9b9410547c04aeaa8a13e9b90ec7ada`. B có 264 điểm. |
| Chi phí dựng B của Day 06 | 14,827 giây, 21.468 token đầu vào, 0,00279084 USD theo log ingest gốc. Day 07 dùng lại collection này. |
| Agent và runtime | API trả model `gpt-5.4-mini-2026-03-17`. OpenAI SDK `2.8.1`, Python `3.11.9`, Windows. |

- Dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng. SHA-256 của `dev.json` là `a606a6bb903cce7b66d3f1c350355ba4f22d82751f2aad0eb6a9c1c98bd76d0a`. [Dev](../../benchmark/dev.json)
- Test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Hai cấu hình test dùng cùng bộ câu hỏi, model và collection B đã ghi trong tệp khóa. [Test](../../benchmark/test.json) · [Xác thực benchmark](../../results/offline-dev/benchmark_validation.json)
- Chấm dựa trên context cuối đã gửi, đối chiếu chunk với corpus theo ID nguồn, anchor, offset và văn bản. Mỗi citation được chấm theo khả năng hỗ trợ khẳng định chính. Hai câu thiếu bằng chứng yêu cầu tìm rồi từ chối.

## Retrieval và thử nghiệm trên dev

Dense dùng cosine exact với query embedding 3.072 chiều. Lexical dùng TF-IDF. Hybrid hợp nhất tối đa 10 ứng viên mỗi nhánh với RRF hằng số 60. Tất cả dùng cùng corpus, scope theo họ API và cutoff 3 trên câu hỏi gốc. Tool search chạy tối đa hai lượt. Lượt đầu dùng query gốc. Model có thể viết lại truy vấn một lần. Agent bỏ chunk trùng theo ID, xếp theo hạng tốt nhất và ưu tiên kết quả mới hơn khi đồng hạng. Agent đưa chunk nguyên vẹn vào context trong giới hạn ký tự. Trace lưu mỗi lượt tìm, context gửi model, câu trả lời đề xuất và trạng thái cuối. [Agent](../../src/agent.py)

| Cấu hình | Top-k / ký tự | Hit@1 gốc | Hit@3 gốc | Tìm đủ gold qua các lượt | Context đủ bằng chứng | Median | Phí dev (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dense` | 3 / 1.800 | 8/15 (53,3%) | 13/15 (86,7%) | 13/15 | 13/15 (86,7%) | 3,16 s | 0,025928 |
| `lexical` | 3 / 1.800 | 7/15 (46,7%) | 10/15 (66,7%) | 12/15 | 12/15 (80,0%) | 2,29 s | 0,027374 |
| `hybrid` | 3 / 1.800 | 8/15 (53,3%) | 13/15 (86,7%) | 15/15 | 15/15 (100,0%) | 2,73 s | 0,027933 |
| `lexical_k6` | 6 / 1.800 | 7/15 (46,7%) | 10/15 (66,7%) | 14/15 | 10/15 (66,7%) | 2,38 s | 0,028523 |
| `lexical_k6_budget3000` | 6 / 3.000 | 7/15 (46,7%) | 10/15 (66,7%) | 14/15 | 14/15 (93,3%) | 2,27 s | 0,031241 |

Mẫu số retrieval và context là 15 câu có đáp án. Cả năm cấu hình đều hoàn thành 17/17 lượt, không ghi nhận lỗi hệ thống. Cột ứng viên qua các lượt dùng khớp gold trong kết quả tìm kiếm. Context được chấm trên các đoạn thật sự gửi cho model.

| Cấu hình | Token embedding query | Token agent vào / ra | Phí embedding / agent (USD) | Tổng latency (s) |
|---|---:|---:|---:|---:|
| `dense` | 317 | 21.676 / 2.140 | 0,00004121 / 0,02588700 | 57,79 |
| `lexical` | 0 | 23.815 / 2.114 | 0,00000000 / 0,02737425 | 40,39 |
| `hybrid` | 352 | 23.719 / 2.244 | 0,00004576 / 0,02788725 | 49,73 |
| `lexical_k6` | 0 | 25.118 / 2.152 | 0,00000000 / 0,02852250 | 40,73 |
| `lexical_k6_budget3000` | 0 | 29.060 / 2.099 | 0,00000000 / 0,03124050 | 39,45 |

Tổng phí dev ước tính 0,14099847 USD cho 228,09 giây xử lý truy vấn. Bảng dùng usage do API trả và đơn giá trong [prices.json](../../benchmark/prices.json).

### Hai thay đổi cấu hình

| Thay đổi | Giả thuyết | Kết quả thực đo |
|---|---|---|
| Tăng top-k lexical 3 → 6, giữ 1.800 ký tự | Thêm ứng viên ngoài ba hạng đầu có thể đưa đoạn bị bỏ sót vào context. | Ứng viên qua các lượt chứa đủ gold tăng 12/15 → 14/15. Context đủ bằng chứng giảm 12/15 → 10/15 vì cách đóng gói và câu truy vấn viết lại khác nhau. Grounded answer giảm 0,60 → 0,53/2. |
| Giữ top-k 6, tăng context 1.800 → 3.000 ký tự | Budget lớn hơn có thể giữ những chunk đã tìm được ở hạng thấp. | Context đủ bằng chứng tăng 10/15 → 14/15, trung bình từ 1.651 lên 2.819 ký tự. Grounded answer tăng 0,53 → 0,60/2. Lượt trả lời giảm từ 7 xuống 5. Phí tăng từ 0,028523 lên 0,031241 USD. |

Hai thử nghiệm giữ nguyên cutoff đánh giá top-3. Mỗi cấu hình chạy một lần trên dev, nên mức dao động do quá trình sinh và viết lại truy vấn chưa được đo qua nhiều lượt chạy.

### Phân tích lỗi từ trace

| Query và cấu hình | Quan sát |
|---|---|
| `D15`, cả 5 cấu hình | Context đều có chunk `269fac4b-7441-5a02-beb5-802eecd0f641` nói về batch callback và lần chạy loop kế tiếp. Cả 5 bản đề xuất trả lời đúng ý chính và đặt ID trong mảng `citations`, nhưng thiếu ID trong văn bản. Agent trả `invalid_citations` và người dùng nhận abstain. [Nguồn](https://docs.python.org/release/3.11.9/library/asyncio-eventloop.html#running-and-stopping-the-loop) |
| `D05`, `lexical_k6` | Đoạn `e17d0cdc-e9b0-566b-86ea-cdf3685a45d7` nằm hạng 4, bị bỏ ở budget 1.800. Agent nhầm sang API gửi coroutine. Budget 3.000 đưa đoạn đúng vào context, song câu trả lời vẫn bị từ chối do thiếu citation trong văn bản. Hybrid đưa đoạn đúng vào hạng 2 và trả lời có citation. [Nguồn](https://docs.python.org/release/3.11.9/library/asyncio-eventloop.html#scheduling-callbacks) |
| `D06`, `lexical` và `lexical_k6` | Cả hai dẫn đoạn nói về batch sau `stop()`. Đoạn xác nhận `close()` bỏ callback đang chờ đứng hạng 5 với top-k 6. Tăng budget lên 3.000 đưa đoạn này vào context và trả lời đúng. [Nguồn](https://docs.python.org/release/3.11.9/library/asyncio-eventloop.html#running-and-stopping-the-loop) |
| `D07`, `hybrid` | Top-3 truy vấn gốc thiếu bằng chứng về thứ tự chờ. Lần hai thêm `asyncio.Lock` và `FIFO fair order`, đưa đoạn xác nhận tính công bằng vào context và trả lời đúng. `lexical_k6` cũng tìm lại nhưng chỉ dẫn đoạn mô tả khóa loại trừ, thiếu căn cứ về FIFO. [Nguồn](https://docs.python.org/release/3.11.9/library/asyncio-sync.html#lock) |

Trong 35 lượt `invalid_citations`, trường `citations` chứa ID trong context và văn bản thiếu citation inline. 32/35 lượt có đoạn gold trong context. Điểm dùng đầu ra đã gửi cho người dùng, nên cả 35 lượt được tính là từ chối nhầm. 

## Chấm agent và citation trên dev

| Cấu hình | Trả lời / từ chối / lỗi | Tool use | Số lượt tìm 1 / 2 | Context 0 / 1 / 2 | Grounded 0 / 1 / 2 | Điểm / 2 | Citation hỗ trợ | Câu trả lời có citation | Từ chối đúng · nhầm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense` | 7 / 10 / 0 | 17/17 | 15 / 2 | 1 / 1 / 13 | 8 / 0 / 7 | 0,93 | 8/8 (100,0%) | 7/7 | 2/2 · 8/15 |
| `lexical` | 6 / 11 / 0 | 17/17 | 13 / 4 | 2 / 1 / 12 | 10 / 1 / 4 | 0,60 | 5/6 (83,3%) | 6/6 | 2/2 · 9/15 |
| `hybrid` | 11 / 6 / 0 | 17/17 | 13 / 4 | 0 / 0 / 15 | 4 / 1 / 10 | 1,40 | 11/11 (100,0%) | 11/11 | 2/2 · 4/15 |
| `lexical_k6` | 7 / 10 / 0 | 17/17 | 12 / 5 | 2 / 3 / 10 | 11 / 0 / 4 | 0,53 | 4/7 (57,1%) | 7/7 | 2/2 · 8/15 |
| `lexical_k6_budget3000` | 5 / 12 / 0 | 17/17 | 14 / 3 | 1 / 0 / 14 | 10 / 1 / 4 | 0,60 | 5/5 (100,0%) | 5/5 | 2/2 · 10/15 |

Ví dụ demo `D01` ở hybrid trả lời cần giữ tham chiếu đến task vì event loop giữ tham chiếu yếu, với citation tới mục tạo task. `DN01` ở hybrid gọi search hai lần rồi từ chối vì corpus thiếu kích thước stack mặc định của Windows. `D15` cho thấy trường hợp có bằng chứng nhưng bị từ chối do citation inline. [Trace hybrid](../../results/agent-dev/dev/hybrid/traces.json)

## Chốt cấu hình trên dev

Theo tiêu chí trong `experiments.json`, hybrid đứng đầu dev với điểm grounded 1,40/2, context đủ 15/15 và citation hỗ trợ 11/11. Baseline là dense. Cả hai cấu hình test dùng top-k 3, budget 1.800 ký tự và tối đa hai lượt search.

- Khóa lúc **12:12:30 14/11/2026 (UTC+7)**. SHA-256 cấu hình hybrid `b17a4e3764d8b62d4d1fec373e5c587f7fdc36ed5e5e53a2af86904563c6d289`. SHA-256 bản khóa `c68908d408171d5e9335aba859a512381336b8a48e0033096c81b6e75823261c`. [Tệp khóa](../../results/agent-dev/selection.json)
- Test chạy từ **12:20:07 đến 12:20:52 14/11/2026 (UTC+7)**. Hai summary khớp tệp khóa về hash đầu vào, model, collection B và benchmark. Hash trace và nhãn dev trong tệp khóa khớp dữ liệu bàn giao. [Dense](../../results/agent-dev/test/dense/summary.json) · [Hybrid](../../results/agent-dev/test/hybrid/summary.json)

## Kết quả test độc lập

| Cấu hình | Hit@1 gốc | Hit@3 gốc | Gold qua mọi lượt tìm | Context đủ bằng chứng | Context 0 / 1 / 2 | Grounded 0 / 1 / 2 | Điểm / 2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dense` | 4/5 (80%) | 4/5 (80%) | 4/5 (80%) | 4/5 (80%) | 0 / 1 / 4 | 5 / 0 / 0 | 0,00 |
| `hybrid` | 4/5 (80%) | 4/5 (80%) | 5/5 (100%) | 5/5 (100%) | 0 / 0 / 5 | 4 / 0 / 1 | 0,40 |

| Cấu hình | Trả lời / từ chối / lỗi | Tool use | Số lượt tìm 1 / 2 | Citation hỗ trợ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dense` | 1 / 6 / 0 | 7/7 (100%) | 6 / 1 | 0/2 (0%) | 1/1 (100%) | 2/2 (100%) | 4/5 (80%) |
| `hybrid` | 1 / 6 / 0 | 7/7 (100%) | 5 / 2 | 1/1 (100%) | 1/1 (100%) | 2/2 (100%) | 4/5 (80%) |

| Cấu hình | Context TB / tối đa (ký tự) | Token embedding query | Token agent vào / ra | Phí embedding / agent (USD) | Median / tổng (giây) |
|---|---:|---:|---:|---:|---:|
| `dense` | 1.527 / 1.754 | 158 | 9.206 / 767 | 0,00002054 / 0,01035600 | 3,69 / 25,76 |
| `hybrid` | 1.512 / 1.784 | 179 | 10.134 / 697 | 0,00002327 / 0,01073700 | 2,55 / 19,21 |

Tổng phí test ước tính **0,02113681 USD**. Cả dev và test của Day 07 dùng **0,16213528 USD**. [Đơn giá](../../benchmark/prices.json)

### Đối chiếu từng câu test

| Câu | Dense | Hybrid | Bằng chứng |
|---|---|---|---|
| `T07-01` | Context 2, answer 0 | Context 2, answer 0 | Cả hai đề xuất cờ Event ban đầu là false. ID chỉ xuất hiện trong mảng citation. [Event](https://docs.python.org/release/3.11.9/library/asyncio-sync.html#event) |
| `T07-02` | Context 2, answer 0 | Context 2, answer 0 | Cả hai đề xuất lấy ưu tiên thấp nhất trước. Thiếu citation inline khiến đầu ra thành abstain. [PriorityQueue](https://docs.python.org/release/3.11.9/library/asyncio-queue.html#priority-queue) |
| `T07-03` | Context 1, answer 0 | Context 2, answer 2 | Dense gán ghi chú InvalidStateError của `set_exception()` cho `add_done_callback()`. Hybrid viết lại query, tìm được chunk `87333265` và trả lời đúng callback chạy ngay. [Future Objects](https://docs.python.org/release/3.11.9/library/concurrent.futures.html#future-objects) |
| `T07-04` | Context 2, answer 0 | Context 2, answer 0 | Cả hai tìm đúng trách nhiệm của `loop_factory`. Thiếu citation inline khiến đầu ra thành abstain. [Runner](https://docs.python.org/release/3.11.9/library/asyncio-runner.html#runner-context-manager) |
| `T07-05` | Context 2, answer 0 | Context 2, answer 0 | Cả hai đề xuất đúng 100 ms và thiếu citation inline. [Debug mode](https://docs.python.org/release/3.11.9/library/asyncio-eventloop.html#enabling-debug-mode) |
| `TN07-01` | Từ chối đúng, 1 lượt tìm | Từ chối đúng, 2 lượt tìm | Corpus thiếu giới hạn số pending callbacks của `call_soon`. [Scheduling callbacks](https://docs.python.org/release/3.11.9/library/asyncio-eventloop.html#scheduling-callbacks) |
| `TN07-02` | Từ chối đúng, 2 lượt tìm | Từ chối đúng, 1 lượt tìm | Corpus thiếu số pickle protocol dùng cho đối số đã submit. [ProcessPoolExecutor](https://docs.python.org/release/3.11.9/library/concurrent.futures.html#processpoolexecutor) |

Hybrid tăng tỷ lệ context đủ bằng chứng từ 4/5 lên 5/5 qua lần tìm lại ở `T07-03`. Cả hai có Hit@3 gốc 4/5. Bốn câu có gold trong context bị chuyển thành abstain ở mỗi cấu hình vì model chỉ đặt ID vào mảng `citations`. Tỷ lệ citation của hybrid dựa trên một citation thực sự xuất hiện trong câu trả lời cuối. Test có bảy câu và một lần chạy mỗi cấu hình.

## Tái lập và đối chiếu bài nộp

Lệnh benchmark test đã dùng `python -m src.run benchmark --split test --mode agent` cùng `--day06`, `--store`, `--output` và `--selection`. Lệnh `grade` tạo lại chỉ số từ trace và nhãn. `python -m unittest discover -s tests -v` đạt 25/25 trong bản mã đối chiếu. [Hướng dẫn PowerShell](../../README.md#4-chạy-test-và-tổng-hợp)

Để tái lập đúng bản chạy Windows, đối chiếu `requirements.txt` với SHA-256 `bc4a1ebb4be916cae7e3a194ee182f66568e7c2a358ab0b0f16285fe61fef6c2` trong tệp khóa. Bản mã đối chiếu có hash `8ef79c622e010e5db1e2f8c2089422d5a6c355b48f9c5d843dfb818cdc4707a6` ở file này. Các hash mã nguồn và cấu hình còn lại trùng. Toàn bộ hash đầu vào trong dev, tệp khóa và test trên Windows trùng nhau.

| Hạng mục rubric / bài nộp | Bằng chứng |
|---|---|
| Dữ liệu và benchmark · 13 điểm | Corpus có giấy phép PSF, 264 chunk, 20 câu có đáp án và 4 câu thiếu evidence trên hai split. [Manifest](../../data/manifest.json) · [Tham chiếu Day 06](../../benchmark/day06_reference.json) · [Benchmark](../../benchmark/) |
| Retrieval · 17 điểm | Dense, TF-IDF và hybrid RRF. Trace có rank, score, ID và phạm vi lọc. [Retriever](../../src/retrieval.py) · [Dev](../../results/agent-dev/dev/) |
| Agent và context · 17 điểm | Tool calling thật, giới hạn search và context, citation và abstention theo câu. [Agent](../../src/agent.py) · [Test](../../results/agent-dev/test/) |
| Đánh giá · 17 điểm | 85 lượt dev và 14 lượt test đã chấm. Nhãn có lý do theo câu và từng citation. [Nhãn và chỉ số](../../results/agent-dev/) · [Mã chấm](../../src/grading.py) |
| Thử nghiệm và cải thiện · 13 điểm | Thử top-k và budget, phân tích D05/D06/D07/D15, khóa theo dev trước khi chạy test. [Cấu hình](../../benchmark/experiments.json) · [Tệp khóa](../../results/agent-dev/selection.json) |
| Tái lập và báo cáo · 8 điểm | Lệnh cài đặt, dữ liệu, tệp khóa, trace, chi phí và giới hạn. [README](../../README.md) · [Unit tests](../../tests/) |
| Hồ sơ nộp | Rà nhãn có AI hỗ trợ, kiểm tra quyền xem repo cá nhân và nộp link theo hướng dẫn giảng viên. [SUBMISSION](../../SUBMISSION.md) · [RUBRIC](../../RUBRIC.md) |

Các phần cập nhật lần này gồm `README.md`, báo cáo cá nhân, `results/agent-dev/dev/`, `results/agent-dev/test/` và `results/agent-dev/selection.json`. Phạm vi đánh giá là phần cơ bản của lab.
