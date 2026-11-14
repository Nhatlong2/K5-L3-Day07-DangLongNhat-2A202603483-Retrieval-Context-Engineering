# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).

## Dữ liệu bài làm

- `corpus.jsonl`, `manifest.json`, `sources.json` và `PYTHON_DOC_LICENSE.txt` giữ nguyên từ Day 06. `page` là anchor HTML và các offset tính trong từng mục nguồn. Bản quyền và điều kiện sử dụng nằm trong [giấy phép](PYTHON_DOC_LICENSE.txt).
- Benchmark nằm ở [`benchmark/dev.json`](../benchmark/dev.json) và [`benchmark/test.json`](../benchmark/test.json). Chỉ áp filter `api_family` khi trường `filter_context` ghi scope đã cung cấp. `gold` có thể chứa các trích đoạn thay thế, mỗi đoạn đủ để tính một hit.
- `prepare` ghi `.day07/chunks.json` dạng `{provenance, chunks}` với 264 chunk đúng schema trên. `validate` ghi hash hai split, số câu và ID từng chunk chứa evidence hợp lệ.
