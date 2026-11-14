# Dev offline trên collection B tái tạo từ corpus

| File/thư mục | Nội dung |
|---|---|
| `benchmark_validation.json` | Hash hai split, kiểm tra anchor/scope và ID của các chunk chứa evidence |
| `day06_trace_check.json` | Đối chiếu 45 hàng top-3 dev đã công bố của Day 06 với các payload B đã dựng lại |
| `dev/lexical/` | Query gốc, TF-IDF, top-k 3, budget 1.800 ký tự |
| `dev/lexical_k6/` | Tăng riêng top-k lên 6 |
| `dev/lexical_k6_budget3000/` | Giữ top-k 6 và tăng riêng budget lên 3.000 ký tự |
| `api_status.json` | Trạng thái kiểm tra điều kiện chạy API và Qdrant |

`traces.json` lưu thứ hạng và `packed_offline` của từng câu. `summary.json` đối chiếu evidence theo nguồn và anchor với query gốc. Trường `context` biểu thị context gửi model và rỗng trong lượt offline. Hướng dẫn và phân tích ở [báo cáo](../../report/individual/2A202603483_DangLongNhat.md).
