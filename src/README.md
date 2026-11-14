# Mã Lab Day 07

| File | Trách nhiệm |
|---|---|
| `run.py` | Lệnh `prepare`, `validate`, `benchmark`, `grade`, `select`, `ask` |
| `day06.py` | Tái tạo B bằng code Day 06, kiểm tra hash, đối chiếu payload và truy vấn Qdrant local |
| `retrieval.py` | Dense exact cosine, TF-IDF, RRF với thứ hạng từng nhánh |
| `agent.py` | Tool calling, đóng gói context, citation và điều kiện dừng |
| `provider.py` | Kết nối OpenAI, usage và chi phí |
| `experiment.py`, `grading.py` | Chạy split, trace, chấm evidence và chốt cấu hình trước test |

Các lệnh PowerShell nằm ở [README gốc](../README.md). Mỗi công cụ runtime nhận query, scope và tham số thực thi. Benchmark gold được dùng ở bước xác thực/chấm trong `grading.py`.
