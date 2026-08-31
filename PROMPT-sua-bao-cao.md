# Prompt để yêu cầu chỉnh sửa báo cáo

File này không nằm trong bundle upload. Copy phần trong khung rồi dán vào phiên
làm việc mới, thay phần `[...]` bằng yêu cầu cụ thể.

---

## Prompt đầy đủ

```
Sửa báo cáo đồ án Object Detection ở reports/report.md.

BỐI CẢNH
- reports/report.md là bản gốc duy nhất. report-prism/ chỉ là bản sao dẫn xuất
  để upload, đừng sửa trực tiếp trong đó.
- Sau khi sửa xong phải chạy CẢ HAI lệnh dựng lại, và báo kết quả kiểm tra:
      python scripts/06_report_bundle.py    -> report-prism/ (bản Markdown)
      python scripts/07_report_latex.py     -> report-latex/ + .zip (bản LaTeX)
- report-latex/report.tex được sinh tự động, KHÔNG sửa tay phần nội dung.
  Ngoại lệ duy nhất là khối \newcommand thông tin nhóm ở đầu file — nếu nó đã
  được điền thì phải giữ lại giá trị đó khi dựng lại (chép ra rồi dán vào).
- Hình nằm ở reports/figures/, sinh lại bằng:
    python scripts/05_dataset_figures.py     (4 hình mô tả dữ liệu)
    python scripts/20_evaluate.py --model all --split test --checkpoint best
                                             (3 hình đánh giá)

RÀNG BUỘC VỀ SỐ LIỆU — quan trọng nhất
- Mọi con số phải truy ngược được về một file trong repo. Nguồn ghi ở mục 9.4.
- Số kết quả lấy từ reports/results/evaluation_test_best.json và
  evaluation_test_pretrained.json. Số huấn luyện lấy từ runs/<run>/history.json.
- KHÔNG được viết ra con số nào từ trí nhớ hay từ suy luận. Nếu cần một con số
  chưa có trong file, phải chạy đo rồi mới viết.
- Nếu phát hiện số nào trong báo cáo sai so với file nguồn, sửa và nói rõ.
- Ngoại lệ duy nhất đã biết: cột "Open Images val" ở mục 6.3 không tái lập được
  vì oi_data/ đã dọn. Chỗ này đã đánh dấu rõ, giữ nguyên cách đánh dấu đó.

VĂN PHONG — đã chốt, không đổi
- Nhận xét phải kèm giải thích cơ chế, không chỉ nêu hiện tượng.
- KHÔNG xếp hạng mô hình theo kiểu "tốt nhất / nên dùng".
- KHÔNG đề xuất cách sửa chữa hay cải tiến.
- Không ẩn dụ, không nhấn mạnh thừa, không tâng bốc.
- Chấp nhận "chưa xác định được nguyên nhân" là kết luận hợp lệ, kèm danh sách
  những khả năng đã loại trừ.
- Tiếng Việt, số thập phân dùng dấu phẩy (0,2642), số nghìn dùng dấu chấm.

QUY TRÌNH
- Lệnh nào chạy quá ~30 giây thì chạy nền, ghi log ra file, và báo đường dẫn
  log ngay trong cùng lượt trả lời.
- Nếu thấy vấn đề ngoài phạm vi đang sửa: nêu một hai câu rồi tiếp tục việc
  chính, đừng tự mở điều tra phụ. Nếu một phép sửa cần quá một vòng thử-sai
  thì dừng lại và hỏi.
- Kiểm tra kỹ trước khi báo xong.

VIỆC CẦN LÀM
[Viết yêu cầu cụ thể ở đây]
```

---

## Vài mẫu yêu cầu thường gặp

Thay vào phần `VIỆC CẦN LÀM`:

**Rút ngắn**
```
Báo cáo đang 858 dòng, cần rút còn khoảng [N] dòng. Ưu tiên giữ nguyên
chương 5 (ưu nhược điểm) vì đó là phần chấm điểm nặng nhất, và giữ đủ các
bảng số liệu. Cắt ở phần diễn giải trùng lặp giữa các chương.
```

**Thêm nội dung**
```
Thêm một mục [X] vào chương [N], nội dung nói về [...]. Nếu cần số liệu mới
thì đo trước rồi mới viết, và ghi nguồn vào bảng ở mục 9.4.
```

**Đổi giọng cho hợp người chấm**
```
Chương [N] đang viết quá kỹ thuật. Viết lại cho người đọc không quen
object detection vẫn theo được, nhưng giữ nguyên toàn bộ con số và giữ
nguyên phần giải thích cơ chế.
```

**Sửa theo nhận xét**
```
Người chấm nhận xét: "[dán nhận xét vào đây]". Sửa báo cáo cho đúng nhận xét
đó. Nếu nhận xét mâu thuẫn với số liệu đang có thì nói rõ thay vì sửa số.
```

**Đổi hình**
```
Hình [tên file] chưa rõ ý ở chỗ [...]. Sửa hàm sinh hình trong
src/data/figures.py (hình dữ liệu) hoặc src/evaluation/figures.py (hình đánh
giá), sinh lại hình, rồi dựng lại bundle. Đừng sửa file PNG trực tiếp.
```

**Kiểm tra lại toàn bộ số**
```
Đối chiếu lại mọi con số trong báo cáo với file nguồn tương ứng, báo cáo chỗ
nào lệch. Không sửa gì khác.
```

---

## Hai bản để nộp, chọn theo nơi nộp

| | Dùng khi | Lệnh dựng lại |
|---|---|---|
| `report-prism/` | Nộp Markdown (Prism). 12 file, 2,2 MB | `python scripts/06_report_bundle.py` |
| `report-latex.zip` | Nộp PDF. Upload lên Overleaf rồi biên dịch | `python scripts/07_report_latex.py` |

Cả hai sinh từ cùng `reports/report.md`, nên nội dung luôn khớp nhau.

## Ghi nhớ khi upload

- Upload cả thư mục `report-prism/` chứ không riêng `report.md` — thiếu ảnh
  thì 11 hình sẽ hỏng. Trong bundle ảnh nằm cùng cấp với `report.md`; nếu Prism
  đòi cấu trúc khác thì báo lại để sửa `src/report_bundle.py`.
- Với Overleaf: sau khi upload zip phải đổi **Menu → Compiler → XeLaTeX**.
  pdfLaTeX sẽ không đặt được dấu tiếng Việt.
- Trang bìa LaTeX còn 5 chỗ cần điền (trường, khoa, học phần, giảng viên,
  tên + MSSV) trong khối `\newcommand` ở đầu `report.tex`.
- Bản gốc chất lượng cao vẫn ở `reports/figures/`, không bị bundle nào đụng tới.
