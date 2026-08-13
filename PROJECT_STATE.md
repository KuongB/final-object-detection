# Trạng thái dự án

Cập nhật: 2026-08-14. File này tóm tắt để bắt đầu một phiên làm việc mới mà không phải đọc lại toàn bộ lịch sử.

## Đề bài

Đồ án cuối kỳ Object Detection. Hai yêu cầu:

1. **(8 điểm)** Tự dựng dataset ít nhất 5 lớp, huấn luyện và so sánh ít nhất 3 kiến trúc khác nhau, viết báo cáo phân tích ưu nhược điểm về độ chính xác, tốc độ, độ phức tạp huấn luyện, khả năng ứng dụng.
2. **(2 điểm)** Chọn mô hình tốt nhất, làm ứng dụng web cho phép tải ảnh lên và trả về kết quả phát hiện.

Nộp: mã nguồn, trọng số, dataset, báo cáo. Đóng gói một file ZIP đặt tên theo MSSV.

## Môi trường

| | |
|---|---|
| Conda env | `objdet`, Python 3.11.15, tại `C:\Users\KUONG\miniconda3\envs\objdet` |
| GPU | RTX 4060 Laptop 8 GB (sm_89) |
| PyTorch | 2.13.0+cu126 |
| Hệ điều hành | Windows 11. Chỉ chạy local, không dùng Colab |

Lưu ý Windows: dataloader dùng `spawn` chứ không `fork`, nên `workers=2` là mức ổn định nhất đã đo; tăng lên 4 từng gây lỗi và chậm hơn.

## Dữ liệu

Tập con COCO 2017, 5 lớp `apple / banana / broccoli / carrot / orange`.

| Split | Ảnh | Instance | Nguồn |
|---|---|---|---|
| train | 5.803 | 31.389 | COCO train2017 |
| val | 1.024 | 5.479 | COCO train2017 (cắt 15%) |
| test | 310 | 1.592 | COCO **val2017** |
| Tổng | 7.137 | 38.460 | |

Kích thước vật thể theo chuẩn COCO: 26,4% nhỏ · 45,1% trung bình · 28,5% lớn.

**Điểm quan trọng cần nhớ**: train và val cùng nguồn train2017 nên val **không độc lập** với train về phân phối; test lấy từ val2017 nên mới là thước đo thật. Khoảng cách val trừ test khoảng 0,02–0,03 ở cả ba mô hình. Mọi kết luận phải lấy từ test.

## Pipeline

Nguyên tắc đã chốt: **`scripts/` chỉ là launcher, mọi logic nằm trong `src/`**.

```
scripts/00_check_env.py       kiểm tra môi trường
scripts/01_download_dataset.py + 01b_fetch_images.py
scripts/02_build_dataset.py   manifest -> images/labels/annotations/data.yaml
scripts/03_verify_dataset.py  đối chiếu nhãn YOLO và COCO
scripts/04_check_images.py    kiểm tra ảnh hỏng
scripts/10_train.py           -> src/training/cli.py
scripts/20_evaluate.py        -> src/evaluation/cli.py
notebooks/01_eda.ipynb        EDA, đã chạy sẵn kết quả
notebooks/02_eval.ipynb       đánh giá 3 mô hình + so sánh 2 lượt YOLO
```

Các script `00`–`04` vẫn giữ logic inline (chưa theo pattern trên), nhưng không trùng lặp với `src/` nên chưa cần sửa.

### Lệnh

```bash
python scripts/10_train.py --model all --epochs 15
python scripts/20_evaluate.py --model all --split test --checkpoint last
python scripts/20_evaluate.py --model all --split test --checkpoint pretrained
```

`--checkpoint` nhận `last` (mặc định), `best`, hoặc `pretrained` (checkpoint COCO chưa fine-tune).

### Hợp đồng lưu trữ

Mỗi lượt train ghi `runs/<key>[_tag]/{train.log, history.json, best.pt, last.pt}`. Cuối lượt, `src/training/artifacts.py` copy best sang `weights/<key>/` và cập nhật **`weights/index.json`** — file duy nhất mà eval và webapp cần đọc.

```python
from src.models.loader import load_trained
loaded = load_trained("yolo11s")     # đọc weights/index.json
```

**Cạm bẫy đã gặp ba lần**: `promote` ghi vào `weights/<model_key>/` theo tên model chứ không theo tag, và nhiều chỗ từng giả định thư mục run trùng tên model. Đã sửa `resolve_checkpoint`, `_training_facts`, `_epoch_series` để tra `run_dir` từ index. Nếu viết code mới đụng tới `runs/`, phải lấy đường dẫn từ index.

## Kết quả

Tất cả trên tập test 310 ảnh, mAP@[.5:.95]:

| Mô hình | Checkpoint gốc | Sau huấn luyện | Thời gian train |
|---|---|---|---|
| SSDLite320-MobileNetV3 (15 ep) | 0,1057 | **0,1237** | 29,7 phút |
| YOLO11s (64 ep) | 0,2814 | 0,2714 | 89,7 phút |
| YOLO11s (15 ep) | 0,2814 | 0,2569 | 24,5 phút |
| D-FINE-N (15 ep) | 0,2715 | 0,0447 | 42,2 phút |

Độ trễ (batch 1, GPU): YOLO11s 6,6 ms · SSDLite 12,4 ms · D-FINE 21,4 ms. Lưu ý giá trị tuyệt đối dao động tới 2× tuỳ xung nhịp GPU laptop; chỉ so sánh được khi đo trong cùng một lượt.

### Ba phát hiện chính

**Checkpoint COCO gốc mạnh hơn hai trong ba bản đã fine-tune.** Cả 5 lớp đều là lớp COCO, và tập train 5.803 ảnh là tập con của chính COCO train2017 mà checkpoint đã học. Fine-tune không thêm dữ liệu mới, chỉ thu hẹp phạm vi. Chỉ SSDLite hưởng lợi vì nó có mốc xuất phát thấp nhất.

**D-FINE-N hỏng trong quá trình huấn luyện, không phải do kiến trúc.** Checkpoint gốc đạt 0,2715, ngang YOLO11s (0,2814) với chỉ 40% số tham số. Sau 15 epoch còn 0,0447. Đặc điểm: AR@100 vẫn 0,4527 (cao hơn SSDLite) nhưng điểm tin cậy cao nhất trên toàn tập test chỉ là 0,334 — dưới ngưỡng hiển thị 0,35, nên webapp sẽ không thấy hộp nào.

Đã kiểm tra và **loại trừ**: warm-start head, định dạng nhãn đưa vào loss (`class_labels` 0–4, hộp cxcywh chuẩn hoá), chuẩn hoá ảnh, toàn bộ đường ống suy luận và ánh xạ lớp. Nguyên nhân nằm trong quá trình tối ưu, **chưa xác định được**. User đã quyết định không đào tiếp.

**Val mAP và test mAP không cùng chiều.** YOLO train thêm 49 epoch: val +0,0021 nhưng test +0,0145. Hệ quả: `patience=20` dừng ở epoch 64 dựa trên val, trong khi test cho thấy phần sau epoch 44 vẫn có tác dụng.

## Việc chưa làm

| Việc | Ghi chú |
|---|---|
| **Ứng dụng web** (Yêu cầu 2) | `webapp/backend`, `webapp/frontend` đang trống hoàn toàn. `fastapi`, `uvicorn`, `jinja2`, `aiofiles` đã có trong requirements |
| **Khôi phục `docs/SETUP.md`** | Đang bị xoá. Lấy lại: `git show 9fd187a:docs/SETUP.md > docs/SETUP.md`. Repo không có README nào ghi cách dựng môi trường |
| **Thử biến thể YOLO lớn hơn** | User đã hoãn. Ứng viên: `yolo11m/l/x`, `yolo26s/m/l/x` (YOLO26 có trong ultralytics 8.4.117, NMS-free). Đã tải sẵn `yolo11m/l/x.pt`, `yolo26s.pt` ở thư mục gốc |
| **Sửa D-FINE** | Xem phần trên. Chỉ nên làm nếu còn thời gian |

### Lưu ý cấu hình cho lần train sau

- `close_mosaic=10` với 15 epoch nghĩa là tắt mosaic 2/3 thời gian; con số này được thiết kế cho run 100 epoch.
- `patience=20` không kích hoạt được nếu tổng epoch ≤ 20.
- `warmup_epochs=3` ăn hết run nếu tổng epoch ≤ 3; learning rate không kịp giảm.

## Cách làm việc user muốn

- Lệnh chạy quá ~30 giây: chạy nền, ghi log ra file, **báo đường dẫn log ngay trong cùng lượt trả lời**.
- Phát hiện vấn đề ngoài phạm vi đang làm: nêu một hai câu rồi **tiếp tục việc chính**, đừng tự mở điều tra phụ. Nếu một phép sửa cần quá một vòng thử-sai thì dừng và hỏi.
- Chấp nhận "chưa biết nguyên nhân" như kết quả hợp lệ; một danh sách đã loại trừ có hệ thống thường đủ.
- Văn phong báo cáo: nhận xét có giải thích cơ chế, **không xếp hạng, không đề xuất sửa chữa**, không ẩn dụ, không nhấn mạnh.
- Đừng để code rải rác giữa `scripts/` và `src/`.

## Lịch sử git

```
2ca582b  Add an evaluation notebook
475f951  Stop tracking the evaluation console log
4e821b2  Update the evaluation report with the pretrained baseline
71d034e  Replace the EDA notebook and make the box drawer reusable
a02b643  Rebuild the training pipeline and add evaluation
9d2d901  clean orphan file
```

Không đưa vào git: `runs/` (251 MB), trọng số `*.pt`, `reports/results/detections_*.json` (43 MB, tái tạo được).
