# Trạng thái dự án

Cập nhật: 2026-08-21. File này tóm tắt để bắt đầu một phiên làm việc mới mà không phải đọc lại lịch sử.

## Đề bài

Đồ án cuối kỳ Object Detection. Hai yêu cầu:

1. **(8 điểm)** Tự dựng dataset ít nhất 5 lớp, huấn luyện và so sánh ít nhất 3 kiến trúc khác nhau, viết báo cáo phân tích ưu nhược điểm về độ chính xác, tốc độ, độ phức tạp huấn luyện, khả năng ứng dụng.
2. **(2 điểm)** Chọn mô hình tốt nhất, làm ứng dụng web cho phép tải ảnh lên và trả về kết quả phát hiện.

Nộp: mã nguồn, trọng số, dataset, báo cáo. Đóng gói một file ZIP đặt tên theo MSSV.

## Việc đang làm

**Yêu cầu 1: xong.** Bốn mô hình đã huấn luyện và đánh giá, báo cáo ở `reports/evaluation.md`.

**Yêu cầu 2: chưa bắt đầu.** `webapp/backend` và `webapp/frontend` đang trống. Kế hoạch ở mục cuối file này.

## Môi trường

| | |
|---|---|
| Conda env | `objdet`, Python 3.11.15, tại `C:\Users\KUONG\miniconda3\envs\objdet` |
| GPU | RTX 4060 Laptop 8 GB (sm_89) |
| PyTorch | 2.13.0+cu126, ultralytics 8.4.117 |
| Hệ điều hành | Windows 11 |
| GitHub | `https://github.com/KuongB/final-object-detection` |

Lưu ý Windows: dataloader dùng `spawn` chứ không `fork`, nên `workers=2` là mức ổn định nhất đã đo. Trên Linux nâng lên 8 được.

## Dữ liệu

Tập con COCO 2017, 5 lớp `apple / banana / broccoli / carrot / orange`.

| Split | Ảnh | Instance | Nguồn |
|---|---|---|---|
| train | 5.803 | 31.389 | COCO train2017 |
| val | 1.024 | 5.479 | COCO train2017 (cắt 15%) |
| test | 310 | 1.592 | COCO **val2017** |

**COCO đã cạn kiệt**: đã đối chiếu với `instances_train2017.json` — mọi ảnh COCO-2017 chứa 5 lớp này đều đã nằm trong dataset, không có cap hay sampling nào.

**Điểm quan trọng**: train và val cùng nguồn train2017 nên val **không độc lập** với train về phân phối; test lấy từ val2017 nên mới là thước đo thật. Mọi kết luận phải lấy từ test.

## Kết quả — tất cả trên tập test COCO (310 ảnh)

| Mô hình | Tham số | Gốc (chưa fine-tune) | Sau fine-tune | Độ trễ |
|---|---|---|---|---|
| **YOLO26m** | 20,4 M | **0,3063** | 0,1353 | 16,9 ms |
| YOLO11s (64 ep) | 9,41 M | 0,2814 | 0,2642 | 13,6 ms |
| SSDLite320 | 2,26 M | 0,1057 | **0,1238** | 25,6 ms |
| D-FINE-N | 3,72 M | 0,2715 | 0,0928 | 42,5 ms |

Cột "gốc" là checkpoint COCO công khai, lọc 5 lớp từ 80. Cột "sau fine-tune" dùng `--checkpoint best`.

**Điểm cao nhất toàn dự án là `yolo26m.pt` gốc: 0,3063.** Không mô hình nào sau khi fine-tune vượt được nó.

### Ba phát hiện chính

**1. Fine-tune trên tập con của dữ liệu tiền huấn luyện làm mô hình tệ đi.** Đã chứng minh ba lần: YOLO11s (−0,017), D-FINE (−0,179), và YOLO26m trên Open Images (−0,171). Chỉ SSDLite khá lên (+0,018) vì nó có mốc xuất phát thấp nhất nên còn dư địa chuyên biệt hoá.

**2. Open Images không cứu được.** Đã fine-tune YOLO26m trên 2.755 ảnh Open Images (dữ liệu checkpoint COCO chưa từng thấy). Kết quả: thua bản gốc trên **cả hai** tập.

| | COCO test | Open Images val |
|---|---|---|
| yolo26m gốc | **0,3063** | **0,4659** |
| yolo26m fine-tune | 0,1353 | 0,3671 |

82 ảnh Open Images val đó chính là tập đã dùng để chọn `best.pt`, tức phép đo **thiên vị có lợi cho bản fine-tune**, mà nó vẫn thua 0,10 mAP. AR@100 cũng thấp hơn (0,63 so với 0,76) nên nó bỏ sót nhiều hơn chứ không chỉ chấm điểm dè dặt.

**3. Val mAP và test mAP không cùng chiều.** YOLO11s train thêm 49 epoch: val +0,0021 nhưng test +0,0145. Hệ quả: `patience=20` dừng ở epoch 64 dựa trên val, trong khi test cho thấy phần sau epoch 44 vẫn có tác dụng.

### D-FINE — vấn đề chưa giải quyết

Checkpoint gốc đạt 0,2715 (ngang YOLO11s với 40% số tham số), sau 15 epoch còn 0,0928. Đặc điểm: AR@100 vẫn 0,4527 nhưng điểm tin cậy cao nhất trên toàn tập test chỉ 0,334 — dưới ngưỡng hiển thị 0,35.

Đã kiểm tra và **loại trừ**: warm-start head, định dạng nhãn đưa vào loss (`class_labels` 0–4, hộp cxcywh chuẩn hoá), chuẩn hoá ảnh, đường ống suy luận và ánh xạ lớp. Nguyên nhân nằm trong quá trình tối ưu, **chưa xác định được**. User đã quyết định không đào tiếp.

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
notebooks/02_eval.ipynb       đánh giá 3 mô hình + so sánh 2 lượt YOLO11s
notebooks/04_kaggle_yolo26m_openimages.ipynb      bản chạy được trên Kaggle
notebooks/04_kaggle_yolo26m_openimages_run.ipynb  bản đã chạy, giữ log
```

### Lệnh

```bash
python scripts/10_train.py --model all --epochs 15
python scripts/20_evaluate.py --model all --split test --checkpoint last
python scripts/20_evaluate.py --model all --split test --checkpoint pretrained
```

`--checkpoint` nhận `last`, `best`, hoặc `pretrained`. `--model` nhận key bất kỳ có trong `weights/index.json`.

### Hợp đồng lưu trữ

Mỗi lượt train ghi `runs/<key>[_tag]/{train.log, history.json, best.pt, last.pt}`. Cuối lượt, `src/training/artifacts.py::promote` copy sang `weights/<key>/` và cập nhật `weights/index.json` — file duy nhất mà eval và webapp cần đọc.

```python
from src.models.loader import load_trained
loaded = load_trained("yolo26m")        # doc weights/index.json
```

Cho checkpoint gốc 80 lớp, dùng `load_pretrained()` trong `src/evaluation/runner.py` — nó trả về cả model lẫn bảng ánh xạ 80 lớp về 5 lớp.

### Cạm bẫy đã gặp nhiều lần

**Thư mục run không trùng tên model.** Nhiều chỗ từng giả định `runs/<model_key>`, sai kể từ khi có `--tag`. Đã sửa `resolve_checkpoint`, `_training_facts`, `_epoch_series`. Code mới đụng tới `runs/` phải lấy `run_dir` từ index.

**`promote` ghi đè toàn bộ `index.json`.** Chép file index từ máy khác đè lên sẽ xoá mất các entry khác. Phải chạy `promote` tại chỗ.

**`yolo26m` cố ý KHÔNG có trong `TRAINERS`.** `src/training/train_yolo.py` gán cứng `MODEL_KEY = "yolo11s"`; nếu thêm vào `TRAINERS`, lệnh `--model yolo26m` sẽ âm thầm train nhầm model rồi promote sai tên. Hiện tại nó báo lỗi rõ ràng. Muốn train YOLO26 tại máy thì phải tham số hoá `run()` trước.

## Trạng thái file

```
runs/ssdlite        runs/yolo11s (15 ep)   runs/yolo11s_deep (64 ep)
runs/dfine          runs/yolo26m_oi
weights/{ssdlite,yolo11s,dfine,yolo26m}/best.pt + index.json
reports/evaluation.md, figures/*.png, results/evaluation_test_{best,last,pretrained}.json
yolo11s.pt, yolo26m.pt        <- checkpoint goc, dung cho --checkpoint pretrained
```

**Không xoá `runs/yolo11s`** dù index đã trỏ sang `yolo11s_deep`: notebook `02_eval` đọc cả hai để dựng biểu đồ so sánh 15 với 64 epoch, và mục 4.3 báo cáo dựa vào đó.

Đã dọn (2026-08-21, ~385 MB): checkpoint `yolo11l/m/x`, `yolo26n/s` từ phần so sánh model zoo đã huỷ; `reports/results/detections_*.json` (dữ liệu thô, tái tạo được); bản sao trùng byte `runs/yolo26m_oi/weights/` và `weights/yolo11s_e15/`.

Không đưa vào git: `runs/`, `data/`, `*.pt`, `reports/results/detections_*.json`.

## Huấn luyện trên Kaggle

Notebook `04_kaggle_yolo26m_openimages.ipynb` tự chứa, tải Open Images trực tiếp từ CSV của Google và bucket S3 công khai.

Bốn điều phải nhớ, đều do chạy thử mà biết:

- **Accelerator phải là `GPU T4 x2`, không phải P100.** PyTorch trên Kaggle không còn hỗ trợ sm_60. Notebook có ô kiểm tra dừng ngay nếu GPU không tương thích.
- **Không dùng `fiftyone` trên Kaggle.** Nó nâng `pillow` lên trạng thái hỏng: `ImportError: cannot import name '_Ink' from 'PIL._typing'`.
- **Đừng tắt Internet giữa chừng** — Kaggle khởi động lại kernel và giết tiến trình train. `model.train()` còn tải thêm một model nano cho phép kiểm tra AMP sau khi dữ liệu đã tải xong.
- **`optimizer="auto"` vứt bỏ `lr0` và `momentum`.** Log ghi rõ `ignoring 'lr0=0.001' and 'momentum=0.948'`, và nó chọn AdamW chứ không phải MuSGD. Muốn dùng đúng công thức phải đặt thẳng tên optimizer.

Kết quả lượt đã chạy: 60/60 epoch, 123,3 phút trên Tesla T4, VRAM đỉnh 9,04 GB (còn dư, batch nâng lên 24–32 được), 2.755 ảnh train / 567 val.

## Kế hoạch ứng dụng web (Yêu cầu 2)

**Model đã chốt: `yolo26m.pt` gốc kèm lọc `classes=[46,47,49,50,51]`** rồi ánh xạ về 5 lớp. Lý do ở mục "Kết quả" — nó đạt điểm cao nhất trong mọi phương án đã đo.

### Giai đoạn 1 — Tải ảnh lên và nhận diện

```
src/webapp/detector.py      nap model, suy luan, ve hop   (logic)
webapp/backend/app.py       FastAPI, 2 route              (mong)
webapp/frontend/index.html  mot trang, JS thuan
scripts/30_serve.py         launcher
```

| Route | Việc |
|---|---|
| `GET /` | Trả trang HTML |
| `POST /detect` | Nhận ảnh + ngưỡng, trả JSON: ảnh đã vẽ hộp (base64), danh sách vật thể, số lượng theo lớp |

- Model nạp **một lần** lúc khởi động qua `lifespan` của FastAPI
- Tái dùng `load_pretrained()` trong `src/evaluation/runner.py`
- Vẽ hộp bằng `CLASS_COLORS_RGB` trong `src/config.py` để màu khớp báo cáo và notebook
- Giao diện: kéo thả ảnh, ảnh gốc và ảnh nhận diện cạnh nhau, bảng đếm theo lớp, thanh trượt ngưỡng (mặc định 0,35), hiện thời gian suy luận
- JS thuần, không cần bước build — người chấm chỉ cần `uvicorn` là chạy được

Gói cần thiết đã có trong `requirements.txt`: `fastapi`, `uvicorn[standard]`, `python-multipart`, `jinja2`, `aiofiles`.

### Giai đoạn 2 — Camera laptop (làm sau)

Trình duyệt lấy camera qua `getUserMedia`, chụp khung hình mỗi ~80 ms, gửi lên `/detect`. Tái dùng toàn bộ backend, đạt khoảng 12–15 khung/giây trên máy local.

Hướng thay thế: xuất ONNX và chạy hẳn trong trình duyệt bằng `onnxruntime-web`. YOLO26 **không cần NMS** nên hậu xử lý trong JS chỉ là lọc theo ngưỡng — khả thi hơn nhiều so với YOLO11.

## Cách làm việc user muốn

- Lệnh chạy quá ~30 giây: chạy nền, ghi log ra file, **báo đường dẫn log ngay trong cùng lượt trả lời**.
- Phát hiện vấn đề ngoài phạm vi đang làm: nêu một hai câu rồi **tiếp tục việc chính**. Nếu một phép sửa cần quá **một vòng** thử-sai thì dừng và hỏi.
- Chấp nhận "chưa biết nguyên nhân" như kết quả hợp lệ; một danh sách đã loại trừ có hệ thống thường đủ.
- Văn phong báo cáo: nhận xét có giải thích cơ chế, **không xếp hạng, không đề xuất sửa chữa**, không ẩn dụ, không nhấn mạnh.
- Đừng để code rải rác giữa `scripts/` và `src/`.
- **Kiểm tra kỹ trước khi giao**, đừng để user phải thử đi thử lại.

## Lịch sử git

```
65397a9  Record the val/test gap, and add a project state summary
2ca582b  Add an evaluation notebook
475f951  Stop tracking the evaluation console log
4e821b2  Update the evaluation report with the pretrained baseline
71d034e  Replace the EDA notebook and make the box drawer reusable
a02b643  Rebuild the training pipeline and add evaluation
9d2d901  clean orphan file
```

Chưa commit tại thời điểm cập nhật file này: sửa markdown hai notebook EDA/Eval, hai notebook Kaggle mới, entry `yolo26m` trong `src/config.py`, bỏ ràng buộc `choices` trong `src/evaluation/cli.py`, và kết quả đánh giá mới.

## Việc chưa làm

| Việc | Ghi chú |
|---|---|
| **Ứng dụng web** | Yêu cầu 2, kế hoạch ở trên |
| **Khôi phục `docs/SETUP.md`** | `git show 9fd187a:docs/SETUP.md > docs/SETUP.md`. Repo không có README ghi cách dựng môi trường |
| **Cập nhật `reports/evaluation.md`** | Chưa có phần YOLO26m. Đã bàn: làm thành mục cải tiến riêng, giữ nguyên phần so sánh 3 kiến trúc |
| **Sửa D-FINE** | Chỉ nếu còn thời gian |
