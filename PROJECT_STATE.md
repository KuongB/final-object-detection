# Trạng thái dự án

Cập nhật: 2026-08-23. File này tóm tắt để bắt đầu một phiên làm việc mới mà không phải đọc lại lịch sử.

## Đề bài

Đồ án cuối kỳ Object Detection. Hai yêu cầu:

1. **(8 điểm)** Tự dựng dataset ít nhất 5 lớp, huấn luyện và so sánh ít nhất 3 kiến trúc khác nhau, viết báo cáo phân tích ưu nhược điểm về độ chính xác, tốc độ, độ phức tạp huấn luyện, khả năng ứng dụng.
2. **(2 điểm)** Chọn mô hình tốt nhất, làm ứng dụng web cho phép tải ảnh lên và trả về kết quả phát hiện.

Nộp: mã nguồn, trọng số, dataset, báo cáo. Đóng gói một file ZIP đặt tên theo MSSV.

## Việc đang làm

**Yêu cầu 1: xong.** Bốn mô hình đã huấn luyện và đánh giá. Báo cáo nộp ở `reports/report.md` — một file phủ cả hai yêu cầu, 9 chương. `reports/evaluation.md` đã bị hấp thụ hoàn toàn vào đó và xoá.

**Yêu cầu 2: xong.** Ứng dụng web chạy được, hai chức năng: tải ảnh lên, và camera thời gian thực. Chi tiết kỹ thuật ở mục cuối file này; phần viết cho người chấm ở chương 7 của `reports/report.md`.

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
scripts/30_serve.py           -> src/webapp/cli.py
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
python scripts/30_serve.py                    # rồi mở http://localhost:8000
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
reports/report.md, figures/*.png, results/evaluation_test_{best,last,pretrained}.json
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

## Ứng dụng web (Yêu cầu 2) — đã xong

**Model: `yolo26m.pt` gốc**, lọc 5 lớp từ 80 rồi ánh xạ về `apple/banana/broccoli/carrot/orange`. Lý do ở mục "Kết quả" — nó đạt điểm cao nhất trong mọi phương án đã đo. Giao diện tiếng Anh, không có dropdown chọn model.

### Chạy

```bash
python scripts/30_serve.py                    # http://localhost:8000
python scripts/30_serve.py --port 8080 --device cpu
```

Phải mở bằng **`localhost`**. Trình duyệt chỉ cấp camera trong secure context, nên vào bằng IP LAN (`http://192.168.x.x:8000`) thì tab Live sẽ bị chặn — trang tự hiện cảnh báo khi gặp trường hợp đó.

### File

```
src/webapp/detector.py    nạp model, suy luận, đếm số lượng, bám vết
src/webapp/session.py     phiên camera: đếm vật thể riêng biệt, ghi JSON + CSV
src/webapp/drawing.py     vẽ hộp bằng PIL + mã hoá data URL
src/webapp/app.py         FastAPI, lifespan + 4 route
src/webapp/cli.py         argparse + uvicorn
scripts/30_serve.py       launcher
webapp/frontend/          index.html + static/{app.js,style.css}, JS thuần, không cần build
webapp/samples/           4 ảnh test (3 ngang, 1 dọc), để chạy demo được khi chưa giải nén data/
webapp/sessions/          bản ghi từng phiên camera, sinh lúc chạy, không vào git
```

Lệch với phác thảo cũ: FastAPI nằm ở `src/webapp/app.py` chứ không phải `webapp/backend/app.py`, vì route là logic. Thư mục `webapp/backend/` rỗng đã bỏ cùng hai dòng `.gitignore` tương ứng.

| Route | Việc |
|---|---|
| `GET /` | trang HTML |
| `GET /api/meta` | 5 lớp, bảng màu, ngưỡng mặc định, thông tin model, danh sách ảnh mẫu |
| `POST /api/detect` | ảnh + ngưỡng → JSON kèm ảnh đã vẽ hộp (data URL), danh sách vật thể, số lượng theo lớp |
| `WS /ws/detect` | khung JPEG vào, **chỉ toạ độ** ra — trình duyệt tự vẽ overlay lên canvas |
| `GET /api/sessions` | danh sách phiên đã lưu, mới nhất trước |
| `GET /sessions/*` | file JSON và `sessions.csv` phục vụ trực tiếp |

Model nạp một lần trong `lifespan`, dùng lại `load_pretrained()` ở `src/evaluation/runner.py` nên bảng ánh xạ COCO-80 → 5 lớp chỉ tồn tại một chỗ. Màu lấy từ `CLASS_COLORS_RGB` và phát qua `/api/meta`, nên overlay của trình duyệt, ảnh server vẽ và hình trong báo cáo không thể lệch nhau. Suy luận là lời gọi chặn nên cả hai route đều đi qua `run_in_threadpool`.

### Đếm theo phiên camera

Một phiên bắt đầu khi bấm Start camera và kết thúc khi bấm Stop. **Socket chính là phiên** — không cần đồng bộ thêm gì, và phiên kết thúc vì đóng tab hay rút dây vẫn được ghi, vì đường ngắt kết nối chạy chung hàm `close()`.

**Đếm theo id riêng biệt của tracker, không phải cộng số đếm từng khung.** Cộng từng khung thì ở 30 fps, một quả táo đứng yên 10 giây sẽ thành 300 — con số đó đo tốc độ khung hình chứ không đo trái cây. Tracker cấp id khi vật thể xuất hiện và giữ id đó suốt lúc nó còn trong khung; rời đi rồi quay lại là id mới, tức quả mới. Tally là phép hợp tập hợp id, nên cùng một quả xuất hiện trong 300 khung vẫn chỉ vào tập một lần.

Đã đo đúng hành vi này:

| Kịch bản | Kết quả |
|---|---|
| 60 khung y hệt, cảnh đứng yên | tổng phiên đứng nguyên ở 11 |
| Rời khung 45 khung rồi quay lại | 11 → 22 |
| Mở rồi đóng, không khung nào | không ghi file |
| Ngắt đột ngột, không báo trước | vẫn ghi đủ 25 khung |

Lưu ở `webapp/sessions/`: mỗi phiên một `session_<id>.json`, và một dòng nối thêm vào `sessions.csv` để so sánh giữa các phiên.

```json
{"session_id": "20260822_081013",
 "started_at": "2026-08-22T08:10:13+07:00",
 "ended_at": "2026-08-22T08:10:18+07:00",
 "duration_seconds": 4.9, "frames": 146,
 "counts": {"apple": 4, "banana": 10, "broccoli": 0, "carrot": 0, "orange": 8},
 "total": 22}
```

**Ảnh camera được lật gương khi hiển thị** (bật mặc định, có ô tắt). Webcam trả về ảnh thật, nhưng mọi ứng dụng gọi video đều soi gương, nên ảnh thô đọc như bị ngược. Chỉ lật thẻ `<video>` bằng CSS; canvas overlay **không** lật, vì lật nó sẽ lật luôn chữ trên nhãn — thay vào đó `drawOverlay` đảo toạ độ x của từng hộp. Đã kiểm chứng đây thuần tuý là hiển thị: `drawImage` bỏ qua CSS transform, ảnh chụp ra giống hệt từng byte dù bật hay tắt gương, nên model luôn thấy ảnh thật và số đếm không đổi.

**Chỉ cho một phiên chạy tại một thời điểm.** Tracker nằm trên instance model, nên tab thứ hai sẽ đổ khung của nó vào cùng tracker và cả hai tally đều sai. Từ chối thẳng tốt hơn là cho ra hai con số tự tin nhưng sai.

### Số đo (RTX 4060 Laptop, đo trên máy này)

| | |
|---|---|
| Suy luận yolo26m, batch 1 | 20,9 ms median (47,9 fps), VRAM đỉnh 0,22 / 8,59 GB |
| WebSocket end-to-end | 29,9 ms round trip median, p95 36,3 ms → ~33 fps |
| WebSocket kèm bám vết | 31,4 ms median, p95 39,8 ms — bám vết tốn ~1,5 ms |
| Request đầu sau khởi động | 84 ms, rồi về ~20 ms |

GPU thừa sức, **không cần GPU serverless** (Modal hay tương tự). Ba mức imgsz 320/480/640 cho cùng một thời gian, nghĩa là ~21 ms đó gần như toàn bộ là tiền/hậu xử lý phía CPU của ultralytics.

Đã bỏ phương án ONNX chạy trong trình duyệt: phải viết lại letterbox và giải mã đầu ra YOLO26 bằng JS, mọi sai lệch nhỏ sẽ âm thầm cho kết quả khác báo cáo, lại phải tải ~80 MB về máy khách.

### Tám cạm bẫy, đều do chạy thử mà biết

**1. `rect=False` là bắt buộc.** Ultralytics letterbox ảnh đơn chỉ tới bội số stride gần nhất (640×480), còn lúc đánh giá ảnh đi theo batch 16 lẫn kích thước nên bị đệm thành vuông 640×640. Hai cách cho kết quả khác nhau trên **62/310** ảnh test, lệch tới 3 vật thể — `000000061658.jpg` ra 7 broccoli kiểu chữ nhật, 10 kiểu vuông. Báo cáo đo bằng kiểu vuông. Sau khi ép `rect=False`: **0/310 ảnh lệch**.

**2. Ultralytics coi mảng numpy là BGR.** `np.array(pil)` (RGB) làm `000000002149.jpg` tụt từ 3 xuống 1 hộp, không báo lỗi gì. Phải truyền thẳng đối tượng PIL.

**3. WebSocket phải trả lời mọi khung, kể cả khung hỏng.** Client chỉ gửi khung kế sau khi nhận kết quả khung trước (đó là cơ chế áp lực ngược). Bỏ qua khung hỏng mà không trả lời sẽ làm camera đứng im vĩnh viễn chứ không phải bỏ một khung.

**4. Warmup lúc khởi động.** Không có nó, request đầu tốn 149 ms, và `describe()` báo 21,9 M tham số trước khi ultralytics gộp BatchNorm vào convolution, 20,41 M sau đó — cùng một model, đếm ở hai thời điểm. 20,41 M mới là con số báo cáo dùng.

**5. Khung ảnh phải tính bằng JS, CSS không làm được.** Để `1fr 1fr` thì ảnh dọc nằm trong khung ngang, thừa ~150 px trắng mỗi bên. CSS thuần bó tay: `max-height` chỉ chặn ảnh *hiển thị* cao bao nhiêu, còn bề rộng khung vẫn được bố cục theo kích thước gốc của ảnh — đúng chỗ hở ra. `app.js::fitStage` tính bề rộng cột từ `data.width/height` mà API đã trả về, đặt `gridTemplateColumns` theo px và nghe `resize`. Đo lại: thừa 1 px mỗi bên, đúng bằng đường viền.

**6. Tracker mặc định của YOLO26 im lặng không bám gì cả.** `model.track()` dùng `TRACKTRACK` với `new_track_thresh = 0,7` — nó chỉ mở track cho vật thể tự tin ≥ 0,7, mà phần lớn detection ở đây nằm khoảng 0,35–0,65. Hậu quả: **mọi khung trả về `boxes.id is None`**, không lỗi, không cảnh báo, detection thô vẫn đi qua bình thường — nên nếu không kiểm tra thì phiên nào cũng đếm ra 0. Đo trên một lượt pan 16 khung: TRACKTRACK cho id ở 0/16 khung, ByteTrack 16/16. Phải truyền `tracker="bytetrack.yaml"` (`new_track_thresh = 0,25`), đã gán hằng ở `detector.py::TRACKER_CONFIG`.

Kèm theo: bám vết cần gói `lap`, ultralytics tự tải về lần đầu dùng. Đã ghim `lap==0.5.13` vào `requirements.txt` để lúc demo không phải có mạng.

**7. File tĩnh phải bắt trình duyệt kiểm tra lại.** Sửa `style.css` rồi tải lại trang mà trình duyệt vẫn dùng bản cũ trong cache — không lỗi gì, tính năng chỉ đơn giản là không xuất hiện, và kết luận tự nhiên là "chưa làm". Đã gặp thật với ô Mirror: `index.html` mới nên checkbox hiện ra, nhưng CSS cũ không có quy tắc `.is-mirrored` nên bấm không đổi gì. `RevalidatingStaticFiles` trong `app.py` gắn `Cache-Control: no-cache` — không phải "đừng lưu", mà là "hỏi lại trước khi dùng"; file không đổi trả về 304 rỗng, tốn chưa tới một phần nghìn giây trên localhost.

**8. Thuộc tính `hidden` của HTML thua CSS.** `hidden` chỉ là `display: none` trong stylesheet của trình duyệt, nên bất kỳ quy tắc nào của mình đặt `display` đều thắng nó — và có ba quy tắc như vậy (`.busy`, `.dropzone-empty`, `.dropzone img`). Hậu quả: lớp loading vẫn đè lên kết quả đã hiện, và icon kéo thả chồng lên ảnh vừa tải. Sửa bằng một dòng `[hidden] { display: none !important; }` đặt ngay đầu stylesheet, thay vì vá từng chỗ.

Ngoài ra: `os.chdir(PROJECT_ROOT)` trong `cli.py`, vì `load_pretrained` gọi `YOLO("yolo26m.pt")` bằng đường dẫn tương đối và ultralytics giải nó theo thư mục làm việc — chạy từ chỗ khác thì nó lẳng lặng tải một bản mới về đó.

### Đã kiểm chứng

- Đối chiếu toàn bộ 310 ảnh test: đường webapp và đường `predict_yolo` của evaluation cho **kết quả trùng khớp hoàn toàn**
- `POST /api/detect`: ảnh mẫu 20 vật thể, JSON đủ trường, data URL giải mã ra JPEG hợp lệ; file không phải ảnh trả 400
- `WS /ws/detect`: 40 khung liên tiếp, `seq` đúng thứ tự và liên tục, đổi ngưỡng giữa luồng có tác dụng (15 → 1 vật thể), sống sót qua khung hỏng
- Giao diện: nạp `/api/meta`, bấm ảnh mẫu ra 20 vật thể khớp số đo headless, thanh trượt 0,35 → 0,75 làm 20 → 3, nút tải kết quả hoạt động, console sạch
- Cả hai đường vào của tab Upload — kéo thả và nút Sample — đều cho: ảnh gốc và ảnh kết quả hiện, hai lời nhắc rỗng ẩn, spinner chỉ hiện trong lúc chờ
- Lật gương: hộp ở toạ độ x 100–200 vẽ ra đúng 438–541 trên khung rộng 640 (lệch 2 px là bề dày nét), overlay không dính transform, ảnh gửi model không đổi
- Phiên camera: 60 khung tĩnh giữ nguyên tổng, rời-rồi-quay-lại đếm mới, tab thứ hai bị từ chối, ngắt đột ngột vẫn ghi, phiên rỗng không ghi; giao diện cập nhật tally theo thời gian thực, hiện tên file khi lưu, bảng phiên đã lưu tự làm mới
- Khung ôm sát ảnh ở cả ba tỉ lệ (640×426 ngang, 640×543 gần vuông, 427×640 dọc) và ở cả hai bố cục hai cột lẫn một cột: thừa 1 px mỗi bên, hai khung luôn bằng nhau
- Tab Live: đè `getUserMedia` bằng `canvas.captureStream()` để giả camera — overlay vẽ được hộp, bảng đếm chạy, Start/Stop đúng trạng thái

**Chưa kiểm chứng được:** thông lượng thật của tab Live trong trình duyệt. Pane trình duyệt của công cụ chạy ẩn, mà Chrome bóp `canvas.toBlob` xuống ~1 lần/giây ở tab ẩn (đo được 1012 ms/khung), nên chỉ đạt 1 fps ở đó. Con số đại diện là phép đo WebSocket duy trì phía trên (~33 fps). Cần thử lại bằng camera thật trên tab hiện.

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

## Báo cáo (2026-08-23)

`reports/report.md` là báo cáo nộp, gộp cả hai yêu cầu. Chốt trong lúc viết:

- **Số liệu lấy từ checkpoint `best`**, tức `reports/results/evaluation_test_best.json` — file duy nhất có đủ 4 mô hình, và ba hình eval đã sinh lại từ payload này. Số tốc độ lấy thẳng từ cùng file đó, không đo lại.
- **Bảng tốc độ cũ trong `evaluation.md` (6,6 / 12,4 / 21,4 ms) không truy ngược được về file nào** — JSON đã lưu ghi 13,6 / 25,6 / 42,5 ms. Báo cáo mới dùng số trong JSON.
- **`MODEL_COLORS` thiếu khoá `yolo26m`** nên nó rơi vào màu mặc định trùng `ssdlite`. Đã thêm `#ff7f0e` và sinh lại `eval_per_class_ap.png`.
- **Điểm tin cậy cao nhất của D-FINE đo lại theo checkpoint**: `best` 0,3028, `last` 0,3342. Cả hai dưới ngưỡng 0,35, 0/310 ảnh có hộp. Con số 0,334 trong báo cáo cũ là của `last`.
- **Thêm `src/data/figures.py` + `scripts/05_dataset_figures.py`** sinh 4 hình mô tả dataset. Notebook `01_eda` gọi chính bốn hàm đó rồi hiện file PNG, nên notebook và báo cáo không thể lệch nhau. Đã chạy lại notebook, 13/13 ô có output.
- **Ba ảnh chụp web app** ở `reports/figures/webapp_{upload,live,sessions}.png`. Pane trình duyệt của công cụ không composite frame nên chụp bằng Chrome headless điều khiển qua CDP; script ở scratchpad. Camera giả dựng bằng `canvas.captureStream()`.
- **Số instance có hai cách đếm.** Thô 38.460; sau khi bỏ 600 nhãn `iscrowd` thì còn 37.860 — đó là con số `CocoRecords` nạp và EDA vẽ. Báo cáo ghi cả hai.
- **Hai bản đóng gói, cùng sinh từ `reports/report.md`.** `scripts/06_report_bundle.py` dựng `report-prism/` (Markdown phẳng, 2,2 MB) — `scripts/07_report_latex.py` dựng `report-latex/` + zip cho Overleaf. Máy này KHÔNG có LaTeX lẫn pandoc nên `.tex` chưa được biên dịch thử; đã kiểm tĩnh thay thế: không sót markdown, môi trường cân, số ô khớp số cột, và số mục viết tay trong văn bản khớp số LaTeX tự sinh (9 section / 45 subsection). Phải biên dịch bằng **XeLaTeX**, không phải pdfLaTeX.
- **Hai bẫy đã xử lý khi sinh LaTeX:** một khối code ở mục 9.1 có chữ tiếng Việt (dùng `fancyvrb` thay `listings` vì listings nuốt dấu), và các ký hiệu `→ ≥ − · ×` không có trong font chữ nên đưa qua math mode.
- **`.gitignore` đang loại nhầm cả `src/data/`.** Dòng `data/` không có dấu `/` đầu nên git khớp mọi thư mục tên `data` ở mọi độ sâu. Hậu quả: `src/data/{__init__,build,coco_dataset,transforms}.py` **chưa bao giờ được commit** — clone repo về là không chạy được, và đề bài yêu cầu nộp mã nguồn đầy đủ. Đã đổi thành `/data/*` và `/runs/`. Nhân tiện `!data/.gitkeep` cũng chưa bao giờ có tác dụng vì git không đi vào thư mục đã loại; đổi sang `/data/*` thì nó hoạt động. **Cần `git add src/data/` trước khi đóng gói nộp.**
- **Cột Open Images val ở mục 6.3 không tái lập được**: `oi_data/` đã dọn. Đã đánh dấu rõ trong báo cáo và ở mục giới hạn.

## Việc chưa làm

| Việc | Ghi chú |
|---|---|
| **Thử tab Live bằng camera thật** | Chỉ còn bước này của Yêu cầu 2; server đã đo xong, phần chưa chắc là thông lượng phía trình duyệt |
| **Khôi phục `docs/SETUP.md`** | `git show 9fd187a:docs/SETUP.md > docs/SETUP.md`. Repo không có README ghi cách dựng môi trường |
| **Sửa D-FINE** | Chỉ nếu còn thời gian |
