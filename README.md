# Phát hiện trái cây và rau củ, giả lập đếm trong dây chuyền

Đồ án cuối kỳ môn Object Detection. Dự án tự dựng một tập dữ liệu 5 lớp từ COCO 2017,
huấn luyện và so sánh bốn kiến trúc phát hiện vật thể thuộc ba họ khác nhau, rồi đóng gói
mô hình tốt nhất thành một ứng dụng web cho phép tải ảnh lên hoặc chạy camera thời gian thực.

Năm lớp: `apple`, `banana`, `broccoli`, `carrot`, `orange`.

| | |
|---|---|
| Dữ liệu | 7.137 ảnh / 38.460 instance, tập con COCO 2017 |
| Mô hình | SSDLite320-MobileNetV3 (torchvision), YOLO11s và YOLO26m (ultralytics), D-FINE-N (transformers) |
| Đánh giá | pycocotools COCOeval, cùng một hàm chấm cho cả bốn mô hình |
| Ứng dụng | FastAPI + JavaScript thuần, hai chức năng: tải ảnh và camera |

---

## Mục lục

- [Cài đặt](#cài-đặt)
- [Dựng lại tập dữ liệu](#dựng-lại-tập-dữ-liệu)
- [Huấn luyện](#huấn-luyện)
- [Đánh giá](#đánh-giá)
- [Ứng dụng web](#ứng-dụng-web)
- [Notebook](#notebook)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Hợp đồng lưu trữ mô hình](#hợp-đồng-lưu-trữ-mô-hình)
- [Những gì không nằm trong git](#những-gì-không-nằm-trong-git)
- [Môi trường đã kiểm chứng](#môi-trường-đã-kiểm-chứng)


### Tập dữ liệu

| Split | Ảnh | Instance | Nguồn |
|---|---:|---:|---|
| train | 5.803 | 31.389 | COCO train2017 |
| val | 1.024 | 5.479 | COCO train2017, cắt 15 %, seed 42 |
| test | 310 | 1.592 | COCO **val2017** |
| **Tổng** | **7.137** | **38.460** | |

Trong 38.460 instance có 600 nhãn `iscrowd=1` bị loại khi huấn luyện, còn lại 37.860.
Mọi ảnh COCO 2017 có chứa năm lớp này đều đã nằm trong tập dữ liệu; không có cap hay
sampling nào.

---

## Cài đặt

Cần Python 3.11 và một GPU NVIDIA để huấn luyện (chạy CPU được nhưng rất chậm).

```powershell
conda create -n objdet python=3.11
conda activate objdet

# Bước 1 - PyTorch phải cài trước, từ index CUDA:
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126

# Bước 2 - phần còn lại:
pip install -r requirements.txt

# Kiểm tra:
python scripts/00_check_env.py
```


---

## Dựng lại tập dữ liệu

Tập dữ liệu không nằm trong git. Bốn bước dưới đây dựng lại nó từ đầu; bước 1 tải khoảng
1,2 GB ảnh qua FiftyOne và là bước lâu nhất.

```powershell
python scripts/01_download_dataset.py    # tải COCO 2017 -> data/manifest.json
python scripts/02_build_dataset.py       # manifest -> images/labels/annotations/data.yaml
python scripts/03_verify_dataset.py      # đối chiếu nhãn YOLO với nhãn COCO
python scripts/04_check_images.py        # tìm ảnh hỏng (--repair để tải lại)
```

| Script | Cờ | Mặc định |
|---|---|---|
| `01_download_dataset.py` | `--max-samples` | không giới hạn (đặt số để chạy thử nhanh) |
| | `--attempts` | 12 lần thử lại mỗi split khi rớt mạng |
| | `--num-workers` | 4 luồng tải |
| | `--reset-cache` | dựng lại sổ ghi của FiftyOne từ ảnh đã có trên đĩa |
| `02_build_dataset.py` | `--clean` | xoá `images/` và `labels/` rồi dựng lại |
| `04_check_images.py` | `--repair` | tải lại các ảnh hỏng, có kiểm tra `Content-Length` |
| | `--splits` | `train val test` |

Tách làm hai bước — tải rồi mới dựng — vì bước tải tốn hàng chục phút và phụ thuộc mạng,
còn bước dựng chạy trong vài giây và được chạy lại nhiều lần. `manifest.json` là JSON
phẳng, đọc và diff được, nên bộ lọc `classes=` của FiftyOne được kiểm chứng chứ không
được tin.

`02_build_dataset.py` chia split tất định (seed 42), tạo cây ảnh bằng **hard link** vào
cache FiftyOne chứ không copy, rồi ghi song song hai định dạng nhãn: YOLO `.txt` và COCO
JSON. `03_verify_dataset.py` đưa từng hộp YOLO ngược về pixel tuyệt đối và đòi IoU ≥ 0,995
so với hộp COCO tương ứng — đếm file thì không phát hiện được lỗi hoán vị `xyxy`/`xywh`,
còn phép này thì có.

Nếu đường truyền hay đứt: `01b_fetch_images.py` tải lại từng ảnh có retry riêng lẻ (pool
của FiftyOne không có retry ở mức ảnh, một socket chết kéo cả batch theo). Nó ghi thẳng
vào layout cache của FiftyOne, nên sau đó chạy `01_download_dataset.py --reset-cache` là
FiftyOne nhận ra ảnh đã có và chỉ dựng manifest.

```powershell
python scripts/01b_fetch_images.py --split train --workers 2
python scripts/01_download_dataset.py --reset-cache
```

`data/data.yaml` chứa đường dẫn tuyệt đối, do `02_build_dataset.py` sinh ra. Nếu chép
`data/` sang máy khác thì ghi lại nó:

```powershell
python -c "from src.data.build import write_data_yaml; write_data_yaml('/duong/dan/data')"
```

---

## Huấn luyện

```powershell
python scripts/10_train.py --model all             # ba mô hình, tuần tự
python scripts/10_train.py --model yolo11s --epochs 15
python scripts/10_train.py --model ssdlite --smoke # chạy thử 1 epoch ngắn
python scripts/10_train.py --model dfine --resume  # tiếp tục từ last.pt
```

`--model` nhận `ssdlite`, `yolo11s`, `dfine` hoặc `all`, và là cờ bắt buộc duy nhất.
Mọi cờ khác chỉ để ghi đè giá trị mặc định trong registry `MODELS` ở `src/config.py`:
`--epochs`, `--batch`, `--workers`, `--imgsz`, `--device`, `--tag`, `--validate-every`,
`--limit-train-batches`, `--limit-val-batches`, `--no-warm-start`, `--resume`,
`--overwrite`, `--no-promote`, `--smoke`. Cờ nào một mô hình không nhận được thì bị báo và
bỏ qua, nên `--model all` không chết giữa chừng vì một tuỳ chọn lạc.

**`yolo26m` không huấn luyện được bằng lệnh này.** Nó có trong `MODELS` để đánh giá và
ứng dụng web nạp được, nhưng cố ý vắng mặt trong `TRAINERS` vì
`src/training/train_yolo.py` gán cứng `MODEL_KEY = "yolo11s"` — thêm vào sẽ khiến
`--model yolo26m` âm thầm huấn luyện nhầm mô hình rồi promote sai tên. Muốn tái lập nó thì
dùng `notebooks/04_kaggle_yolo26m_openimages.ipynb` trên Kaggle.

Mỗi lượt ghi vào `runs/<key>[_tag]/`: `train.log`, `history.json`, `best.pt`, `last.pt` —
riêng lượt YOLO thì ultralytics giữ layout của nó, `last.pt` nằm trong
`runs/<name>/weights/`, chỉ `best.pt` được nhân bản ra thư mục gốc của run.
Script từ chối chạy đè lên một lượt đã hoàn tất (có `history.json`) trừ khi có `--resume`,
`--overwrite` hoặc `--tag`; phép kiểm tra này chạy cho mọi mô hình đích **trước khi** huấn
luyện bất cứ thứ gì, nên `--model all` báo lỗi ngay chứ không phải sau một giờ.

Stdout được tee vào `runs/<name>/train.log`, nên chạy nền vẫn theo dõi được:

```powershell
Get-Content runs/ssdlite/train.log -Tail 20 -Wait
```

Chi phí huấn luyện đo trên RTX 4060 Laptop: SSDLite 29,7 phút (96,5 s/epoch, VRAM đỉnh
1,81 GB), YOLO11s 89,7 phút (83,5 s/epoch, 4,24 GB), D-FINE-N 42,2 phút (140,2 s/epoch,
3,09 GB).

`workers=2` là con số đo được chứ không phải phỏng đoán thận trọng: Windows spawn worker
thay vì fork, nên mỗi worker phải import lại toàn bộ stack. Với 4 worker, loader của YOLO
chết giữa epoch và SSDLite chạy 0,201 s/step; với 2 worker nó ổn định và nhanh hơn,
0,162 s/step. Trên Linux nâng lên 8 được.

---

## Đánh giá

```powershell
python scripts/20_evaluate.py --model all --split test --checkpoint best
python scripts/20_evaluate.py --model all --split test --checkpoint pretrained
```

`--checkpoint` nhận ba giá trị: `last` (cuối lượt huấn luyện, **mặc định**), `best`
(epoch có val mAP cao nhất, lấy từ `weights/index.json`), `pretrained` (checkpoint COCO
công khai trước khi fine-tune — chính là cột "gốc" trong bảng kết quả). Bảng số liệu
trong báo cáo dựng từ `best` và `pretrained`, nên chạy lệnh trần không tái lập cột nào.

`--model` lặp lại được và mặc định là mọi mô hình có trong `weights/index.json`; nó cố ý
không có `choices=` vì tập mô hình *đánh giá được* không trùng tập *huấn luyện được*.
Các cờ còn lại: `--split` (`train`/`val`/`test`, mặc định `test`), `--device`, `--batch`
(16), `--workers` (2), `--benchmark-iterations` (50), `--no-save-detections`,
`--no-figures`.

Cả bốn mô hình đi qua **một** hàm chấm duy nhất, `pycocotools.COCOeval` trong
`src/evaluation/coco_eval.py`, đối chiếu với `data/annotations/instances_<split>.json`.
Validator riêng của từng framework bị bỏ qua có chủ đích: như vậy con số so sánh mô hình
với nhau chứ không so sánh cách hiện thực phép đo.

Kết quả ghi vào `reports/results/evaluation_<split>_<checkpoint>.json` — file này được
**gộp** chứ không ghi đè, nên đánh giá một mô hình không xoá dòng của các mô hình khác.
Trừ khi có `--no-figures`, ba hình cho báo cáo được vẽ vào `reports/figures/`.
Thư mục `reports/` được tạo lúc chạy.

Hai lưu ý khi đọc số:

- **Độ trễ không phải hằng số.** GPU laptop hạ xung khi tải nhẹ (2370 MHz so với 3105 MHz
  tối đa), nên cùng một mô hình có thể lệch tới hai lần giữa hai lần đo cách nhau vài
  phút. Mọi số tốc độ trong báo cáo đều lấy từ **một** lần chạy duy nhất, và chỉ các mô
  hình đo trong cùng một lần chạy mới so sánh được với nhau.
- **`weights/index.json` lưu val mAP, không phải test mAP.** Riêng `yolo26m` thì con số
  val đó (0,3335) còn là trên Open Images, được đánh dấu bằng khoá `val_dataset`.

---

## Ứng dụng web

```powershell
python scripts/30_serve.py                      # http://localhost:8000
python scripts/30_serve.py --port 8080 --device cpu
```

Cờ: `--host` (`127.0.0.1`), `--port` (`8000`), `--device` (`auto`), `--model`
(`yolo26m`, chỉ nhận checkpoint ultralytics), `--reload` (chỉ để phát triển, nó nạp lại
cả mô hình).

**Phải mở bằng `localhost`.** Trình duyệt chỉ cấp quyền camera trong secure context, nên
vào bằng IP LAN thì tab Live bị chặn; trang tự hiện cảnh báo khi gặp trường hợp đó. Tab
Upload vẫn chạy bình thường.

Hai chức năng:

- **Upload** — kéo thả hoặc chọn một ảnh, server vẽ hộp rồi trả ảnh về dưới dạng data URL
  kèm danh sách vật thể và số lượng theo lớp. Có sẵn bốn ảnh mẫu để demo được khi chưa
  dựng `data/`.
- **Live** — camera chạy qua WebSocket. Server chỉ trả **toạ độ**, trình duyệt tự vẽ
  overlay lên canvas, nên không phải mã hoá lại ảnh mỗi khung.

| Route | Việc |
|---|---|
| `GET /` | trang HTML |
| `GET /api/health` | trạng thái và thiết bị |
| `GET /api/meta` | 5 lớp, bảng màu, ngưỡng mặc định, thông tin mô hình, danh sách ảnh mẫu |
| `POST /api/detect` | ảnh + ngưỡng → JSON kèm ảnh đã vẽ hộp, danh sách vật thể, số lượng theo lớp |
| `WS /ws/detect` | khung JPEG vào, toạ độ ra |
| `GET /api/sessions` | các phiên camera đã lưu, mới nhất trước |
| `GET /sessions/*` | file JSON và `sessions.csv` |

Mô hình được nạp một lần trong `lifespan`, dùng lại `load_pretrained()` của
`src/evaluation/runner.py`, nên bảng ánh xạ COCO-80 → 5 lớp chỉ tồn tại ở một chỗ và
ứng dụng không thể lệch khỏi con số trong báo cáo. Bảng màu lấy từ `CLASS_COLORS_RGB` và
phát qua `/api/meta`, nên overlay trong trình duyệt, ảnh server vẽ và hình trong báo cáo
luôn dùng chung một bộ màu. Hai bản sao mô hình được nạp lúc khởi động — một để predict,
một để track — vì ultralytics gắn tracker vào chính instance mô hình.

**Đếm vật thể theo phiên camera.** Một phiên bắt đầu khi bấm Start và kết thúc khi bấm
Stop; socket chính là phiên. Số đếm là phép hợp tập hợp **id riêng biệt của tracker**,
không phải tổng số đếm từng khung: ở 30 fps, cộng dồn từng khung sẽ biến một quả táo đứng
yên 10 giây thành 300 quả. Vật thể rời khung rồi quay lại nhận id mới, tức được tính là
quả mới. Mỗi phiên ghi một `webapp/sessions/session_<id>.json` và nối một dòng vào
`sessions.csv`. Phiên không có khung nào thì không ghi file; ngắt kết nối đột ngột vẫn
được ghi vì đường ngắt đi chung hàm `close()`.

Chỉ cho phép một phiên camera tại một thời điểm, toàn server. Tracker nằm trên instance mô
hình, nên tab thứ hai sẽ đổ khung của nó vào cùng tracker và làm sai cả hai bảng đếm; kết
nối thứ hai bị từ chối thẳng.

Số đo trên RTX 4060 Laptop: suy luận 20,9 ms trung vị (47,9 fps), WebSocket khứ hồi
29,9 ms trung vị và p95 36,3 ms (~33 fps), có bám vết thì 31,4 ms. Ba mức `imgsz`
320/480/640 cho cùng một thời gian, nghĩa là phần lớn trong ~21 ms đó là tiền và hậu xử lý
phía CPU của ultralytics chứ không phải phép nhân ma trận.

---

## Notebook

| Notebook | Nội dung |
|---|---|
| `01_eda.ipynb` | Khảo sát dữ liệu: bảng split và lớp, phân bố kích thước vật thể, ma trận đồng xuất hiện, heatmap tâm hộp, rồi chạy ba checkpoint gốc trên cùng ba ảnh test. Đã chạy sẵn, còn output. |
| `02_eval.ipynb` | Đánh giá: đọc `history.json` của bốn lượt huấn luyện, chấm điểm trên tập test, so sánh YOLO11s 15 epoch với 64 epoch và phân tích khoảng cách val/test. Đã chạy sẵn. |
| `04_kaggle_yolo26m_openimages.ipynb` | Bản chạy được trên Kaggle: tự tải Open Images V7 từ CSV của Google và bucket S3 công khai, không import gì từ `src/`. Chưa chạy, không có output. |
| `04_kaggle_yolo26m_openimages_run.ipynb` | Bản đã chạy thật, giữ nguyên log của phiên Kaggle. Nội dung ô giống hệt bản trên, thêm một ô đóng gói kết quả cuối phiên. |

`01_eda.ipynb` gọi đúng các hàm trong `src/data/figures.py` mà báo cáo dùng, nên hình
trong notebook và hình trong báo cáo không thể lệch nhau.

Ba điều cần nhớ khi chạy hai notebook Kaggle, đều do chạy thử mà biết: accelerator phải là
`GPU T4 x2` chứ không phải P100 (PyTorch không còn hỗ trợ sm_60, notebook có ô kiểm tra
dừng ngay); không cài `fiftyone`; và đừng tắt bật Internet giữa chừng vì Kaggle sẽ khởi
động lại kernel và giết tiến trình huấn luyện.

---

## Cấu trúc thư mục

```
src/                    toàn bộ logic
  config.py             đường dẫn, 5 lớp, registry MODELS, bảng màu, hằng số đánh giá
  data/                 dựng dataset, lớp Dataset COCO, augment, hình khảo sát
  models/               nạp mô hình, chuyển head 80 lớp -> 5 lớp
  training/             CLI, vòng lặp dùng chung
  webapp/               FastAPI, detector, phiên camera, vẽ hộp
scripts/                
  00_check_env.py       kiểm tra môi trường
  01_download_dataset.py + 01b_fetch_images.py
  02_build_dataset.py   manifest -> images/labels/annotations/data.yaml
  03_verify_dataset.py  đối chiếu nhãn YOLO và COCO
  04_check_images.py    tìm và sửa ảnh hỏng
  10_train.py           -> src/training/cli.py
  20_evaluate.py        -> src/evaluation/cli.py
  30_serve.py           -> src/webapp/cli.py
webapp/
  frontend/             
  samples/              4 ảnh mẫu cho demo
  sessions/             bản ghi phiên camera, sinh lúc chạy
notebooks/              4 notebook
data/                   sinh ra bởi scripts/01-04
runs/                   một thư mục cho mỗi lượt huấn luyện
weights/                checkpoint đã promote + index.json + dfine/hf/ (cấu hình transformers)

```


