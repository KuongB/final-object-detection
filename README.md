# Phát hiện & đếm rau củ quả — Đồ án cuối kỳ Object Detection

Phát hiện và đếm 5 loại rau củ quả (`apple`, `banana`, `broccoli`, `carrot`,
`orange`) từ subset COCO 2017, mô phỏng bài toán kiểm kê nông sản qua camera
trên băng chuyền / quầy tính tiền tự động.

Đồ án gồm hai phần:

1. **(8đ)** Xây dựng dataset, huấn luyện và so sánh **3 kiến trúc khác dòng**:

   | Dòng kiến trúc | Model | Thư viện |
   |---|---|---|
   | CNN one-stage | SSD300-VGG16 | `torchvision` |
   | YOLO | YOLOv8 | `ultralytics` |
   | Transformer | RT-DETR | `ultralytics` |

2. **(2đ)** Web application: upload ảnh → trả về ảnh đã vẽ bounding box **kèm
   bảng đếm số lượng theo từng loại**. Mở rộng (Phase 2): xử lý video với
   tracking + counting line để không đếm trùng.

---

## Cấu trúc thư mục

```
final-object-detection/
├── configs/                    # cấu hình huấn luyện cho từng model
├── data/                       # (không commit vào git)
│   ├── fiftyone_export/        #   dump thô từ FiftyOne
│   ├── coco/                   #   COCO JSON  -> SSD (torchvision)
│   │   ├── images/{train,val,test}/
│   │   └── annotations/instances_{train,val,test}.json
│   └── yolo/                   #   YOLO txt   -> YOLOv8 / RT-DETR
│       ├── images/{train,val,test}/
│       ├── labels/{train,val,test}/
│       └── data.yaml
├── docs/
│   └── SETUP.md                # hướng dẫn dựng môi trường chi tiết
├── notebooks/                  # EDA, training, evaluation (chạy được trên Colab/Kaggle)
├── reports/
│   ├── figures/                # biểu đồ cho báo cáo
│   └── results/                # bảng metrics dạng CSV/JSON
├── runs/                       # output thô của các lần train
├── scripts/                    # pipeline chạy bằng CLI
├── src/                        # code dùng chung
│   ├── config.py               # ⭐ single source of truth: path + class mapping
│   ├── data/  models/  training/  evaluation/  utils/
├── weights/                    # checkpoint tốt nhất của từng model
└── webapp/
    ├── backend/                # FastAPI
    └── frontend/               # giao diện demo
```

---

## Bắt đầu

```bash
conda activate objdet
python scripts/00_check_env.py     # tất cả phải OK
```

Chi tiết cài đặt: xem [docs/SETUP.md](docs/SETUP.md).

---

## Tiến độ

- [x] **Bước 1** — Setup môi trường + cấu trúc dự án
- [x] **Bước 2** — Thu thập dữ liệu (FiftyOne → COCO JSON + YOLO txt)
- [x] **Bước 3** — EDA ([notebooks/01_eda.ipynb](notebooks/01_eda.ipynb))
- [ ] **Bước 4** — Huấn luyện 3 model
- [ ] **Bước 5** — Đánh giá & so sánh
- [ ] **Bước 6** — Web app Phase 1 (ảnh)
- [ ] **Bước 7** — Web app Phase 2 (video + tracking)

---

## Dataset

Subset COCO 2017 chứa 5 lớp rau củ quả — **7.137 ảnh / 38.460 instances**.

| Split | Nguồn | Ảnh | Instances | obj/ảnh | Vai trò |
|---|---|---:|---:|---:|---|
| `train` | `coco-2017/train` (85%) | 5.803 | 31.389 | 5,41 | huấn luyện |
| `val` | `coco-2017/train` (15%) | 1.024 | 5.479 | 5,35 | chọn checkpoint, early stopping |
| `test` | `coco-2017/validation` | 310 | 1.592 | 5,14 | **đánh giá cuối cùng** |

Test set lấy từ `val2017` — hoàn toàn tách biệt khỏi `train2017`, đã verify 0
ảnh trùng giữa các split — nên kết quả so sánh giữa 3 model là khách quan.

### Phân bố lớp

| Lớp | Ảnh (train2017) | Instances | train % | val % | test % |
|---|---:|---:|---:|---:|---:|
| apple | 1.586 | 5.851 | 16,0 | 15,1 | 15,0 |
| banana | 2.243 | 9.458 | 25,5 | 26,8 | 23,8 |
| broccoli | 1.939 | 7.308 | 19,9 | 19,2 | 19,8 |
| carrot | 1.683 | 7.852 | 21,4 | 20,9 | 23,3 |
| orange | 1.699 | 6.399 | 17,2 | 18,0 | 18,0 |

Split được stratify theo **tổ hợp lớp** có trong mỗi ảnh (multi-label
stratification) → chênh lệch tỷ lệ lớp giữa train và val tối đa **1,34 điểm
phần trăm**.

### Pipeline

```bash
python scripts/01_download_dataset.py     # FiftyOne -> data/manifest.json
python scripts/01b_fetch_images.py        # (chỉ khi mạng chập chờn)
python scripts/02_build_dataset.py        # manifest -> COCO JSON + YOLO txt
python scripts/03_verify_dataset.py       # verify toạ độ + ultralytics
```

`03_verify_dataset.py` round-trip **toàn bộ 37.860 box** từ YOLO ngược về pixel
tuyệt đối và so với box COCO gốc — IoU thấp nhất **0,9996**. Đây là bước bắt
buộc: các check đếm số lượng không phát hiện được lỗi lẫn `xywh`/`xyxy`.

## EDA — kết quả chính

Chạy: `jupyter lab notebooks/01_eda.ipynb` (kernel `Python (objdet)`), khoảng 15
giây. Notebook chỉ đọc 3 file annotation JSON, không đọc ảnh.

| Chỉ số | Giá trị | Ý nghĩa |
|---|---:|---|
| Mất cân bằng lớp | **1,62×** | Nhẹ. mAP trung bình đáng tin, không cần focal loss có trọng số |
| Drift lớp train↔val | 1,34 điểm % | Split được stratify tốt |
| Drift lớp train↔test | 1,94 điểm % | test giữ nguyên phân bố gốc COCO |
| Vật thể **small** (COCO) | **25,6%** | Sẽ là điểm phân hoá 3 model rõ nhất |
| Vật thể medium / large | 45,1% / 29,4% | |
| Kích thước tương đối trung vị | 0,114 | Vật thể chiếm ~11% cạnh ảnh |
| Box trong khoảng 1:3–3:1 | 95,1% | Anchor mặc định của SSD phủ được đa số |
| Ảnh đa lớp | 27,0% | |
| Ảnh có 2 vật thể IoU > 0,5 | 5,1% | Giới hạn recall do NMS |

**Dự đoán trước khi train** (ghi lại để đối chiếu ở bước 5): SSD300 nhanh nhất
nhưng `AP_small` thấp nhất — nó resize ảnh về 300×300, nên vật thể ở mức trung
vị (11,4% cạnh ảnh) chỉ còn ~34 px, còn nhóm small chỉ còn khoảng 10 px.
YOLOv8 cân bằng nhất. RT-DETR mạnh ở vật thể trung bình/lớn và không dùng NMS
nên ít mất recall ở 5,1% ảnh chồng lấn, nhưng cần nhiều epoch hơn để hội tụ.

10 hình trong [reports/figures/](reports/figures/) (`eda_*.png`), số liệu tổng
hợp trong [reports/results/eda_summary.json](reports/results/eda_summary.json).

### Xử lý `iscrowd`

600 box `iscrowd=1` (đám đông vật thể gộp thành 1 box) được **giữ trong COCO
JSON** — `COCOeval` hiểu và tự bỏ qua đúng cách — nhưng **loại khỏi label
YOLO** vì ultralytics không có khái niệm này và sẽ học nguyên đống thành một
vật thể khổng lồ.
