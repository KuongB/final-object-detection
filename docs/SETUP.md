# Thiết lập môi trường

Môi trường đã được dựng và kiểm thử trên máy phát triển:

| Hạng mục | Giá trị |
|---|---|
| OS | Windows 11 Home Single Language (10.0.26200) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, compute capability 8.9 |
| Driver | 592.00 (CUDA runtime tối đa 13.1) |
| Conda | miniconda3, conda 26.5.3 |
| Env | `objdet`, Python 3.11.15 |

## 1. Tạo môi trường

```bash
conda create -n objdet python=3.11 pip -y
conda activate objdet
```

## 2. Cài PyTorch bản CUDA

Phải cài **trước** và **từ index riêng của PyTorch** — nếu cài `pip install torch`
từ PyPI thông thường sẽ nhận bản CPU-only và không dùng được GPU.

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
```

Chọn build `cu126` vì driver 592.00 hỗ trợ mọi runtime CUDA 12.x.

## 3. Cài các thư viện còn lại

Thứ tự cài có ý nghĩa: `fiftyone` được cài trước `ultralytics` vì nó ràng buộc
version chặt hơn (numpy, pandas, opencv).

```bash
pip install fiftyone
pip install ultralytics pycocotools
pip install fastapi "uvicorn[standard]" python-multipart jinja2 aiofiles seaborn ipykernel jupyterlab
```

Hoặc dùng file khoá version:

```bash
pip install -r requirements.txt
```

## 4. Kiểm tra môi trường

```bash
python scripts/00_check_env.py
```

Script in ra bảng trạng thái của từng thành phần; tất cả phải `OK` trước khi
sang bước thu thập dữ liệu.

## 5. Đăng ký kernel cho Jupyter

```bash
python -m ipykernel install --user --name objdet --display-name "Python (objdet)"
```

---

## Ghi chú về các version đã cài

| Thư viện | Version | Ghi chú |
|---|---|---|
| torch | 2.13.0+cu126 | CUDA khả dụng, đã test matmul trên GPU |
| torchvision | 0.28.0+cu126 | cung cấp `ssdlite320_mobilenet_v3_large` + weights pretrained COCO |
| fiftyone | 1.20.1 | tải subset COCO theo class; đi kèm MongoDB nhúng (`fiftyone-db`) |
| ultralytics | 8.4.117 | YOLOv8m |
| transformers | 5.15.0 | D-FINE-N (`ustc-community/dfine-nano-coco`) |
| timm | 1.0.28 | registry backbone mà transformers nạp D-FINE qua |
| pycocotools | 2.0.11 | có sẵn wheel Windows, không cần build tool C++ |
| numpy | 2.4.4 | không bị fiftyone hạ cấp xuống numpy 1.x |
| opencv-python | 5.0.0.93 | xem cảnh báo bên dưới |

### Cảnh báo: hai bản OpenCV cùng tồn tại

`fiftyone` kéo theo `opencv-python-headless==4.14.0.94`, còn `ultralytics` yêu
cầu `opencv-python`. Cả hai cùng cài vào một package `cv2`, bản cài sau
(`opencv-python 5.0.0.93`) ghi đè lên. Đã kiểm tra `import cv2`, vẽ hình và
`cv2.imencode` đều chạy đúng với version 5.0.0.

**Không gỡ** `opencv-python-headless` — pip sẽ báo thiếu dependency của
fiftyone. Nếu sau này `cv2` lỗi, cài lại theo đúng thứ tự:

```bash
pip install --force-reinstall opencv-python==5.0.0.93
```

### Ổ đĩa

Env `objdet` chiếm khoảng 8–9 GB (phần lớn là các thư viện CUDA đi kèm torch).
Dataset dự kiến ~1.5 GB. Ổ C: cần còn tối thiểu ~12 GB trống trước khi bắt đầu.
