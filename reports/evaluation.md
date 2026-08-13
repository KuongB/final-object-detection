# Ghi nhận kết quả huấn luyện và đánh giá ba mô hình

## 1. Bối cảnh

Ba kiến trúc thuộc ba hướng tiếp cận khác nhau được huấn luyện trên cùng một tập dữ liệu, cùng một phần cứng, cùng số epoch, rồi chấm điểm bằng cùng một bộ đo.

| Hạng mục | Nội dung |
|---|---|
| Dữ liệu | Tập con COCO 2017, 5 lớp `apple / banana / broccoli / carrot / orange` |
| Phân chia | train 5.803 ảnh · val 1.024 ảnh · **test 310 ảnh** |
| Nguồn tập test | Trích từ COCO `val2017`, không tham gia huấn luyện lẫn chọn checkpoint |
| Phần cứng | NVIDIA RTX 4060 Laptop 8 GB, PyTorch 2.13.0+cu126, Python 3.11.15 |
| Bộ đo | `pycocotools` COCOeval, áp dụng như nhau cho cả ba mô hình |
| Trọng số dùng để chấm | Checkpoint **cuối** (`last.pt`, epoch 15) của mỗi mô hình |

Điểm cần nêu về phương pháp: cả ba đều được khởi tạo đầu bằng cách sao chép trọng số của 5 lớp tương ứng từ checkpoint COCO gốc thay vì khởi tạo ngẫu nhiên, vì cả 5 lớp này đều nằm trong 80 lớp COCO. Do đó điểm số tại epoch 0 không bằng 0.

Về việc chấm trên `last.pt`: trong quá trình huấn luyện, mAP validation được tính trên bản EMA của trọng số. Khi nạp lại `last.pt`, bản EMA cũng là bản được lấy ra, nên con số ở đây nối liền được với đường cong huấn luyện ở mục 4.

---

## 2. Cấu hình và chi phí huấn luyện

| | SSDLite320-MobileNetV3 | YOLO11s | D-FINE-N |
|---|---|---|---|
| Hướng tiếp cận | CNN một giai đoạn, dựa trên anchor | YOLO một giai đoạn, không anchor | Transformer (họ DETR), đầu-cuối |
| Thư viện | torchvision | ultralytics 8.4.117 | transformers 5.15.0 |
| Tham số | 2.261.960 | 9.414.735 | 3.722.121 |
| Kích thước ảnh vào | 320×320 | 640×640 | 640×640 |
| Dung lượng checkpoint | 8,9 MB | 18,3 MB | 14,5 MB |

| Tham số huấn luyện | SSDLite | YOLO11s | D-FINE-N |
|---|---|---|---|
| Epoch | 15 | 15 | 15 |
| Batch size | 32 | 16 | 16 |
| Bước / epoch | 181 | — | 362 |
| Optimizer | SGD, momentum 0.9, nesterov | AdamW (`optimizer=auto` tự chọn) | AdamW |
| Learning rate | 0.01 | 0.01 → 0.0011 (tự điều chỉnh) | 2.5e-4 (backbone 2.5e-5) |
| Weight decay | 4e-5 | 5e-4 | 1e-4 |
| Warmup | 3 epoch | 3 epoch | 2 epoch |
| Lịch LR | cosine | cosine | cosine |
| Cắt gradient | 10.0 | — | 0.1 |
| EMA | 0.999 | nội bộ ultralytics | 0.999 |
| AMP | có | có | có |
| Tăng cường dữ liệu | photometric, zoom-out, IoU-crop, lật | mosaic, HSV, scale, lật | lật, photometric nhẹ |

| Chi phí thực tế | SSDLite | YOLO11s | D-FINE-N |
|---|---|---|---|
| Tổng thời gian | 29,7 phút | 24,5 phút | 42,2 phút |
| Thời gian trung vị mỗi epoch | 96,5 s | 83,6 s | 140,2 s |
| VRAM đỉnh | 1,81 GB | 4,24 GB | 3,09 GB |

Tổng thời gian huấn luyện cả ba: 1 giờ 37 phút.

Số `workers=2` được dùng cho cả ba. Trên Windows, tiến trình nạp dữ liệu được sinh bằng `spawn` chứ không `fork`, nên mỗi worker phải nạp lại toàn bộ thư viện; con số 2 là mức đo được ổn định trên máy này.

---

## 3. Kết quả trên tập test

| Mô hình | mAP@[.5:.95] | mAP@.5 | mAP@.75 | AR@100 | ms/ảnh | FPS |
|---|---|---|---|---|---|---|
| SSDLite320-MobileNetV3 | 0,1237 | 0,2303 | 0,1178 | 0,2721 | 12,39 | 80,7 |
| YOLO11s | 0,2569 | 0,3999 | 0,2708 | 0,5545 | 6,63 | 150,8 |
| D-FINE-N | 0,0447 | 0,0903 | 0,0391 | 0,4527 | 21,35 | 46,8 |

Độ trễ đo ở batch = 1 trên GPU, bỏ qua các vòng khởi động và có đồng bộ CUDA, tức là chi phí của một lượt suy luận đơn lẻ — đúng với tình huống người dùng tải lên một tấm ảnh.

### AP theo kích thước vật thể

| Mô hình | nhỏ | trung bình | lớn |
|---|---|---|---|
| SSDLite | 0,0014 | 0,0801 | 0,3210 |
| YOLO11s | 0,0778 | 0,2867 | 0,3957 |
| D-FINE-N | 0,0135 | 0,0850 | 0,0715 |

### AP theo từng lớp

![AP theo từng lớp](figures/eval_per_class_ap.png)

---

## 4. Nhận xét

### 4.1 Diễn biến trong 15 epoch

![Đường cong validation mAP](figures/eval_learning_curves.png)

Cả ba đường cong đều đi xuống trong ba epoch đầu. Đây là hệ quả trực tiếp của giai đoạn warmup: learning rate tăng tuyến tính từ gần 0 lên mức đỉnh trong 3 epoch đầu, và trong lúc đó phần đầu phân loại vừa được sao chép từ COCO bị đẩy ra khỏi vị trí ban đầu nhanh hơn tốc độ nó học lại. Điểm thấp nhất của cả SSDLite lẫn YOLO11s rơi đúng vào epoch 3 — thời điểm learning rate chạm đỉnh.

Từ epoch 4 trở đi, khi cosine bắt đầu hạ learning rate, SSDLite và YOLO11s cùng đi lên và giữ xu hướng tăng tới hết. Đường của SSDLite phẳng dần ở khoảng epoch 13–15 (0,1514 → 0,1509), cho thấy nó đã tiến gần điểm bão hoà với cấu hình hiện tại. Đường của YOLO11s vẫn còn độ dốc dương tại epoch 15 (0,2868 ở epoch cuối), tức là nó chưa hội tụ và số epoch chứ không phải kiến trúc đang là ràng buộc.

D-FINE-N đi theo hướng ngược lại: đạt 0,1277 ngay tại epoch 1 rồi giảm đều đến 0,0593 ở epoch 15. Điều đáng chú ý là learning rate của nó đã hoàn tất warmup từ epoch 2 và giảm xuống 2,5e-06 ở epoch cuối, nhưng đường cong không hồi lại. Trong khi đó hàm mất mát vẫn giảm đơn điệu suốt quá trình (17,29 → 14,74). Hai dấu hiệu này cùng tồn tại có nghĩa: quá trình tối ưu vẫn vận hành đúng về mặt cơ học, nhưng đại lượng đang được tối thiểu hoá không đi cùng chiều với chất lượng phát hiện đo bằng COCO mAP.

### 4.2 Khoảng cách giữa khả năng tìm và khả năng chấm điểm

Đặt cạnh nhau tỉ lệ AR@100 (khả năng tìm ra vật thể) và mAP (có tính đến độ tin cậy và độ khớp của hộp):

| Mô hình | AR@100 | mAP@[.5:.95] | Tỉ lệ AR/mAP |
|---|---|---|---|
| SSDLite | 0,2721 | 0,1237 | 2,2× |
| YOLO11s | 0,5545 | 0,2569 | 2,2× |
| D-FINE-N | 0,4527 | 0,0447 | **10,1×** |

SSDLite và YOLO11s có cùng tỉ lệ 2,2×, là mức thông thường. D-FINE-N thì khác hẳn: nó tìm ra vật thể ở mức khá — AR 0,4527, cao hơn SSDLite — nhưng mAP lại thấp gấp mười lần AR.

Chênh lệch này chỉ ra vị trí của vấn đề. Nếu mô hình không định vị được vật thể thì AR phải thấp; AR 0,45 cho thấy các truy vấn của nó vẫn phủ đúng vùng có vật thể. Thứ sụp đổ là điểm tin cậy và độ chính xác của hộp: mAP@.75 (0,0391) chỉ bằng 43% mAP@.5 (0,0903), nghĩa là hộp có phủ đúng vật thể nhưng lệch nhiều khi yêu cầu độ khớp chặt hơn.

Một con số cụ thể hoá điều này: **điểm tin cậy cao nhất mà D-FINE-N đưa ra trên toàn bộ 310 ảnh test là 0,334**. Toàn bộ dự đoán của nó nằm dưới ngưỡng 0,35 mà ứng dụng web dự kiến dùng để hiển thị. Để so sánh, SSDLite có 499 hộp và YOLO11s có 915 hộp vượt ngưỡng này.

### 4.3 Ảnh hưởng của kích thước ảnh đầu vào

AP trên vật thể nhỏ của SSDLite là 0,0014 — gần như bằng không, trong khi AP trên vật thể lớn của nó đạt 0,3210, tức cùng bậc với YOLO11s (0,3957).

Nguyên nhân nằm ở kích thước đầu vào cố định 320×320 của kiến trúc này. Một quả táo chiếm 32×32 pixel trong ảnh gốc 640×480 sẽ còn khoảng 16×16 pixel sau khi thu nhỏ, và sau sáu tầng giảm mẫu của MobileNetV3 thì nó không còn đủ tín hiệu ở bất kỳ tầng đặc trưng nào để anchor bắt được. YOLO11s và D-FINE-N nhận ảnh 640×640, gấp bốn lần diện tích, nên giữ lại được các vật thể ở dải kích thước này.

Đây cũng là lời giải thích cho phần lớn khoảng cách mAP tổng thể giữa SSDLite và YOLO11s: trên vật thể lớn hai bên khá gần nhau, chênh lệch tập trung ở dải nhỏ và trung bình.

### 4.4 Phân bố theo lớp

Với SSDLite, `orange` (0,1989) cao hơn `apple` (0,0766) và `carrot` (0,0756) khoảng 2,6 lần. Cam trong tập dữ liệu này thường xuất hiện dưới dạng vật thể tròn, đơn lẻ, tương phản cao với nền; táo thường nằm thành đống trong sọt với ranh giới giữa các quả rất mờ, và cà rốt thường bị che khuất một phần hoặc nằm bó chồng lên nhau — cả hai đều là tình huống mà anchor cố định khó tách từng thể hiện riêng.

YOLO11s trải đều hơn: khoảng cách giữa lớp cao nhất (`orange` 0,3201) và thấp nhất (`apple` 0,2205) chỉ là 1,45 lần. Cơ chế gán nhãn theo độ khớp nhiệm vụ và tăng cường mosaic khiến mô hình gặp nhiều cấu hình chồng lấn khác nhau hơn trong lúc huấn luyện.

D-FINE-N thấp đều ở cả năm lớp (0,0250–0,0642), không có lớp nào nổi bật. Việc suy giảm xảy ra đồng đều chứ không tập trung vào một lớp cụ thể cho thấy nó không phải hiện tượng gắn với đặc điểm hình ảnh của một loại rau củ nào.

### 4.5 Quan sát định tính

![Phát hiện trên tập test](figures/eval_qualitative.png)

Ba ảnh được chọn theo phân vị mật độ hộp (thưa – trung bình – dày) thay vì lấy các ảnh dày nhất, vì những ảnh dày nhất trong tập test là quầy chợ 25 hộp và ảnh ghép lưới mà không mô hình nào xử lý được, chúng nói về tập dữ liệu nhiều hơn nói về mô hình. Ngưỡng hiển thị là 0,35.

Ở hàng giữa (5 hộp thật), SSDLite và YOLO11s cùng đưa ra 5 hộp bao phủ các cụm bông cải. Ở hàng dưới (13 hộp thật, bông cải rải trên mặt bánh pizza), cả hai đều gom nhiều bông cải nhỏ liền kề thành một hộp lớn thay vì tách riêng — SSDLite còn 3 hộp, YOLO11s còn 5. Đây là biểu hiện thị giác của cùng hiện tượng mà cột AP vật thể nhỏ ở mục 4.3 đã đo được bằng số.

Các ô của D-FINE-N trống ở cả ba hàng. Đây không phải do mô hình không sinh ra dự đoán — nó vẫn xuất đủ 100 hộp mỗi ảnh — mà do toàn bộ điểm tin cậy đều dưới 0,35 như đã nêu ở mục 4.2.

### 4.6 Tốc độ suy luận

Thứ tự độ trễ (YOLO11s 6,63 ms < SSDLite 12,39 ms < D-FINE-N 21,35 ms) không đi cùng thứ tự số tham số (SSDLite 2,26 M < D-FINE-N 3,72 M < YOLO11s 9,41 M). YOLO11s có số tham số gấp bốn lần SSDLite nhưng chạy nhanh gần gấp đôi.

Lý do là số tham số không quyết định độ trễ trên GPU; số lần phóng nhân (kernel launch) và mức độ song song mới quyết định. YOLO11s được hợp nhất lớp (`fused`) và gồm các khối tích chập lớn, chạy hết công suất trên GPU trong ít lần gọi. SSDLite dùng tích chập tách theo chiều sâu — tiết kiệm tham số nhưng chia phép tính thành nhiều nhân nhỏ, mỗi nhân không lấp đầy GPU; trên Windows với trình điều khiển WDDM thì chi phí mỗi lần phóng nhân lại càng đáng kể. D-FINE-N thêm phần tự chú ý trên 300 truy vấn và sáu tầng decoder tuần tự, mỗi tầng phải chờ tầng trước.

Cả ba đều nằm trong ngưỡng dùng được cho ứng dụng web nhận một ảnh mỗi lượt: chậm nhất là 21 ms, tương đương 47 ảnh mỗi giây.

---

## 5. Ghi chú về giới hạn của đợt đo này

- **15 epoch là ngân sách cố định cho cả ba, không phải điểm hội tụ của mô hình nào.** Đường cong của YOLO11s vẫn còn dốc lên tại epoch cuối, nên con số 0,2569 là điểm dừng của một quá trình chưa kết thúc chứ không phải trần của kiến trúc.
- **Tập test có 310 ảnh.** Đây là cỡ mẫu nhỏ; chênh lệch nhỏ giữa hai lần đo trên tập này không mang nhiều ý nghĩa thống kê.
- **Số của D-FINE-N phản ánh trạng thái cuối của một quá trình huấn luyện đi xuống**, không phản ánh năng lực của kiến trúc D-FINE nói chung. Checkpoint tốt nhất của nó nằm ở epoch 1 với mAP test 0,0928 — tức là ở trạng thái gần như chưa được huấn luyện trên tập này.
- Các tham số huấn luyện được lấy nguyên từ registry trong `src/config.py` và chưa qua tinh chỉnh riêng cho tập dữ liệu 5 lớp này.

## 6. Cách tái lập

```bash
python scripts/10_train.py --model all --epochs 15
```

```bash
python scripts/20_evaluate.py --model all --split test --checkpoint last
```

Kết quả thô: `reports/results/evaluation_test_last.json`, các file dự đoán `reports/results/detections_<model>_test_last.json`, hình trong `reports/figures/`.
