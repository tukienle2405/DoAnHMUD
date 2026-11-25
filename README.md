# 🌱 ĐỒ ÁN HỌC MÁY VÀ ỨNG DỤNG
## ĐỀ TÀI: NHẬN DIỆN BỆNH LÁ CÂY TRỒNG VÀ HỖ TRỢ PHÁC ĐỒ ĐIỀU TRỊ THÔNG MINH ỨNG DỤNG HỌC SÂU (DEEP LEARNING) VÀ EXPLAINABLE AI (Grad-CAM)

---

## 📌 1. Cấu Trúc Mã Nguồn (Tên file rõ ràng & Tách bạch module)

Dự án được thiết kế theo kiến trúc module hóa chuẩn công nghiệp:

| Tên File | Vai trò & Chức năng chi tiết |
| :--- | :--- |
| **`disease_knowledge.py`** | **Cơ sở tri thức bệnh học toàn diện**: Chứa thông tin về **26 bệnh lá cây trồng** (15 bệnh đã có trong tập dữ liệu AI và 11 bệnh nguy hiểm thực địa mở rộng như Thán thư, Phấn trắng, Héo xanh vi khuẩn, Héo vàng Fusarium...). Cung cấp phác đồ điều trị 3 bước: Canh tác - Sinh học - Hóa học, phục hồi cây và bảng phân biệt nhầm lẫn. |
| **`model_architecture.py`** | **Kiến trúc mô hình**: Định nghĩa mạng Baseline CNN (đối chuẩn) và mô hình Transfer Learning (MobileNetV2 cải tiến). |
| **`train_models.py`** | **Huấn luyện & Đánh giá**: Phân chia tập dữ liệu (Train/Val/Test), tăng cường dữ liệu (Augmentation), huấn luyện mô hình, lưu trọng số `best_plant_model.pth`, vẽ ma trận nhầm lẫn (Confusion Matrix) và đồ thị Loss/Accuracy. |
| **`explainability.py`** | **Trí tuệ nhân tạo có thể giải thích (Explainable AI - XAI)**: Triển khai thuật toán Grad-CAM sinh bản đồ nhiệt (Heatmap) chỉ ra vị trí đốm lá bị bệnh trên ảnh. |
| **`app_streamlit.py`** | **Ứng dụng Web tương tác (Streamlit)**: Giao diện trực quan: tải ảnh ngoài / chụp camera, cơ chế phát hiện ảnh lạ (OOD), hiển thị Grad-CAM, đơn thuốc và tra cứu cẩm nang 26 bệnh. |
| **`best_plant_model.pth`** | File lưu trọng số mô hình PyTorch đã được huấn luyện tốt nhất. |
| **`evaluation_results/`** | Thư mục lưu kết quả đánh giá (Đồ thị Loss/Accuracy, Confusion Matrix, Classification Report). |

---

## 🔬 2. Danh Mục 15 Lớp Bệnh & Phác Đồ Điều Trị, Phục Hồi Cây

Tập dữ liệu bao gồm 3 loại cây trồng chính: **Ớt chuông (Bell Pepper)**, **Khoai tây (Potato)**, **Cà chua (Tomato)**:

### A. Ớt chuông (Bell Pepper)
1. **`Pepper__bell___Bacterial_spot` (Đốm vi khuẩn trên ớt chuông)**
   * *Tác nhân*: Vi khuẩn *Xanthomonas campestris pv. vesicatoria*.
   * *Triệu chứng*: Đốm ngậm nước 1-3mm, chuyển màu nâu đen, quầng vàng hẹp, mặt dưới lá hơi gồ dạng nốt sần, làm rụng lá ồ ạt.
   * *Phác đồ điều trị*: Cắt tỉa cành bệnh; tránh tưới phun mưa; phun thuốc gốc Đồng (*Copper Hydroxide*, *Kocide*) kết hợp kháng sinh thực vật (*Kasugamycin*, *Starner*).
   * *Phục hồi sau bệnh*: Bổ sung phân bón lá Canxi-Bo và Silic làm dày tế bào; tưới gốc Axit Humic kích rễ phục hồi.
2. **`Pepper__bell___healthy` (Ớt chuông khỏe mạnh)**
   * Lá xanh bóng đồng đều, sinh trưởng tốt. Duy trì bón lót Trichoderma phòng nấm rễ.

---

### B. Khoai tây (Potato)
3. **`Potato___Early_blight` (Đốm vòng / Cháy lá sớm trên khoai tây)**
   * *Tác nhân*: Nấm *Alternaria solani*.
   * *Triệu chứng*: Xuất hiện ở lá già dưới gốc; đốm nâu đen có **VÒNG ĐỒNG TÂM HÌNH TÂM BIA BẮN (Target board)**, quầng vàng xung quanh.
   * *Phác đồ điều trị*: Luân canh cây trồng; nhặt sạch lá già sát đất; phun *Mancozeb*, *Chlorothalonil* phòng ngừa, phun *Difenoconazole* (Score) hoặc *Azoxystrobin* khi chớm bệnh.
   * *Phục hồi sau bệnh*: Bổ sung Kali trắng ($K_2SO_4$) và Bo để đẩy nhanh quá trình phình củ bù đắp năng suất.
4. **`Potato___Late_blight` (Mốc sương / Cháy lá muộn trên khoai tây)**
   * *Tác nhân*: Nấm thủy mi *Phytophthora infestans*.
   * *Triệu chứng*: **CỰC KỲ NGUY HIỂM**. Vết loang ủng nước màu xám đen, lan siêu tốc làm cháy rụi toàn cây; sáng sớm có lớp **MỐC TRẮNG MỎNG NHƯ BÔNG** ở mặt dưới lá; củ bị thối nâu khô.
   * *Phác đồ điều trị*: Cắt ly khẩn cấp; phun ngay hoạt chất nội hấp cực mạnh: *Metalaxyl + Mancozeb* (Ridomil Gold), *Dimethomorph*, *Cymoxanil*.
   * *Phục hồi sau bệnh*: Cắt bỏ cành chết; phun phân vi lượng Kẽm (Zn) và Magie (Mg) tái tạo diệp lục.
5. **`Potato___healthy` (Khoai tây khỏe mạnh)**
   * Tán lá nguyên vẹn, sinh trưởng bình thường.

---

### C. Cà chua (Tomato)
6. **`Tomato_Bacterial_spot` (Đốm vi khuẩn trên cà chua)**
   * *Tác nhân*: Vi khuẩn *Xanthomonas spp.*
   * *Triệu chứng*: Đốm nâu đen 1-3mm, **TÂM ĐỐM DỄ KHÔ RỤNG TẠO LỖ THỦNG NHƯ VẾT ĐẠN BẮN (Shot-hole)**; trên quả có nốt sần mụn cóc.
   * *Phác đồ*: Phun thuốc gốc Đồng + *Kasugamycin* hoặc *Bismerthiazol*.
7. **`Tomato_Early_blight` (Đốm vòng sớm trên cà chua)**
   * *Tác nhân*: Nấm *Alternaria solani*.
   * *Triệu chứng*: Đốm khô lớn (đến 1.5cm) có **VÂN ĐỒNG TÂM RÕ RỆT**, bắt đầu từ lá gốc.
   * *Phác đồ*: Vặt lá già chân gốc; phủ bạt nilon luống; phun *Difenoconazole*, *Amistar Top*.
8. **`Tomato_Late_blight` (Mốc sương muộn trên cà chua)**
   * *Tác nhân*: *Phytophthora infestans*.
   * *Triệu chứng*: Vết úng nước thâm đen loang nhanh, mặt dưới có mốc trắng bông tuyết khi ẩm, thân bị tóp đen gãy gục, quả chai sần nâu.
   * *Phác đồ*: Phun khẩn cấp *Metalaxyl*, *Dimethomorph*.
9. **`Tomato_Leaf_Mold` (Mốc lá cà chua)**
   * *Tác nhân*: Nấm *Passalora fulva*.
   * *Triệu chứng*: Phổ biến trong nhà kính/nhà màng ẩm; mặt trên lá có đốm vàng loang lổ; **MẶT DƯỚI LÁ NỞ LỚP MỐC MÀU XANH Ô LIU ĐẾN NÂU NHUNG MỊN**.
   * *Phác đồ*: Hạ độ ẩm nhà màng (< 85%), tăng thông gió; phun *Chlorothalonil*, *Difenoconazole* ướt đều mặt dưới lá.
10. **`Tomato_Septoria_leaf_spot` (Đốm lá Septoria)**
    * *Tác nhân*: Nấm *Septoria lycopersici*.
    * *Triệu chứng*: Vô số đốm nhỏ li ti (1-3mm), **VIỀN NÂU ĐẬM, TÂM MÀU XÁM TRẮNG CÓ CHẤM ĐEN LI TI (Ổ BÀO TỬ PYCNIDIA)**; làm rụng lá trơ cành.
    * *Phác đồ*: Lót bạt luống; ngắt lá bệnh sát gốc; phun *Chlorothalonil* (Daconil).
11. **`Tomato_Spider_mites_Two_spotted_spider_mite` (Nhện đỏ hai chấm hại cà chua)**
    * *Tác nhân*: Côn trùng *Tetranychus urticae*.
    * *Triệu chứng*: Mặt trên lá bị châm kim lấm tấm màu vàng/trắng bạc; **MẶT DƯỚI LÁ CÓ MẠNG TƠ MỎNG VÀ CÁC CHẤM LI TI DI CHUYỂN**. Lá khô ráp như giấy giòn.
    * *Phác đồ*: Tăng độ ẩm phun sương; phun dầu khoáng sinh học, dầu Neem; phun thuốc đặc trị nhện: *Abamectin*, *Spiromesifen*. Phun kỹ mặt dưới lá.
12. **`Tomato__Target_Spot` (Đốm mắt cua / Đốm đích)**
    * *Tác nhân*: Nấm *Corynespora cassiicola*.
    * *Triệu chứng*: Đốm tròn màu nâu có vân đồng tâm mờ hơn Early Blight, phân bố cả ở tầng lá giữa, tâm đốm rất dễ nứt rách.
    * *Phác đồ*: Phun *Pyraclostrobin*, *Azoxystrobin*.
13. **`Tomato__Tomato_mosaic_virus` (Virus khảm cà chua - ToMV)**
    * *Tác nhân*: Virus ToMV.
    * *Triệu chứng*: Mặt lá loang lổ HOA VĂN KHẢM (xanh đậm xen xanh vàng nhạt), lá xoăn dạng lá liễu hẹp, cây còi cọc.
    * *Phác đồ*: **KHÔNG CÓ THUỐC ĐẶC TRỊ VIRUS**. Nhổ bỏ ngay lập tức cây bệnh đem đốt; khử trùng tay/dụng cụ bằng sữa không đường hoặc xà phòng; người hút thuốc lá không được chạm vào cây.
14. **`Tomato__Tomato_YellowLeaf__Curl_Virus` (Virus xoăn vàng lá cà chua - TYLCV)**
    * *Tác nhân*: Virus TYLCV do **Bọ phấn trắng (*Bemisia tabaci*)** truyền bệnh.
    * *Triệu chứng*: Mép lá non **UỐN CONG HÌNH CHIẾC CHÉN/CHIẾC THUYỀN**, lá nhỏ lại, **VÀNG RỰC TOÀN BỘ VIỀN VÀ PHIẾN LÁ**, cây lùn sít chùm, hoa rụng sạch.
    * *Phác đồ*: Nhổ bỏ cây bệnh; căng lưới chắn côn trùng 50 mesh; treo bẫy dính màu vàng; phun thuốc diệt bọ phấn trắng: *Pymetrozine*, *Thiamethoxam*, *Movento*.
15. **`Tomato_healthy` (Cà chua khỏe mạnh)**
    * Sinh trưởng bình thường, diệp lục đồng đều.

---

## ⚖️ 3. Đánh Giá Điểm Tương Đồng Dễ Nhầm Lẫn Giữa Các Bệnh (Confusion Analysis)

Trong học máy thị giác máy tính và thực địa, các cặp bệnh sau đây có biểu hiện rất giống nhau dẫn đến nguy cơ nhầm lẫn cao:

| Nhóm bệnh tương đồng | Triệu chứng giống nhau | Điểm mấu chốt để phân biệt chính xác |
| :--- | :--- | :--- |
| **Tomato Early Blight vs Late Blight** | Đều gây cháy lá màu nâu đen, rụi lá. | • **Early Blight**: Khô ráo, có **vân đồng tâm hình tâm bia**, không có mốc trắng.<br>• **Late Blight**: Úng nước loang lổ, **mặt dưới có mốc trắng bông tuyết**, lây lan cực nhanh. |
| **Tomato Septoria vs Bacterial Spot** | Đều là đốm nhỏ 1-3mm màu nâu sẫm trên lá. | • **Septoria**: Tâm màu **xám trắng với chấm đen li ti (ổ bào tử)**, viền nâu đậm sắc nét.<br>• **Bacterial Spot**: Tâm khô giòn **rụng thủng thành lỗ như vết đạn bắn**. |
| **Tomato Mosaic Virus vs TYLCV (Xoăn vàng lá)** | Đều do virus làm lá cong queo, biến dạng, cây lùn. | • **ToMV**: Phiến lá có hoa văn khảm loang lổ (xanh đậm xen nhạt), lá hẹp dài như lá liễu.<br>• **TYLCV**: Mép lá **uốn cong hình chiếc chén ngửa lên**, phiến lá non **vàng rực toàn phần**, không có hoa văn khảm. |
| **Tomato Leaf Mold vs Nhện Đỏ hại lá** | Đều làm mặt trên lá vàng úa và mặt dưới có màng bám. | • **Leaf Mold**: Mặt dưới có **lớp mốc nỉ nhung màu xanh ô liu/nâu hung** do nấm.<br>• **Nhện Đỏ**: Mặt trên có **chấm kim lấm tấm**, mặt dưới có **mạng tơ nhện mỏng** và con nhện li ti di chuyển. |

---

## 🚀 4. Hướng Dẫn Chạy Chương Trình

### 1. Khởi chạy Ứng dụng Web trực quan (Streamlit)
Mở terminal và gõ lệnh:
```bash
streamlit run app_streamlit.py
```
*Trình duyệt sẽ tự động mở giao diện tại địa chỉ `http://localhost:8501`.*

### 2. Huấn luyện lại mô hình (Tùy chọn)
Nếu muốn huấn luyện lại mô hình với các tham số khác:
```bash
# Huấn luyện nhanh trên CPU (mặc định 4 epochs, 80 ảnh/lớp)
python train_models.py --epochs 5 --samples 100 --batch_size 32
```
