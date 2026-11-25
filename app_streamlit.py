
# -*- coding: utf-8 -*-
"""
app_streamlit.py

He thong nhan dien benh la cay - MobileNetV2 + Grad-CAM

Ho tro 9 loai cay:
- Tomato, Potato, Bell Pepper
- Chili, Coffee, Corn, Rubber, Mango, Rice

Giu cac chuc nang:
1. Nhan dien anh bang AI
2. Grad-CAM
3. Bao cao huan luyen
4. Cam nang benh hoc
5. Phan tich benh de nham lan
"""

import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parent
PLANT = ROOT / "PlantVillage"
RUNS = ROOT / "training_runs"

os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")

from model_architecture import get_mobilenet_v2_model
from explainability import GradCAM
from disease_knowledge import DISEASE_DATABASE, SIMILARITY_GROUPS


# ============================================================
# 1. CAU HINH GIAO DIEN
# ============================================================

st.set_page_config(
    page_title="Chẩn đoán Bệnh Cây Trồng AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #2e7d32, #1b5e20);
    padding: 24px;
    border-radius: 12px;
    color: white;
    margin-bottom: 25px;
}
.main-header h1 {
    color: white;
    margin: 0;
    font-size: 28px;
}
.main-header p {
    color: #c8e6c9;
    margin: 8px 0 0 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. TEN CAY VA DUONG DAN ANH
# ============================================================

CROP_NAMES = {
    "Tomato": "Cà chua",
    "Potato": "Khoai tây",
    "Pepper": "Ớt chuông",
    "Chili": "Ớt cay",
    "Coffee": "Cà phê",
    "Corn": "Ngô",
    "Rubber": "Cao su",
    "Mango": "Xoài",
    "Rice": "Lúa nước",
}


def crop_of_class(class_name):
    for prefix, crop in CROP_NAMES.items():
        if class_name.startswith(prefix):
            return crop.lower()
    return "khác"


# Duong dan ca phe co HAI dau cach sau "Machine"
COFFEE_DIR = (
    PLANT
    / "A Machine  Learning Dataset for Classification "
      "of Common Coffee Leaf Diseases in Uganda"
    / "coffee dataset"
)

NEW_CLASS_DIRS = {
    # OT CAY
    "Chili___Bacterial_spot":
        PLANT / "chileleaf" / "Bacterial_Spot",
    "Chili___Cercospora_leaf_spot":
        PLANT / "chileleaf" / "Cercospora_Leaf_Spot",
    "Chili___Curl_virus":
        PLANT / "chileleaf" / "Curl_Virus",
    "Chili___healthy":
        PLANT / "chileleaf" / "Healthy_Leaf",

    # CA PHE
    "Coffee___healthy":
        COFFEE_DIR / "Health leaves",
    "Coffee___Rust":
        COFFEE_DIR / "leaf rust",
    "Coffee___Phoma":
        COFFEE_DIR / "phoma",

    # NGO
    "Corn___Blight":
        PLANT / "corndisease" / "Blight",
    "Corn___Common_rust":
        PLANT / "corndisease" / "Common_Rust",
    "Corn___Gray_leaf_spot":
        PLANT / "corndisease" / "Gray_Leaf_Spot",
    "Corn___healthy":
        PLANT / "corndisease" / "Healthy",

    # CAO SU - CHI RAW
    "Rubber___Anthracnose":
        PLANT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Anthracnose",
    "Rubber___Dry_leaf":
        PLANT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Dry_Leaf",
    "Rubber___healthy":
        PLANT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Healthy",
    "Rubber___Leaf_spot":
        PLANT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Leaf_Spot",

    # XOAI
    "Mango___Anthracnose":
        PLANT / "Mango Dataset" / "Anthracnose",
    "Mango___Bacterial_canker":
        PLANT / "Mango Dataset" / "Bacterial Canker",
    "Mango___Cutting_weevil":
        PLANT / "Mango Dataset" / "Cutting Weevil",
    "Mango___Die_back":
        PLANT / "Mango Dataset" / "die back",
    "Mango___Gall_midge":
        PLANT / "Mango Dataset" / "Gall Midge",
    "Mango___healthy":
        PLANT / "Mango Dataset" / "Healthy",
    "Mango___Powdery_mildew":
        PLANT / "Mango Dataset" / "Powdery Mildew",
    "Mango___Sooty_mould":
        PLANT / "Mango Dataset" / "Sooty Mould",

    # LUA
    "Rice___Bacterial_leaf_blight":
        PLANT / "rice_leaf_diseases"
        / "Bacterial leaf blight",
    "Rice___Brown_spot":
        PLANT / "rice_leaf_diseases" / "Brown spot",
    "Rice___Leaf_smut":
        PLANT / "rice_leaf_diseases" / "Leaf smut",
}

# Giu nguyen bay vi du cu cua giao dien
OLD_EXAMPLES = {
    "Cà chua - Đốm vòng sớm (Early Blight)":
        "Tomato_Early_blight",
    "Cà chua - Mốc sương muộn (Late Blight)":
        "Tomato_Late_blight",
    "Cà chua - Xoăn vàng lá (TYLCV Virus)":
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Khoai tây - Đốm vòng (Early Blight)":
        "Potato___Early_blight",
    "Ớt chuông - Đốm vi khuẩn (Bacterial Spot)":
        "Pepper__bell___Bacterial_spot",
    "Ớt chuông - Lá khỏe mạnh (Healthy)":
        "Pepper__bell___healthy",
    "Cà chua - Lá khỏe mạnh (Healthy)":
        "Tomato_healthy",
}

IMAGE_EXT = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


# ============================================================
# 3. TIM CHECKPOINT MOI NHAT
# ============================================================

def find_checkpoints():
    checkpoints = []

    old_path = ROOT / "best_plant_model.pth"

    if old_path.is_file():
        checkpoints.append(old_path)

    if RUNS.is_dir():
        checkpoints.extend(
            p for p in RUNS.glob(
                "*/best_plant_model.pth"
            )
            if p.is_file()
        )

    return sorted(
        checkpoints,
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )


@st.cache_resource(show_spinner="Đang tải mô hình AI...")
def load_trained_model(path_text, modified_time):
    """
    modified_time la tham so cache:
    checkpoint thay doi thi mo hinh duoc tai lai.
    """
    device = torch.device("cpu")

    try:
        checkpoint = torch.load(
            path_text,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        # Tuong thich ban PyTorch cu.
        checkpoint = torch.load(
            path_text,
            map_location=device
        )

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint khong dung dinh dang.")

    class_names = checkpoint.get("class_names")
    state = checkpoint.get("model_state_dict")

    if not class_names:
        raise ValueError(
            "Checkpoint khong chua class_names."
        )

    if state is None:
        raise ValueError(
            "Checkpoint khong chua model_state_dict."
        )

    if len(class_names) != len(set(class_names)):
        raise ValueError(
            "Checkpoint chua ten lop bi trung."
        )

    model = get_mobilenet_v2_model(
        num_classes=len(class_names),
        pretrained=False
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, list(class_names)


# ============================================================
# 4. DOC ANH MAU TU THU MUC THUC TE
# ============================================================

@st.cache_data(show_spinner=False)
def find_readable_image(folder_text):
    """Tim anh dau tien ma Pillow doc duoc hoan toan."""

    folder = Path(folder_text)

    if not folder.is_dir():
        return None

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXT:
            continue

        try:
            with Image.open(path) as image:
                image.load()
                image.convert("RGB")

            return str(path)

        except Exception:
            # Khong su dung anh hong lam anh mau.
            continue

    return None


@st.cache_data(show_spinner=False)
def get_sample_options():
    """
    Giu 7 vi du cu va them cac vi du cua 26 lop moi.
    Khong duong dan nao duoc tao tu ten nhan sai.
    """
    options = {}

    for label, class_name in OLD_EXAMPLES.items():
        folder = PLANT / class_name
        image_path = find_readable_image(str(folder))

        if image_path:
            options[label] = {
                "class_name": class_name,
                "path": image_path
            }

    for class_name, folder in NEW_CLASS_DIRS.items():
        image_path = find_readable_image(str(folder))

        if not image_path:
            continue

        info = DISEASE_DATABASE.get(class_name, {})
        crop = info.get(
            "crop",
            crop_of_class(class_name)
        )
        disease = info.get(
            "vietnamese_name",
            class_name
        )

        label = f"{crop} - {disease}"

        options[label] = {
            "class_name": class_name,
            "path": image_path
        }

    return options


# ============================================================
# 5. TIEN XU LY ANH
# ============================================================

def get_inference_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def show_dash_list(items):
    for item in items:
        st.markdown(f"- {item}")


# ============================================================
# 6. HEADER VA CHON MO HINH
# ============================================================

st.markdown("""
<div class="main-header">
  <h1>🌱 HỆ THỐNG CHẨN ĐOÁN BỆNH CÂY TRỒNG AI</h1>
  <p>
    MobileNetV2 (Transfer Learning) |
    Explainable AI (Grad-CAM) |
    Cẩm nang điều trị và phục hồi cây
  </p>
</div>
""", unsafe_allow_html=True)

checkpoints = find_checkpoints()

model = None
class_names = []
selected_checkpoint = None

if checkpoints:
    selected_checkpoint = st.sidebar.selectbox(
        "🧠 Chọn mô hình AI:",
        checkpoints,
        format_func=lambda p: str(
            p.relative_to(ROOT)
        ),
        help=(
            "Ưu tiên checkpoint mới nhất. "
            "Bạn cũng có thể chọn mô hình cũ."
        )
    )

    try:
        model, class_names = load_trained_model(
            str(selected_checkpoint),
            selected_checkpoint.stat().st_mtime_ns
        )

    except Exception as exc:
        st.error(
            f"Không tải được checkpoint: {exc}"
        )
        st.stop()

    st.sidebar.success(
        f"Đã tải mô hình {len(class_names)} lớp"
    )

    st.sidebar.caption(
        f"Checkpoint: {selected_checkpoint}"
    )

    if len(class_names) < 41:
        st.warning(
            f"⚠️ Mô hình hiện tại chỉ có "
            f"{len(class_names)} lớp. "
            "Các ảnh cây mới vẫn có thể xuất hiện "
            "trong danh sách kiểm thử, nhưng mô hình "
            "chưa thể dự đoán đúng những nhãn "
            "không có trong checkpoint."
        )

else:
    st.sidebar.warning(
        "Chưa tìm thấy checkpoint MobileNetV2."
    )


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📌 Menu Đồ Án")

    menu = st.radio(
        "Chọn chức năng:",
        [
            "🌿 Chẩn đoán Bệnh từ Ảnh (AI)",
            "📊 Báo cáo Đánh giá Học máy",
            "📖 Cẩm nang Tra cứu Bệnh Toàn diện",
            "⚖️ Phân tích Bệnh Dễ Nhầm Lẫn"
        ]
    )

    st.markdown("---")

    confidence_threshold = st.slider(
        "Ngưỡng cảnh báo độ tin cậy thấp (%):",
        min_value=30,
        max_value=90,
        value=60,
        step=5
    )

    st.caption(
        "Đây là ngưỡng cảnh báo dựa trên Softmax, "
        "không phải bộ phát hiện bệnh lạ đã được "
        "kiểm định độc lập."
    )

    st.markdown("---")
    st.markdown("### 🎓 Thông tin đồ án")
    st.markdown(
        "**Mô hình:** MobileNetV2\n\n"
        "**Giải thích AI:** Grad-CAM\n\n"
        f"**Số lớp checkpoint:** {len(class_names)}\n\n"
        "**Số loại cây hỗ trợ trong dữ liệu:** 9\n\n"
        "**Framework:** PyTorch & Streamlit"
    )


# ============================================================
# 8. CHUC NANG CHAN DOAN ANH
# ============================================================

if menu == "🌿 Chẩn đoán Bệnh từ Ảnh (AI)":

    st.subheader(
        "📸 Nạp ảnh lá cây để hệ thống phân tích"
    )

    input_method = st.radio(
        "Chọn nguồn ảnh:",
        [
            "📁 Tải ảnh từ máy tính",
            "📷 Chụp trực tiếp bằng Camera",
            "🧪 Chọn ảnh mẫu có sẵn trong tập dữ liệu"
        ],
        horizontal=True
    )

    img_to_process = None
    expected_class = None

    if input_method == "📁 Tải ảnh từ máy tính":

        uploaded_file = st.file_uploader(
            "Chọn ảnh lá cây:",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            try:
                with Image.open(uploaded_file) as image:
                    img_to_process = image.convert("RGB")

            except Exception as exc:
                st.error(
                    f"Không đọc được ảnh: {exc}"
                )

    elif input_method == "📷 Chụp trực tiếp bằng Camera":

        camera_file = st.camera_input(
            "Chụp ảnh lá cây:"
        )

        if camera_file is not None:
            try:
                with Image.open(camera_file) as image:
                    img_to_process = image.convert("RGB")

            except Exception as exc:
                st.error(
                    f"Không đọc được ảnh camera: {exc}"
                )

    else:
        samples = get_sample_options()

        if not samples:
            st.warning(
                "Không tìm thấy ảnh mẫu hợp lệ. "
                "Hãy kiểm tra thư mục PlantVillage."
            )

        else:
            sample_choice = st.selectbox(
                "Chọn ảnh kiểm thử nhanh:",
                list(samples.keys())
            )

            sample = samples[sample_choice]

            expected_class = sample["class_name"]
            sample_path = sample["path"]

            st.caption(
                f"Ảnh mẫu: {sample_path}"
            )

            try:
                with Image.open(sample_path) as image:
                    img_to_process = image.convert("RGB")

            except Exception as exc:
                st.error(
                    f"Ảnh mẫu không còn đọc được: {exc}"
                )

            if expected_class not in class_names:
                st.warning(
                    "Nhãn ảnh mẫu này chưa có trong "
                    "checkpoint đang chọn. "
                    "Kết quả dự đoán sẽ không phản ánh "
                    "khả năng nhận diện lớp mới."
                )

    if img_to_process is not None:

        st.markdown("---")

        if model is None:
            st.error(
                "Chưa có mô hình. Hãy kiểm tra "
                "thư mục training_runs."
            )
            st.stop()

        transform = get_inference_transform()
        input_tensor = transform(
            img_to_process
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(
                outputs, dim=1
            ).squeeze(0).cpu().numpy()

        # Tinh xac suat theo tat ca 9 loai cay
        crop_scores = {}

        for idx, probability in enumerate(probs):
            class_name = class_names[idx]
            crop = crop_of_class(class_name)

            crop_scores[crop] = (
                crop_scores.get(crop, 0.0)
                + float(probability)
            )

        likely_crop = max(
            crop_scores,
            key=crop_scores.get
        ) if crop_scores else "khác"

        top_indices = np.argsort(probs)[::-1][:3]
        best_idx = int(top_indices[0])

        predicted_class = class_names[best_idx]
        confidence = float(probs[best_idx]) * 100

        disease_info = DISEASE_DATABASE.get(
            predicted_class
        )

        low_confidence = (
            confidence < confidence_threshold
        )

        if low_confidence:
            st.warning(
                f"⚠️ Độ tin cậy thấp: {confidence:.2f}%. "
                "Ảnh có thể khó phân loại hoặc nằm ngoài "
                "phạm vi mô hình đã học."
            )

        st.subheader(
            "🎯 Kết quả nhận diện & Grad-CAM"
        )

        col1, col2, col3 = st.columns(
            [1, 1, 1.2]
        )

        with col1:
            st.markdown("**Ảnh gốc đầu vào:**")
            st.image(
                img_to_process,
                width="stretch"
            )

        with col2:
            st.markdown(
                "**Vùng AI tập trung phân tích:**"
            )

            grad_cam = None

            try:
                target_layer = model.features[-1]

                grad_cam = GradCAM(
                    model, target_layer
                )

                cam, _ = grad_cam.generate_heatmap(
                    input_tensor,
                    target_class=best_idx
                )

                blended_cam, _ = (
                    grad_cam.overlay_heatmap_on_image(
                        img_to_process,
                        cam,
                        alpha=0.55
                    )
                )

                st.image(
                    blended_cam,
                    width="stretch"
                )

                st.caption(
                    "Grad-CAM thể hiện vùng mô hình "
                    "chú ý, không phải bằng chứng "
                    "xác nhận chính xác vị trí bệnh."
                )

            except Exception as exc:
                st.warning(
                    f"Chưa tạo được Grad-CAM: {exc}"
                )

            finally:
                if grad_cam is not None:
                    grad_cam.remove_hooks()

        with col3:
            st.markdown(
                "**Thông tin chẩn đoán sơ bộ:**"
            )

            disease_name = (
                disease_info["vietnamese_name"]
                if disease_info else predicted_class
            )

            if low_confidence:
                st.warning(disease_name)

            elif "healthy" in predicted_class.lower():
                st.success(disease_name)

            else:
                st.error(disease_name)

            st.metric(
                "Độ tin cậy (Confidence)",
                f"{confidence:.2f}%"
            )

            st.markdown(
                f"**Loại cây được dự đoán:** "
                f"{likely_crop.title()}"
            )

            if disease_info:
                st.markdown(
                    f"**Cây trồng:** {disease_info['crop']}"
                )
                st.markdown(
                    f"**Tác nhân:** "
                    f"{disease_info['pathogen_type']}"
                )
                st.markdown(
                    f"**Mức độ nguy hại:** "
                    f"{disease_info['danger_level']}"
                )

            st.markdown("**Top 3 dự đoán:**")

            for idx in top_indices:
                class_name = class_names[int(idx)]
                score = float(probs[int(idx)])

                info = DISEASE_DATABASE.get(
                    class_name, {}
                )

                name = info.get(
                    "vietnamese_name",
                    class_name
                )

                st.markdown(
                    f"- **{name}**: {score * 100:.2f}%"
                )

                st.progress(score)

        # Canh bao neu mau co nhan that
        if expected_class is not None:

            st.markdown("---")

            if expected_class == predicted_class:
                st.success(
                    "✅ Kết quả dự đoán khớp với "
                    "nhãn của ảnh mẫu."
                )

            elif expected_class in class_names:
                st.warning(
                    "⚠️ Kết quả dự đoán chưa khớp "
                    "với nhãn của ảnh mẫu."
                )

            expected_info = DISEASE_DATABASE.get(
                expected_class, {}
            )

            st.caption(
                "Nhãn dữ liệu: "
                + expected_info.get(
                    "vietnamese_name",
                    expected_class
                )
            )

        # Goi y cac benh chua duoc model hoc
        if low_confidence:

            st.markdown("---")
            st.subheader(
                "💡 Các bệnh khác có thể tham khảo"
            )

            other_diseases = {
                key: value
                for key, value in DISEASE_DATABASE.items()
                if key not in class_names
                and crop_of_class(key) == likely_crop
            }

            if other_diseases:
                for key, info in other_diseases.items():
                    with st.expander(
                        f"📌 {info['vietnamese_name']}"
                    ):
                        st.markdown(
                            "**Triệu chứng:**"
                        )
                        show_dash_list(
                            info["symptoms"]
                        )

                        st.markdown(
                            "**Hướng xử lý canh tác:**"
                        )
                        show_dash_list(
                            info["treatment_protocol"][
                                "farming_practices"
                            ]
                        )

            else:
                st.info(
                    "Chưa có bệnh tham khảo khác "
                    "cho loại cây này."
                )

        # Cam nang chuyen sau
        if disease_info:

            st.markdown("---")
            st.subheader(
                "📘 Cẩm nang bệnh học & hướng dẫn xử lý"
            )

            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Triệu chứng",
                "💊 Phác đồ điều trị",
                "🌱 Phục hồi & dưỡng cây",
                "🔍 Bệnh dễ nhầm lẫn"
            ])

            with tab1:
                st.markdown(
                    f"### {disease_info['vietnamese_name']}"
                )
                st.markdown(
                    f"**Tên khoa học:** "
                    f"{disease_info['scientific_name']}"
                )
                st.markdown(
                    f"**Tác nhân:** "
                    f"{disease_info['pathogen_type']}"
                )
                st.markdown(
                    f"**Mức độ:** "
                    f"{disease_info['danger_level']}"
                )
                show_dash_list(
                    disease_info["symptoms"]
                )

            with tab2:
                st.markdown(
                    "### Phác đồ xử lý 3 bước"
                )

                t1, t2, t3 = st.columns(3)
                protocol = disease_info[
                    "treatment_protocol"
                ]

                with t1:
                    st.markdown(
                        "#### 1. Canh tác và cách ly"
                    )
                    show_dash_list(
                        protocol["farming_practices"]
                    )

                with t2:
                    st.markdown(
                        "#### 2. Biện pháp sinh học"
                    )
                    show_dash_list(
                        protocol["biological_control"]
                    )

                with t3:
                    st.markdown(
                        "#### 3. Hóa học / Thuốc BVTV"
                    )
                    show_dash_list(
                        protocol["chemical_control"]
                    )

            with tab3:
                st.markdown(
                    "### Hướng dẫn phục hồi cây"
                )

                show_dash_list(
                    disease_info[
                        "post_disease_recovery"
                    ]
                )

                st.info(
                    "Các bộ phận đã hoại tử thường "
                    "không hồi phục; cần chăm sóc "
                    "để cây phát triển mô và lá mới "
                    "nếu còn khả năng sinh trưởng."
                )

            with tab4:
                st.markdown(
                    "### Phân tích bệnh dễ nhầm lẫn"
                )

                confusion = disease_info[
                    "confusion_analysis"
                ]

                similar = confusion[
                    "similar_diseases"
                ]

                st.markdown(
                    "**Các bệnh tương tự:** "
                    + (", ".join(similar) if similar else "Không có")
                )

                st.markdown(
                    "**Đặc điểm tương đồng:** "
                    + confusion["common_features"]
                )

                st.markdown(
                    "**Cách phân biệt:**"
                )

                show_dash_list(
                    confusion["key_distinctions"]
                )

        else:
            st.info(
                "Chưa có thông tin kiến thức bệnh "
                "cho nhãn dự đoán này."
            )


# ============================================================
# 9. BAO CAO DANH GIA
# ============================================================

elif menu == "📊 Báo cáo Đánh giá Học máy":

    st.subheader(
        "📊 Báo cáo thực nghiệm & đánh giá mô hình"
    )

    if selected_checkpoint is None:
        st.warning("Chưa có checkpoint.")
        st.stop()

    # File cu nam trong evaluation_results.
    # File moi nam ngay trong thu muc training_runs.
    if selected_checkpoint.parent == ROOT:
        report_dir = ROOT / "evaluation_results"
    else:
        report_dir = selected_checkpoint.parent

    st.caption(
        f"Đang đọc báo cáo tại: {report_dir}"
    )

    curves_path = report_dir / "training_curves.png"
    matrix_path = report_dir / "confusion_matrix.png"
    report_path = report_dir / "classification_report.txt"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "#### Đồ thị Loss / Accuracy"
        )

        if curves_path.is_file():
            st.image(
                str(curves_path),
                width="stretch"
            )
        else:
            st.info(
                "Chưa có biểu đồ huấn luyện "
                "trong thư mục này."
            )

    with col2:
        st.markdown(
            "#### Ma trận nhầm lẫn"
        )

        if matrix_path.is_file():
            st.image(
                str(matrix_path),
                width="stretch"
            )
        else:
            st.info(
                "Chưa có ma trận nhầm lẫn."
            )

    st.markdown("---")
    st.markdown(
        "#### Precision / Recall / F1-Score"
    )

    if report_path.is_file():
        report_text = report_path.read_text(
            encoding="utf-8"
        )
        st.code(report_text, language="text")
    else:
        st.warning(
            "Chưa có Classification Report. "
            "Checkpoint có thể được lưu trước "
            "khi quá trình đánh giá hoàn tất."
        )


# ============================================================
# 10. CAM NANG BENH HOC
# ============================================================

elif menu == "📖 Cẩm nang Tra cứu Bệnh Toàn diện":

    st.subheader(
        "📖 Cẩm nang bệnh học cho 9 loại cây trồng"
    )

    st.markdown(
        f"**{len(DISEASE_DATABASE)} mục kiến thức** | "
        f"**{len(class_names)} lớp trong checkpoint**"
    )

    col1, col2 = st.columns(2)

    with col1:
        crop_filter = st.selectbox(
            "Lọc theo loại cây:",
            [
                "Tất cả cây trồng",
                "Cà chua",
                "Khoai tây",
                "Ớt chuông",
                "Ớt cay",
                "Cà phê",
                "Ngô",
                "Cao su",
                "Xoài",
                "Lúa nước"
            ]
        )

    with col2:
        dataset_filter = st.selectbox(
            "Lọc theo trạng thái:",
            [
                "Tất cả",
                "Đã có trong checkpoint",
                "Chỉ tra cứu / chưa có trong checkpoint"
            ]
        )

    filtered_keys = []

    for key, info in DISEASE_DATABASE.items():

        crop = crop_of_class(key)

        if crop_filter != "Tất cả cây trồng":
            if crop != crop_filter.lower():
                continue

        trained = key in class_names

        if dataset_filter == "Đã có trong checkpoint":
            if not trained:
                continue

        if dataset_filter == "Chỉ tra cứu / chưa có trong checkpoint":
            if trained:
                continue

        filtered_keys.append(key)

    if not filtered_keys:
        st.warning(
            "Không tìm thấy bệnh phù hợp."
        )

    else:
        st.write(
            f"Tìm thấy **{len(filtered_keys)}** bệnh."
        )

        disease_key = st.selectbox(
            "Chọn bệnh:",
            filtered_keys,
            format_func=lambda key: (
                f"{'[ĐÃ TRAIN AI]' if key in class_names else '[TRA CỨU]'} "
                f"{DISEASE_DATABASE[key]['crop']} - "
                f"{DISEASE_DATABASE[key]['vietnamese_name']}"
            )
        )

        info = DISEASE_DATABASE[disease_key]

        if disease_key in class_names:
            st.success(
                "✅ Có trong checkpoint AI hiện tại"
            )
        else:
            st.info(
                "📖 Có thông tin trong cơ sở tri thức, "
                "nhưng chưa có trong checkpoint hiện tại."
            )

        st.markdown(
            f"### {info['vietnamese_name']}"
        )

        st.markdown(
            f"**Tên khoa học:** "
            f"{info['scientific_name']}"
        )

        st.markdown(
            f"**Tác nhân:** {info['pathogen_type']}"
        )

        st.markdown(
            f"**Mức độ:** {info['danger_level']}"
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Triệu chứng")
            show_dash_list(info["symptoms"])

            st.markdown("#### Phục hồi cây")
            show_dash_list(
                info["post_disease_recovery"]
            )

        with c2:
            st.markdown("#### Hướng xử lý")

            protocol = info["treatment_protocol"]

            st.markdown("**Canh tác:**")
            show_dash_list(
                protocol["farming_practices"]
            )

            st.markdown("**Sinh học:**")
            show_dash_list(
                protocol["biological_control"]
            )

            st.markdown("**Hóa học:**")
            show_dash_list(
                protocol["chemical_control"]
            )


# ============================================================
# 11. BENH DE NHAM LAN
# ============================================================

elif menu == "⚖️ Phân tích Bệnh Dễ Nhầm Lẫn":

    st.subheader(
        "⚖️ Phân tích và phân biệt bệnh tương đồng"
    )

    st.markdown(
        "Các bệnh có triệu chứng giống nhau "
        "có thể khiến mô hình phân loại nhầm."
    )

    for idx, group in enumerate(SIMILARITY_GROUPS):

        with st.expander(
            f"📌 {group['group_name']}",
            expanded=(idx == 0)
        ):

            st.markdown(
                "**Các lớp:** "
                + ", ".join(group["classes"])
            )

            st.markdown(
                "**Lý do dễ nhầm lẫn:**"
            )
            st.write(
                group["confusion_reason"]
            )

            st.markdown(
                "**Hướng dẫn phân biệt:**"
            )
            st.write(
                group["differentiation_guide"]
            )
