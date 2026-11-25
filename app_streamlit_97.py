
# -*- coding: utf-8 -*-
"""
STREAMLIT - NHAN DIEN BENH LA CAY 97 LOP
MobileNetV2 + Grad-CAM + Disease Knowledge + Test Report

Chay:
streamlit run app_streamlit_97.py
"""

import os
import csv
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "training_runs"
MANIFEST = (
    ROOT / "dataset_review"
    / "dataset_97_lop"
    / "split_97_lop"
    / "dataset_manifest_SPLIT_DU_KIEN.csv"
)

os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image, ImageOps
from torchvision import transforms

from model_architecture import get_mobilenet_v2_model
from explainability import GradCAM
from reference_gallery_97 import show_reference_gallery
from disease_knowledge_97 import DISEASE_DATABASE, SIMILARITY_GROUPS

from confusion_insights_97 import (
    show_confusion_insights,
    show_global_confusions
)

# ======================================================
# 1. GIAO DIEN VA DANH SACH 18 LOAI CAY
# ======================================================

st.set_page_config(
    page_title="Plant Disease AI - 97 Classes",
    page_icon="🌱",
    layout="wide"
)

CROPS = {
    "Banana": "Chuối",
    "Cashew": "Điều",
    "Chili": "Ớt cay",
    "Coffee": "Cà phê",
    "Corn": "Ngô",
    "Durian": "Sầu riêng",
    "Guava": "Ổi",
    "Jackfruit": "Mít",
    "Lemon": "Chanh",
    "Mango": "Xoài",
    "Papaya": "Đu đủ",
    "Pepper__bell": "Ớt chuông",
    "Potato": "Khoai tây",
    "Rice": "Lúa",
    "Rubber": "Cao su",
    "Sugarcane": "Mía",
    "Tea": "Chè",
    "Tomato": "Cà chua",
}

DISEASE_NAMES = {
    "healthy": "Lá khỏe mạnh",
    "Anthracnose": "Thán thư",
    "Bacterial_spot": "Đốm vi khuẩn",
    "Bacterial_blight": "Cháy lá vi khuẩn",
    "Bacterial_leaf_blight": "Bạc lá vi khuẩn",
    "Blight": "Cháy lá",
    "Early_blight": "Cháy lá sớm",
    "Late_blight": "Cháy lá muộn",
    "Brown_spot": "Đốm nâu",
    "Brown_rust": "Gỉ sắt nâu",
    "Common_rust": "Gỉ sắt",
    "Rust": "Gỉ sắt",
    "Leaf_spot": "Đốm lá",
    "Gray_leaf_spot": "Đốm lá xám",
    "Canker": "Loét",
    "Citrus_canker": "Loét cây có múi",
    "Curl": "Xoăn lá",
    "Curl_virus": "Virus xoăn lá",
    "Mosaic": "Khảm lá",
    "Ringspot": "Đốm vòng",
    "Sooty_mould": "Nấm bồ hóng",
    "Powdery_mildew": "Phấn trắng",
    "Mealybug": "Rệp sáp",
    "Mite_damage": "Nhện hại",
    "Spider_mites": "Nhện hại",
    "Red_spider_mite": "Nhện đỏ",
    "Dried_leaves": "Khô lá",
    "Dry_leaf": "Khô lá",
    "Deficiency_leaf": "Dấu hiệu thiếu dinh dưỡng",
    "Algal": "Đốm tảo",
    "Phoma": "Bệnh Phoma",
    "Sigatoka": "Bệnh Sigatoka",
    "Cordana": "Bệnh Cordana",
    "Leaf_smut": "Than lá",
    "Grassy_shoot": "Chồi cỏ",
    "Yellow_leaf": "Vàng lá",
    "Pokkah_boeng": "Bệnh Pokkah Boeng",
    "Smut": "Bệnh than",
}

# Nhan da tung duoc danh dau can xem lai
REVIEW_LABELS = {
    "Banana___healthy",
    "Guava___Dot",
    "Sugarcane___Sett_rot",
}


def crop_key(class_name):
    if class_name.startswith("Pepper__bell___"):
        return "Pepper__bell"
    if class_name.startswith("Tomato_"):
        return "Tomato"
    return class_name.split("___", 1)[0]


def class_display(class_name):
    info = DISEASE_DATABASE.get(class_name)

    if info and info.get("vietnamese_name"):
        return info["vietnamese_name"]

    crop = crop_key(class_name)
    crop_name = CROPS.get(crop, crop)

    if class_name.startswith("Tomato_"):
        disease = class_name[len("Tomato_"):].lstrip("_")
    else:
        disease = class_name.split("___", 1)[-1]

    description = DISEASE_NAMES.get(
        disease, disease.replace("_", " ")
    )

    return f"{crop_name} - {description}"


def needs_review(class_name):
    return (
        class_name.startswith("Papaya___")
        or class_name in REVIEW_LABELS
    )


# ======================================================
# 2. TIM VA TAI CHECKPOINT 97 LOP
# ======================================================

def find_models():
    candidates = []

    candidates.extend(
        RUNS.glob("plant97_full_*/best_plant_model.pth")
    )

    candidates.extend(
        RUNS.glob("plant97_*/best_plant_model.pth")
    )

    candidates = [
        p for p in candidates if p.is_file()
    ]

    return sorted(
        set(candidates),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_model(path_text, modified_time):
    device = torch.device("cpu")

    checkpoint = torch.load(
        path_text,
        map_location=device,
        weights_only=True
    )

    classes = list(checkpoint["class_names"])

    if len(classes) != 97 or len(set(classes)) != 97:
        raise ValueError(
            "Checkpoint khong co dung 97 lop!"
        )

    model = get_mobilenet_v2_model(
        num_classes=97,
        pretrained=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, classes


# ======================================================
# 3. ANH MAU - CHI LAY TU TRAIN
# ======================================================

@st.cache_data(show_spinner=False)
def sample_paths(manifest_mtime):
    examples = {}

    if not MANIFEST.is_file():
        return examples

    with MANIFEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"] != "train":
                continue

            name = row["class"]

            if name not in examples:
                examples[name] = row["path"]

    return examples


def inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def show_items(values):
    if not values:
        st.info("Chưa có thông tin chi tiết.")
        return

    for value in values:
        st.markdown(f"- {value}")


# ======================================================
# 4. CAM NANG BENH
# ======================================================

def show_disease_details(class_name):
    info = DISEASE_DATABASE.get(class_name)

    st.subheader("📘 Thông tin bệnh và hướng xử lý")

    if not info:
        st.warning(
            "Nhãn này đã được mô hình học, nhưng "
            "cơ sở tri thức chưa có hồ sơ bệnh "
            "được chuẩn hóa cho lớp này."
        )

        st.write("**Nhãn dữ liệu:**", class_name)

        st.info(
            "Nên quan sát cả hai mặt lá, kiểm tra "
            "nhiều lá trên cùng cây và đối chiếu "
            "triệu chứng với tài liệu chuyên ngành. "
            "Không nên quyết định sử dụng thuốc "
            "chỉ dựa trên một ảnh dự đoán."
        )
        return

    st.markdown(
        f"### {info.get('vietnamese_name', class_name)}"
    )

    st.write(
        "**Cây trồng:**",
        info.get("crop", "Chưa xác định")
    )

    st.write(
        "**Tác nhân:**",
        info.get("pathogen_type", "Chưa xác định")
    )

    st.write(
        "**Mức độ tham khảo:**",
        info.get("danger_level", "Chưa đánh giá")
    )

    tab1, tab2, tab3 = st.tabs([
        "🔎 Triệu chứng",
        "🌿 Phòng ngừa và chăm sóc",
        "⚖️ Bệnh dễ nhầm"
    ])

    with tab1:
        show_items(info.get("symptoms", []))

    with tab2:
        protocol = info.get("treatment_protocol", {})

        st.markdown("**Biện pháp canh tác:**")
        show_items(
            protocol.get("farming_practices", [])
        )

        st.markdown("**Biện pháp sinh học tham khảo:**")
        show_items(
            protocol.get("biological_control", [])
        )

        st.markdown("**Chăm sóc phục hồi:**")
        show_items(
            info.get("post_disease_recovery", [])
        )

        st.caption(
            "Các hướng dẫn mang tính tham khảo. "
            "Việc xử lý thực tế cần xác minh "
            "tác nhân và điều kiện cây trồng."
        )

    with tab3:
        confusion = info.get("confusion_analysis", {})

        similar = confusion.get(
            "similar_diseases", []
        )

        if similar:
            for name in similar:
                st.write("•", class_display(name))
        else:
            st.info("Chưa có danh sách bệnh tương tự.")

        for distinction in confusion.get(
            "key_distinctions", []
        ):
            st.markdown(f"- {distinction}")


# ======================================================
# 5. DU DOAN VA GRAD-CAM
# ======================================================

def diagnose(model, classes, report_dir):
    st.header("🌿 Nhận diện bệnh lá cây")

    st.caption(
        "Nhận diện 18 loại cây, 97 lớp bệnh "
        "và tình trạng lá. Kết quả AI chỉ "
        "mang tính sàng lọc và tham khảo."
    )

    crop_choice = st.selectbox(
        "Bạn có biết ảnh thuộc loại cây nào không?",
        ["auto"] + list(CROPS.keys()),
        format_func=lambda key: (
            "Để AI tự nhận diện loại cây"
            if key == "auto"
            else CROPS[key]
        )
    )

    source = st.radio(
        "Nguồn ảnh",
        [
            "Tải ảnh",
            "Camera",
            "Ảnh mẫu từ Train"
        ],
        horizontal=True
    )

    image = None
    expected_class = None

    if source == "Tải ảnh":
        uploaded = st.file_uploader(
            "Chọn ảnh lá cây",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded is not None:
            try:
                with Image.open(uploaded) as img:
                    image = ImageOps.exif_transpose(
                        img
                    ).convert("RGB")
            except Exception as exc:
                st.error(f"Không đọc được ảnh: {exc}")
                return

    elif source == "Camera":
        camera = st.camera_input(
            "Chụp ảnh lá cây"
        )

        if camera is not None:
            try:
                with Image.open(camera) as img:
                    image = ImageOps.exif_transpose(
                        img
                    ).convert("RGB")
            except Exception as exc:
                st.error(f"Không đọc được ảnh: {exc}")
                return

    else:
        if not MANIFEST.is_file():
            st.warning(
                "Không tìm thấy manifest ảnh mẫu."
            )
            return

        examples = sample_paths(
            MANIFEST.stat().st_mtime_ns
        )

        choices = [
            name for name in classes
            if name in examples
        ]

        if not choices:
            st.warning("Không có ảnh mẫu.")
            return

        expected_class = st.selectbox(
            "Chọn ảnh mẫu",
            choices,
            format_func=class_display
        )

        path = Path(examples[expected_class])

        try:
            with Image.open(path) as img:
                image = ImageOps.exif_transpose(
                    img
                ).convert("RGB")
        except Exception as exc:
            st.error(f"Không mở được ảnh mẫu: {exc}")
            return

        st.caption(
            "Ảnh mẫu thuộc tập Train, "
            "chỉ dùng demo chức năng. "
            "Không dùng để chứng minh "
            "độ chính xác trên Test."
        )

    if image is None:
        st.info("Hãy chọn hoặc chụp ảnh để bắt đầu.")
        return


    # ROI_GUARD_V3
    # ==============================================
    # 1. CHON VUNG LA CAN PHAN TICH
    # ==============================================

    crop_enabled = st.checkbox(
        "✂️ Cắt vùng lá trước khi nhận diện",
        value=False,
        help=(
            "Chọn vùng chứa lá cần kiểm tra, "
            "hạn chế nền xung quanh."
        )
    )

    analysis_image = image

    if crop_enabled:
        st.markdown(
            "### Chọn vùng ảnh cần phân tích"
        )

        c1, c2 = st.columns(2)

        with c1:
            horizontal = st.slider(
                "Khoảng ngang (%)",
                min_value=0,
                max_value=100,
                value=(0, 100),
                step=1
            )

        with c2:
            vertical = st.slider(
                "Khoảng dọc (%)",
                min_value=0,
                max_value=100,
                value=(0, 100),
                step=1
            )

        if (
            horizontal[1] - horizontal[0] < 10
            or vertical[1] - vertical[0] < 10
        ):
            st.error(
                "Vùng cắt quá nhỏ. "
                "Hãy chọn vùng rộng hơn."
            )
            return

        width, height = image.size

        left = int(
            width * horizontal[0] / 100
        )
        right = int(
            width * horizontal[1] / 100
        )

        top = int(
            height * vertical[0] / 100
        )
        bottom = int(
            height * vertical[1] / 100
        )

        right = max(left + 1, right)
        bottom = max(top + 1, bottom)

        analysis_image = image.crop(
            (left, top, right, bottom)
        )

        preview1, preview2 = st.columns(2)

        with preview1:
            st.image(
                image,
                caption="Ảnh gốc",
                width="stretch"
            )

        with preview2:
            st.image(
                analysis_image,
                caption="Vùng lá được chọn",
                width="stretch"
            )

        st.caption(
            "Hãy giữ nguyên các vết bệnh cần "
            "quan sát. Cắt ảnh có thể giảm nhiễu "
            "nhưng không đảm bảo tăng độ chính xác."
        )

    threshold = st.slider(
        "Ngưỡng cảnh báo Softmax thấp",
        30, 95, 60, 5
    ) / 100.0

    # ==============================================
    # 2. SUY LUAN TREN ANH GOC VA VUNG CAT
    # ==============================================

    transform = inference_transform()

    def predict_image(pil_image):
        input_tensor = transform(
            pil_image
        ).unsqueeze(0)

        with torch.inference_mode():
            output = model(input_tensor)

            scores = F.softmax(
                output, dim=1
            )[0].cpu().numpy()

        return scores, input_tensor

    with st.spinner("AI đang phân tích ảnh..."):
        if crop_enabled:
            before_probs, _ = predict_image(image)
        else:
            before_probs = None

        probabilities, tensor = predict_image(
            analysis_image
        )

    # ==============================================
    # 3. TINH DIEM CHO 18 LOAI CAY
    # ==============================================

    def score_by_crop(scores):
        result = {
            key: 0.0 for key in CROPS
        }

        for idx, name in enumerate(classes):
            crop = crop_key(name)

            if crop in result:
                result[crop] += float(
                    scores[idx]
                )

        return result

    crop_scores = score_by_crop(
        probabilities
    )

    ranked_crops = sorted(
        crop_scores,
        key=crop_scores.get,
        reverse=True
    )

    suggested_crop = ranked_crops[0]

    # ==============================================
    # 4. XAC NHAN LOAI CAY
    # ==============================================

    if crop_choice == "auto":

        st.info(
            f"AI gợi ý: {CROPS[suggested_crop]}. "
            "Đây chưa phải kết quả nhận diện "
            "loại cây đã được kiểm chứng."
        )

        st.markdown(
            "**Ba loại cây AI nghi ngờ nhất:**"
        )

        for crop in ranked_crops[:3]:
            st.write(
                f"- {CROPS[crop]}: "
                f"{crop_scores[crop] * 100:.2f}%"
            )

        image_key = hashlib.sha256(
            image.resize((48, 48)).tobytes()
        ).hexdigest()[:16]

        crop_options = list(CROPS.keys())
        default_idx = crop_options.index(suggested_crop) if suggested_crop in crop_options else 0

        selected_crop = st.selectbox(
            "Xác nhận loại cây trong ảnh (AI đã tự động chọn cây có khả năng cao nhất, bạn có thể đổi nếu muốn):",
            crop_options,
            index=default_idx,
            format_func=lambda key: CROPS.get(key, key),
            key=f"confirmed_crop_{image_key}"
        )

    else:
        selected_crop = crop_choice

    # ==============================================
    # 5. CHI XEP HANG BENH CUA CAY DA CHON
    # ==============================================

    allowed = [
        i for i, name in enumerate(classes)
        if crop_key(name) == selected_crop
    ]

    if not allowed:
        st.error(
            "Không có lớp bệnh thuộc cây đã chọn."
        )
        return

    top_ids = sorted(
        allowed,
        key=lambda i: float(probabilities[i]),
        reverse=True
    )[:3]

    best_id = top_ids[0]
    predicted = classes[best_id]

    confidence = float(
        probabilities[best_id]
    )

    second = (
        float(probabilities[top_ids[1]])
        if len(top_ids) > 1
        else 0.0
    )

    crop_mass = float(
        crop_scores[selected_crop]
    )

    # ==============================================
    # 6. SO SANH TRUOC VA SAU KHI CAT
    # ==============================================

    if crop_enabled and before_probs is not None:
        before_scores = score_by_crop(
            before_probs
        )

        before_suggested = max(
            before_scores,
            key=before_scores.get
        )

        before_allowed = sorted(
            allowed,
            key=lambda i: float(before_probs[i]),
            reverse=True
        )

        before_best = before_allowed[0]

        st.subheader(
            "📊 So sánh trước và sau khi cắt lá"
        )

        compare = pd.DataFrame([
            {
                "Chỉ số": "Cây AI gợi ý",
                "Ảnh gốc": CROPS[before_suggested],
                "Vùng cắt": CROPS[suggested_crop]
            },
            {
                "Chỉ số": "Tổng điểm cây đã xác nhận",
                "Ảnh gốc": (
                    f"{before_scores[selected_crop]*100:.2f}%"
                ),
                "Vùng cắt": (
                    f"{crop_mass*100:.2f}%"
                )
            },
            {
                "Chỉ số": "Lớp có điểm cao nhất trong cây",
                "Ảnh gốc": class_display(
                    classes[before_best]
                ),
                "Vùng cắt": class_display(
                    predicted
                )
            },
            {
                "Chỉ số": "Điểm lớp trên 97 lớp",
                "Ảnh gốc": (
                    f"{float(before_probs[before_best])*100:.2f}%"
                ),
                "Vùng cắt": (
                    f"{confidence*100:.2f}%"
                )
            }
        ])

        st.dataframe(
            compare,
            width="stretch",
            hide_index=True
        )

        st.caption(
            "Các điểm trên là Softmax chưa hiệu chuẩn. "
            "Điểm tăng sau khi cắt không tự chứng minh "
            "mô hình dự đoán đúng."
        )

    # ==============================================
    # 7. KIEM TRA DO PHU HOP CUA LOAI CAY
    # ==============================================

    if selected_crop != suggested_crop:
        st.warning(
            f"AI nghi ngờ {CROPS[suggested_crop]}, "
            f"nhưng bạn xác nhận "
            f"{CROPS[selected_crop]}."
        )

    uncertain = (
        confidence < threshold
        or confidence - second < 0.15
        or crop_mass < 0.50
    )

    if crop_mass < 0.50:
        st.warning(
            f"⚠️ **Lưu ý độ tin cậy:** Tổng điểm các lớp thuộc cây **{CROPS[selected_crop]}** chỉ đạt **{crop_mass * 100:.2f}%** (dưới 50%). "
            "Điều này thường xảy ra khi lá có phông nền phức tạp, góc chụp khác biệt, hoặc bệnh trên lá chưa có trong tập dữ liệu huấn luyện. "
            "Kết quả phân tích dưới đây mang tính chất tham khảo kỹ thuật, hãy đối chiếu thêm với cẩm nang triệu chứng."
        )

    # Dam bao anh hien thi va Grad-CAM su dung
    # cung vung anh da duoc phan tich.
    image = analysis_image
    if uncertain:
        st.warning(
            "⚠️ Kết quả chưa chắc chắn. "
            "Nên cung cấp ảnh rõ hơn hoặc "
            "đối chiếu triệu chứng thực tế."
        )

    if needs_review(predicted):
        st.warning(
            "⚠️ Nhãn này thuộc nhóm dữ liệu "
            "cần xác minh bổ sung. "
            "Không nên xem kết quả là kết luận bệnh."
        )

    left, middle, right = st.columns(
        [1, 1, 1.2]
    )

    with left:
        st.markdown("**Ảnh gốc**")
        st.image(image, width="stretch")

    with middle:
        st.markdown("**Grad-CAM**")

        grad_cam = None

        try:
            grad_cam = GradCAM(
                model,
                model.features[-1]
            )

            heatmap, _ = (
                grad_cam.generate_heatmap(
                    tensor,
                    target_class=best_id
                )
            )

            overlay, _ = (
                grad_cam.overlay_heatmap_on_image(
                    image,
                    heatmap,
                    alpha=0.55
                )
            )

            st.image(
                overlay,
                width="stretch"
            )

        except Exception as exc:
            st.warning(
                f"Không tạo được Grad-CAM: {exc}"
            )

        finally:
            if grad_cam is not None:
                grad_cam.remove_hooks()

        st.caption(
            "Grad-CAM thể hiện vùng ảnh "
            "ảnh hưởng đến dự đoán, không "
            "phải bản đồ chính xác của vết bệnh."
        )

    with right:
        st.markdown("**Kết quả dự đoán:**")

        if uncertain:
            st.warning(class_display(predicted))
        elif "healthy" in predicted.lower():
            st.success(class_display(predicted))
        else:
            st.info(class_display(predicted))

        st.metric(
            "Điểm Softmax trên 97 lớp",
            f"{confidence * 100:.2f}%"
        )

        st.caption(
            f"Tổng điểm cho cây "
            f"{CROPS[selected_crop]}: "
            f"{crop_mass * 100:.2f}%"
        )

        st.write(
            "**Loại cây:**",
            CROPS.get(
                crop_key(predicted),
                crop_key(predicted)
            )
        )

        st.markdown("**Top 3 lớp của cây đã chọn:**")

        for idx in top_ids:
            score = float(probabilities[idx])

            st.markdown(
                f"**{class_display(classes[idx])}** "
                f"— {score * 100:.2f}%"
            )

            st.progress(
                min(1.0, max(0.0, score))
            )

        with st.expander("🌐 Xem Top 3 dự đoán toàn hệ thống (97 lớp)", expanded=False):
            global_top3_ids = probabilities.argsort()[::-1][:3]
            for g_idx in global_top3_ids:
                st.write(f"- **{class_display(classes[g_idx])}**: {float(probabilities[g_idx])*100:.2f}%")

    st.caption(
        "Điểm Softmax chưa được hiệu chuẩn "
        "thành xác suất bệnh thực tế. "
        "Việc chọn loại cây chỉ lọc kết quả "
        "hiển thị, không làm tăng độ tin cậy."
    )

    if expected_class is not None:
        st.write(
            "**Nhãn ảnh mẫu:**",
            class_display(expected_class)
        )

        if expected_class == predicted:
            st.success("Dự đoán khớp nhãn ảnh mẫu.")
        else:
            st.warning(
                "Dự đoán không khớp nhãn ảnh mẫu."
            )

        # So sanh anh du doan voi 3 anh mau tu Train
    st.divider()

    show_reference_gallery(
        class_name=predicted,
        manifest_path=MANIFEST,
        exclude_path=(
            str(path)
            if source == "Ảnh mẫu từ Train"
            else None
        ),
        max_images=3
    )

    # Phan tich nham lan dua tren Test
    st.divider()

    show_confusion_insights(
        class_name=predicted,
        report_dir=report_dir,
        display_name=class_display,
        max_pairs=5
    )

    # Cam nang benh hoc
    st.divider()
    show_disease_details(predicted)


# ======================================================
# 6. BAO CAO TEST
# ======================================================

def show_report(report_dir):
    st.header("📊 Báo cáo đánh giá mô hình")

    st.caption(
        "Kết quả kiểm thử nội bộ trên tập Test. "
        "Một số nhãn còn cần xác minh; "
        "không phải kết quả kiểm định "
        "chẩn đoán ngoài thực địa."
    )

    summary_path = (
        report_dir / "test_summary.json"
    )

    if summary_path.is_file():
        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Test Accuracy",
            f"{summary['test_accuracy'] * 100:.2f}%"
        )

        c2.metric(
            "Test Macro F1",
            f"{summary['test_macro_f1'] * 100:.2f}%"
        )

        c3.metric(
            "Ảnh Test",
            f"{summary['test_images']:,}"
        )

        st.warning(
            "Trong tập Test dự kiến có 452 ảnh "
            "thuộc nhóm nhãn cần xác minh."
        )
    else:
        st.info("Chưa tìm thấy test_summary.json.")

    history_path = report_dir / "history.csv"

    if history_path.is_file():
        st.subheader("📈 Lịch sử huấn luyện")

        history = pd.read_csv(history_path)

        if {"epoch", "train_acc", "val_acc"}.issubset(
            history.columns
        ):
            st.line_chart(
                history.set_index("epoch")[
                    ["train_acc", "val_acc"]
                ]
            )

        if {
            "epoch", "val_macro_f1"
        }.issubset(history.columns):
            st.line_chart(
                history.set_index("epoch")[
                    ["val_macro_f1"]
                ]
            )

    matrix_path = (
        report_dir / "confusion_matrix.csv"
    )

    if matrix_path.is_file():
        st.subheader("⚖️ Các bệnh dễ nhầm lẫn")

        cm = pd.read_csv(matrix_path)

        actual_column = cm.columns[0]
        mistakes = []

        for _, row in cm.iterrows():
            actual = row[actual_column]

            for predicted in cm.columns[1:]:
                if actual == predicted:
                    continue

                count = int(row[predicted])

                if count > 0:
                    mistakes.append({
                        "Bệnh thực tế":
                            class_display(actual),
                        "Dự đoán nhầm":
                            class_display(predicted),
                        "Số ảnh": count
                    })

        if mistakes:
            mistakes.sort(
                key=lambda x: x["Số ảnh"],
                reverse=True
            )

            st.dataframe(
                pd.DataFrame(mistakes[:20]),
                width="stretch",
                hide_index=True
            )

        with st.expander(
            "Xem toàn bộ Confusion Matrix"
        ):
            st.dataframe(
                cm,
                width="stretch"
            )

    report_path = (
        report_dir / "classification_report.txt"
    )

    if report_path.is_file():
        with st.expander(
            "Xem Precision / Recall / F1 "
            "của toàn bộ 97 lớp"
        ):
            st.code(
                report_path.read_text(
                    encoding="utf-8"
                ),
                language="text"
            )


# ======================================================
# 7. CAM NANG TRA CUU
# ======================================================

def knowledge_page(classes):
    st.header("📖 Cẩm nang bệnh cây trồng")

    crop = st.selectbox(
        "Chọn cây",
        ["all"] + list(CROPS.keys()),
        format_func=lambda x: (
            "Tất cả cây trồng"
            if x == "all" else CROPS[x]
        )
    )

    available = [
        name for name in classes
        if crop == "all" or crop_key(name) == crop
    ]

    if not available:
        st.warning("Không tìm thấy lớp phù hợp.")
        return

    selected = st.selectbox(
        "Chọn bệnh hoặc tình trạng",
        available,
        format_func=class_display
    )

    if selected in DISEASE_DATABASE:
        st.success("Đã có hồ sơ bệnh học.")
    else:
        st.info(
            "Có trong mô hình AI nhưng "
            "chưa có hồ sơ bệnh học đầy đủ."
        )

    show_disease_details(selected)


# ======================================================
# 8. NHOM BENH TUONG TU
# ======================================================

def similarity_page(report_dir):
    st.header("⚖️ Phân tích bệnh dễ nhầm lẫn")
    show_global_confusions(
        report_dir=report_dir,
        display_name=class_display,
        max_rows=20
    )

    st.divider()
    st.subheader("📖 Phân biệt triệu chứng tham khảo")
    st.info(
        "Trang này trình bày thông tin phân biệt "
        "triệu chứng từ cơ sở tri thức cũ. "
        "Để xem những lỗi AI thực sự mắc phải, "
        "hãy mở trang Báo cáo đánh giá."
    )

    for group in SIMILARITY_GROUPS:
        with st.expander(
            group.get("group_name", "Nhóm bệnh")
        ):
            st.write(
                "**Các lớp:**",
                ", ".join(group.get("classes", []))
            )

            st.write(
                "**Nguyên nhân dễ nhầm:**",
                group.get("confusion_reason", "")
            )

            st.write(
                "**Cách phân biệt:**",
                group.get("differentiation_guide", "")
            )


# ======================================================
# 9. MAIN
# ======================================================

st.title("🌱 HỆ THỐNG NHẬN DIỆN BỆNH LÁ CÂY AI")

st.caption(
    "MobileNetV2 · 18 loại cây · 97 lớp "
    "· Explainable AI (Grad-CAM)"
)

checkpoints = find_models()

if not checkpoints:
    st.error(
        "Không tìm thấy checkpoint 97 lớp "
        "trong thư mục training_runs."
    )
    st.stop()

selected_checkpoint = st.sidebar.selectbox(
    "🧠 Chọn mô hình 97 lớp",
    checkpoints,
    format_func=lambda p: p.parent.name
)

try:
    model, class_names = load_model(
        str(selected_checkpoint),
        selected_checkpoint.stat().st_mtime_ns
    )
except Exception as exc:
    st.error(f"Không thể tải mô hình: {exc}")
    st.stop()

st.sidebar.success(
    f"Đã tải mô hình {len(class_names)} lớp"
)

st.sidebar.write("**18 loại cây trồng**")

page = st.sidebar.radio(
    "📌 Chức năng",
    [
        "🌿 Chẩn đoán bằng AI",
        "📊 Báo cáo đánh giá",
        "📖 Cẩm nang bệnh",
        "⚖️ Bệnh dễ nhầm lẫn"
    ]
)

st.sidebar.caption(
    "Mô hình chạy suy luận trên CPU "
    "để hạn chế chiếm VRAM khi demo."
)

if page == "🌿 Chẩn đoán bằng AI":
    diagnose(
        model,
        class_names,
        selected_checkpoint.parent
    )

elif page == "📊 Báo cáo đánh giá":
    show_report(selected_checkpoint.parent)

elif page == "📖 Cẩm nang bệnh":
    knowledge_page(class_names)

elif page == "⚖️ Bệnh dễ nhầm lẫn":
    similarity_page(
        selected_checkpoint.parent
    )
