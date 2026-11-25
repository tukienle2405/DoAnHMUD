
# -*- coding: utf-8 -*-
"""
So sanh anh benh voi anh mau trong tap Train.

- Chi lay anh Train
- Chi lay anh cung nhan du doan
- Uu tien nhom anh khac nhau
- Tranh anh goc dang demo neu la anh Train
- Khong thay doi du lieu hay mo hinh
"""

import os
import csv
import random
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps


def normalize_path(path):
    return os.path.normcase(
        os.path.normpath(str(path))
    )


@st.cache_data(show_spinner=False)
def get_reference_candidates(
    manifest_path,
    manifest_mtime,
    class_name,
    excluded_path=""
):
    """
    Doc manifest va tim cac nhom anh Train
    thuoc dung lop benh can so sanh.
    """

    groups = {}
    excluded_group = None

    excluded = (
        normalize_path(excluded_path)
        if excluded_path else None
    )

    with open(
        manifest_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row.get("split") != "train":
                continue

            if row.get("class") != class_name:
                continue

            path = row.get("path", "").strip()

            if not path:
                continue

            # Nhom Tea duoc tao theo anh goc.
            if row.get("source") == "TEA_GROUPED":
                group_id = row.get("group_id", "")
            else:
                group_id = (
                    row.get("near_group_id", "")
                    or row.get("group_id", "")
                )

            # Neu thieu thong tin nhom,
            # su dung duong dan rieng cua anh.
            if not group_id:
                group_id = normalize_path(path)

            if excluded and normalize_path(path) == excluded:
                excluded_group = group_id

            if group_id not in groups:
                groups[group_id] = path

    candidates = [
        (group_id, path)
        for group_id, path in groups.items()
        if group_id != excluded_group
    ]

    return candidates


def show_reference_gallery(
    class_name,
    manifest_path,
    exclude_path=None,
    max_images=3
):
    """
    Hien thi toi da 3 anh tham khao
    cua cung lop benh.
    """

    st.subheader(
        "🖼️ So sánh với ảnh bệnh tham khảo"
    )

    st.caption(
        "Các ảnh được lấy từ tập Train và có "
        "cùng nhãn với kết quả AI dự đoán. "
        "Đây là ảnh tham khảo, không phải "
        "bằng chứng xác nhận bệnh."
    )

    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        st.info(
            "Không tìm thấy manifest để "
            "hiển thị ảnh tham khảo."
        )
        return

    try:
        candidates = get_reference_candidates(
            str(manifest_path),
            manifest_path.stat().st_mtime_ns,
            class_name,
            str(exclude_path) if exclude_path else ""
        )

    except Exception as exc:
        st.warning(
            f"Không đọc được danh sách ảnh mẫu: {exc}"
        )
        return

    # Tron theo seed co dinh de moi lan mo
    # cung mot benh thi danh sach on dinh.
    seed = 42 + sum(ord(c) for c in class_name)
    random.Random(seed).shuffle(candidates)

    images = []

    for group_id, path in candidates:
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                rgb = img.convert("RGB")
                rgb.thumbnail((800, 800))
                images.append(rgb.copy())

        except (OSError, ValueError):
            continue

        if len(images) >= max_images:
            break

    if not images:
        st.info(
            "Chưa tìm được ảnh Train hợp lệ "
            "cho lớp bệnh này."
        )
        return

    columns = st.columns(len(images))

    for index, (column, image) in enumerate(
        zip(columns, images), start=1
    ):
        with column:
            st.image(
                image,
                width="stretch"
            )
            st.caption(
                f"Ảnh tham khảo {index} · Tập Train"
            )

    if len(images) < max_images:
        st.caption(
            f"Chỉ tìm được {len(images)} ảnh "
            "tham khảo phù hợp."
        )

    st.info(
        "Hãy đối chiếu màu sắc, hình dạng và "
        "phân bố vết bệnh. Các nhóm ảnh trong "
        "manifest giúp hạn chế ảnh gần trùng, "
        "nhưng chưa đảm bảo mọi ảnh là "
        "những lá hoàn toàn khác nhau."
    )
