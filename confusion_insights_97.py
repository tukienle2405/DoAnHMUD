
# -*- coding: utf-8 -*-
"""
Phan tich benh de nham lan tu Confusion Matrix 97 lop.

Doc truc tiep confusion_matrix.csv cua tap Test.

Hang = nhan thuc te
Cot  = nhan du doan

Khong thay doi trong so AI hoac du lieu Test.
"""

from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_confusion_matrix(path_text, modified_time):
    """Doc va kiem tra ma tran nham lan."""

    cm = pd.read_csv(
        path_text,
        index_col=0,
        encoding="utf-8-sig"
    )

    cm.index = cm.index.astype(str).str.strip()
    cm.columns = cm.columns.astype(str).str.strip()

    if cm.empty:
        raise ValueError("Confusion Matrix rong.")

    if cm.index.has_duplicates or cm.columns.has_duplicates:
        raise ValueError("Co nhan bi trung trong ma tran.")

    if set(cm.index) != set(cm.columns):
        raise ValueError(
            "Nhãn hàng và cột của Confusion Matrix không khớp."
        )

    cm = cm.apply(pd.to_numeric, errors="raise")

    if cm.isna().any().any():
        raise ValueError("Ma tran co gia tri rong.")

    if (cm < 0).any().any():
        raise ValueError("Ma tran co so am.")

    if ((cm % 1) != 0).any().any():
        raise ValueError("Ma tran co so anh khong nguyen.")

    return cm.astype(int)


def get_matrix(report_dir):
    """Lay ma tran tu thu muc checkpoint duoc chon."""

    path = Path(report_dir) / "confusion_matrix.csv"

    if not path.is_file():
        return None

    return load_confusion_matrix(
        str(path),
        path.stat().st_mtime_ns
    )


def get_class_confusions(cm, class_name):
    """
    Hai huong nham lan:

    outgoing:
      Nhan that la class_name
      nhung AI doan thanh nhan khac.

    incoming:
      Nhan that la benh khac
      nhung AI doan thanh class_name.
    """

    if class_name not in cm.index:
        return None

    row = cm.loc[class_name]
    column = cm[class_name]

    support = int(row.sum())
    predicted_total = int(column.sum())
    correct = int(cm.loc[class_name, class_name])

    outgoing = []

    for predicted, count in row.items():
        count = int(count)

        if predicted == class_name or count == 0:
            continue

        outgoing.append({
            "class_name": predicted,
            "count": count,
            "actual_support": support
        })

    incoming = []

    for actual, count in column.items():
        count = int(count)

        if actual == class_name or count == 0:
            continue

        incoming.append({
            "class_name": actual,
            "count": count,
            "actual_support": int(cm.loc[actual].sum())
        })

    outgoing.sort(
        key=lambda x: x["count"],
        reverse=True
    )

    incoming.sort(
        key=lambda x: x["count"],
        reverse=True
    )

    return {
        "support": support,
        "correct": correct,
        "false_negative": support - correct,
        "false_positive": predicted_total - correct,
        "recall": (
            correct / support if support else 0
        ),
        "precision": (
            correct / predicted_total
            if predicted_total else 0
        ),
        "outgoing": outgoing,
        "incoming": incoming
    }


def display_confusion_rows(
    items,
    display_name,
    direction,
    max_pairs=5
):
    """Hien thi bang nham lan theo tung huong."""

    if not items:
        st.success(
            "Không ghi nhận nhầm lẫn thuộc nhóm này "
            "trong tập Test."
        )
        return

    rows = []

    for item in items[:max_pairs]:
        support = item["actual_support"]
        count = item["count"]

        rows.append({
            "Bệnh liên quan": display_name(
                item["class_name"]
            ),
            "Số ảnh nhầm": count,
            "Tỷ lệ trong lớp thực tế": (
                f"{count / support * 100:.2f}%"
                if support else "N/A"
            )
        })

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True
    )

    if direction == "outgoing":
        st.caption(
            "Bệnh thật là lớp đang xem, nhưng AI "
            "dự đoán thành bệnh khác. Tỷ lệ tính "
            "trên số ảnh Test của lớp đang xem."
        )
    else:
        st.caption(
            "Bệnh thật là lớp khác, nhưng AI "
            "dự đoán thành lớp đang xem. Tỷ lệ "
            "tính trên số ảnh Test của từng lớp thật."
        )


def show_confusion_insights(
    class_name,
    report_dir,
    display_name,
    max_pairs=5
):
    """
    Goi ngay sau khi AI du doan anh.
    """

    st.subheader(
        "⚖️ AI thường nhầm bệnh này với bệnh nào?"
    )

    try:
        cm = get_matrix(report_dir)
    except Exception as exc:
        st.warning(
            f"Không đọc được Confusion Matrix: {exc}"
        )
        return

    if cm is None:
        st.info(
            "Chưa tìm thấy confusion_matrix.csv "
            "trong thư mục checkpoint."
        )
        return

    result = get_class_confusions(cm, class_name)

    if result is None:
        st.info(
            "Nhãn hiện tại không xuất hiện "
            "trong báo cáo Test."
        )
        return

    st.write(
        "**Nhãn đang phân tích:**",
        display_name(class_name)
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Ảnh Test của lớp",
        result["support"]
    )

    c2.metric(
        "Precision trên Test",
        f"{result['precision'] * 100:.2f}%"
    )

    c3.metric(
        "Recall trên Test",
        f"{result['recall'] * 100:.2f}%"
    )

    with st.expander(
        "🔎 Xem chi tiết các trường hợp nhầm lẫn",
        expanded=True
    ):

        tab1, tab2 = st.tabs([
            "Bệnh này bị nhầm thành bệnh khác",
            "Bệnh khác bị nhầm thành bệnh này"
        ])

        with tab1:
            st.write(
                "**Số ảnh của lớp này bị dự đoán sai:**",
                result["false_negative"]
            )

            display_confusion_rows(
                result["outgoing"],
                display_name,
                "outgoing",
                max_pairs
            )

        with tab2:
            st.write(
                "**Số ảnh bệnh khác bị nhận nhầm "
                "thành lớp này:**",
                result["false_positive"]
            )

            display_confusion_rows(
                result["incoming"],
                display_name,
                "incoming",
                max_pairs
            )

    st.caption(
        "Thống kê từ tập Test nội bộ, không "
        "phải xác suất mắc bệnh. Một số nhãn "
        "và nhóm ảnh vẫn cần xác minh bổ sung."
    )


def show_global_confusions(
    report_dir,
    display_name,
    max_rows=20
):
    """
    Hien thi cac cap nham lan nhieu nhat
    tren toan bo 97 lop.
    """

    st.subheader(
        "📊 Những cặp bệnh AI nhầm nhiều nhất"
    )

    try:
        cm = get_matrix(report_dir)
    except Exception as exc:
        st.warning(
            f"Không đọc được Confusion Matrix: {exc}"
        )
        return

    if cm is None:
        st.warning(
            "Chưa tìm thấy confusion_matrix.csv."
        )
        return

    rows = []
    total_test = int(cm.to_numpy().sum())
    total_correct = sum(
        int(cm.loc[name, name])
        for name in cm.index
    )

    total_wrong = total_test - total_correct

    c1, c2 = st.columns(2)

    c1.metric("Tổng ảnh Test", total_test)
    c2.metric("Số ảnh dự đoán sai", total_wrong)

    for actual in cm.index:
        support = int(cm.loc[actual].sum())

        for predicted in cm.columns:
            if actual == predicted:
                continue

            count = int(cm.loc[actual, predicted])

            if count <= 0:
                continue

            rows.append({
                "Nhãn thực tế": display_name(actual),
                "AI dự đoán thành": display_name(predicted),
                "Số ảnh nhầm": count,
                "Tỷ lệ trong lớp thực tế": (
                    f"{count / support * 100:.2f}%"
                    if support else "N/A"
                )
            })

    rows.sort(
        key=lambda x: x["Số ảnh nhầm"],
        reverse=True
    )

    if rows:
        st.dataframe(
            pd.DataFrame(rows[:max_rows]),
            width="stretch",
            hide_index=True
        )
    else:
        st.success(
            "Không tìm thấy trường hợp dự đoán sai."
        )

    st.caption(
        "Các cặp có số ảnh nhầm lớn không nhất "
        "thiết có tỷ lệ lỗi cao nhất. Cần xét cả "
        "số ảnh Test của từng lớp."
    )
