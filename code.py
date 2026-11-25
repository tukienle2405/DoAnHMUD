# -*- coding: utf-8 -*-
"""
code.py
Điểm vào chính (Main Entrypoint) của Đồ án Học máy và Ứng dụng:
Nhận diện bệnh lá cây trồng & Tư vấn phác đồ điều trị thông minh
"""

import os
import sys
import subprocess

# Đảm bảo đường dẫn cache không bị lỗi font Unicode
os.environ["TORCH_HOME"] = "D:/DoanHMvUD/.cache/torch"
os.environ["MPLCONFIGDIR"] = "D:/DoanHMvUD/.cache/matplotlib"
sys.stdout.reconfigure(encoding='utf-8')

def print_banner():
    print("=" * 70)
    print("🌱 ĐỒ ÁN MÔN HỌC MÁY VÀ ỨNG DỤNG")
    print("   HỆ THỐNG NHẬN DIỆN BỆNH LÁ CÂY TRỒNG & PHÁC ĐỒ ĐIỀU TRỊ THÔNG MINH")
    print("   Mô hình: MobileNetV2 (Transfer Learning) + Explainable AI (Grad-CAM)")
    print("=" * 70)

def main():
    print_banner()
    print("\nChọn chức năng bạn muốn thực hiện:")
    print("  [1] Khởi chạy Giao diện Web Ứng dụng (Streamlit App)")
    print("  [2] Huấn luyện lại mô hình (train_models.py)")
    print("  [3] Xem cẩm nang 15 bệnh và cách điều trị, phục hồi cây")
    print("  [4] Xem bảng phân tích các bệnh dễ gây nhầm lẫn")
    print("  [0] Thoát")
    
    choice = input("\nNhập lựa chọn của bạn (1-4, mặc định là 1): ").strip()
    if not choice:
        choice = "1"

    if choice == "1":
        print("\n[*] Đang khởi chạy ứng dụng Web Streamlit...")
        print("[*] Trình duyệt sẽ mở tại: http://localhost:8501")
        subprocess.run(["streamlit", "run", "app_streamlit.py"])
    elif choice == "2":
        print("\n[*] Đang bắt đầu huấn luyện mô hình MobileNetV2...")
        subprocess.run([sys.executable, "-u", "train_models.py", "--epochs", "5", "--samples", "100"])
    elif choice == "3":
        from disease_knowledge import DISEASE_DATABASE
        print(f"\n[*] CÓ TỔNG CỘNG {len(DISEASE_DATABASE)} LỚP TRONG CƠ SỞ TRI THỨC:")
        for idx, (k, v) in enumerate(DISEASE_DATABASE.items(), 1):
            tag = "[ĐÃ TRAIN AI]" if v.get("in_dataset", False) else "[MỞ RỘNG NGOÀI DATA]"
            print(f"\n[{idx:02d}] {tag} {v['crop']} - {v['vietnamese_name']}")
            print(f"     + Tác nhân: {v['pathogen_type']} ({v['scientific_name']})")
            print(f"     + Mức độ nguy hại: {v['danger_level']}")
            print(f"     + Triệu chứng đầu: {v['symptoms'][0]}")
            print(f"     + Biện pháp hóa học tiêu biểu: {v['treatment_protocol']['chemical_control'][0]}")
    elif choice == "4":
        from disease_knowledge import SIMILARITY_GROUPS
        print("\n[*] BẢNG PHÂN TÍCH CÁC NHÓM BỆNH DỄ GÂY NHẦM LẪN:")
        for grp in SIMILARITY_GROUPS:
            print("\n" + "-" * 60)
            print(f"📌 {grp['group_name']}")
            print(f"⚠️ Lý do nhầm lẫn: {grp['confusion_reason']}")
            print(f"💡 Cách phân biệt:\n{grp['differentiation_guide']}")
    else:
        print("\nTạm biệt!")

if __name__ == "__main__":
    main()
