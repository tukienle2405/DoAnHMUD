# -*- coding: utf-8 -*-
"""
disease_knowledge.py
Cơ sở tri thức bệnh học thực vật toàn diện (Comprehensive Plant Disease Knowledge Base)
Dành cho Đồ án Học máy và Ứng dụng: Cà chua, Ớt chuông và Khoai tây.
Bao gồm:
1. 15 lớp bệnh gốc đã được huấn luyện trong bộ dữ liệu PlantVillage (in_dataset: True)
2. Toàn bộ các bệnh phổ biến khác trên lá của Cà chua, Ớt chuông, Khoai tây chưa có trong dữ liệu huấn luyện (in_dataset: False)
   (Thán thư, Phấn trắng, Héo xanh vi khuẩn, Héo vàng Fusarium, Đốm mắt ếch Cercospora, Đốm xám Stemphylium, Mốc xám Botrytis, Virus Y...)
3. Phác đồ điều trị 3 bước (Canh tác, Sinh học, Hóa học / Thuốc BVTV)
4. Hướng dẫn phục hồi & dưỡng cây sau bệnh
5. Phân tích điểm tương đồng & Cách phân biệt các bệnh dễ gây nhầm lẫn
"""

DISEASE_DATABASE = {
    # =========================================================================
    # PHẦN 1: 15 LỚP BỆNH TRONG TẬP DỮ LIỆU HUẤN LUYỆN (in_dataset: True)
    # =========================================================================
    "Pepper__bell___Bacterial_spot": {
        "crop": "Ớt chuông (Bell Pepper)",
        "in_dataset": True,
        "vietnamese_name": "Bệnh đốm vi khuẩn trên ớt chuông",
        "scientific_name": "Xanthomonas campestris pv. vesicatoria",
        "pathogen_type": "Vi khuẩn (Bacteria)",
        "danger_level": "Cao - Gây rụng lá hàng loạt và làm hư hại chất lượng quả",
        "symptoms": [
            "Xuất hiện các đốm nhỏ (1-3 mm), dạng ngậm nước (water-soaked) ở mặt dưới lá non.",
            "Sau đó đốm chuyển sang màu nâu sẫm hoặc đen, hình tròn hoặc góc cạnh theo gân lá.",
            "Xung quanh vết bệnh thường có quầng viền màu vàng hẹp.",
            "Khi bị nặng, các đốm liên kết lại làm lá cháy khô từng mảng lớn, vàng úa và rụng sớm.",
            "Trên quả non có thể thấy các nốt mụn cóc màu nâu sần sùi."
        ],
        "treatment_protocol": {
            "farming_practices": [
                "Cắt tỉa và tiêu hủy ngay các lá và cành bị bệnh (đem đốt hoặc chôn sâu, không vứt tại mương nước).",
                "Tuyệt đối tránh tưới nước dạng phun sương/phun mưa lên tán lá vào buổi chiều tối; chuyển sang tưới gốc/nhỏ giọt.",
                "Tăng cường thông gió trong nhà lưới/vườn ươm để giảm độ ẩm bề mặt lá.",
                "Khử trùng kéo cắt cành bằng cồn 70 độ hoặc nước Javel pha loãng khi chuyển giữa các cây."
            ],
            "biological_control": [
                "Phun chế phẩm sinh học chứa vi khuẩn đối kháng Bacillus subtilis hoặc Streptomyces định kỳ 7-10 ngày/lần.",
                "Tưới gốc nấm đối kháng Trichoderma spp. để tăng cường sức đề kháng vùng rễ."
            ],
            "chemical_control": [
                "Phun thuốc bảo vệ thực vật gốc Đồng (Copper Hydroxide, Đồng Oxyclorua như Champion, Kocide 53.8 DF).",
                "Kết hợp hoạt chất kháng sinh nông nghiệp: Kasugamycin (Kasumin), Oxytetracycline hoặc Ningnanmycin.",
                "Lưu ý: Phun khi vết bệnh chớm xuất hiện và luân phiên hoạt chất để tránh vi khuẩn kháng thuốc."
            ]
        },
        "post_disease_recovery": [
            "Bổ sung phân bón lá giàu Silic và Canxi (Ca-Bo) để làm dày vách tế bào lá non mới, chống tái nhiễm.",
            "Bón phân hữu cơ hoai mục kết hợp Axit Humic tưới gốc để kích thích cây bung chồi, phục hồi bộ rễ.",
            "Tránh bón quá nhiều phân đạm (N) làm mô cây mềm xốp, dễ bị vi khuẩn xâm nhập trở lại."
        ],
        "confusion_analysis": {
            "similar_diseases": ["Tomato_Bacterial_spot", "Pepper__Anthracnose"],
            "common_features": "Đều tạo ra các đốm màu nâu đen trên lá kèm quầng vàng xung quanh.",
            "key_distinctions": [
                "So với Bệnh thán thư ớt: Đốm vi khuẩn nhỏ hơn (1-3mm), góc cạnh theo gân lá và không bị trũng lõm xuống như Thán thư.",
                "So với Tomato Bacterial Spot: Cùng chi vi khuẩn Xanthomonas nhưng ký chủ chính là ớt, nốt mụn ở mặt dưới rõ hơn."
            ]
        }
    },

    "Pepper__bell___healthy": {
        "crop": "Ớt chuông (Bell Pepper)",
        "in_dataset": True,
        "vietnamese_name": "Ớt chuông khỏe mạnh (Không phát hiện bệnh)",
        "scientific_name": "Capsicum annuum",
        "pathogen_type": "Không có (Cây khỏe)",
        "danger_level": "An toàn",
        "symptoms": [
            "Phiến lá màu xanh đậm đồng đều, phẳng phiu, mặt lá bóng mượt.",
            "Không có vết đốm hoại tử, không quăn queo, không có mạng nhện hay phấn mốc.",
            "Chồi ngọn phát triển mạnh mẽ, cành lá vươn đều."
        ],
        "treatment_protocol": {
            "farming_practices": [
                "Duy trì chế độ tưới nước đều đặn, đảm bảo độ ẩm đất 65-75%, thoát nước tốt.",
                "Làm sạch cỏ dại quanh gốc để cắt đứt nơi cư trú của côn trùng chích hút."
            ],
            "biological_control": [
                "Tưới gốc Trichoderma định kỳ 15-20 ngày/lần để bảo vệ vùng rễ.",
                "Treo bẫy dính màu vàng/xanh theo dõi côn trùng."
            ],
            "chemical_control": ["Không cần phun thuốc hóa học."]
        },
        "post_disease_recovery": ["Duy trì chế độ dinh dưỡng N-P-K theo chu kỳ sinh trưởng."],
        "confusion_analysis": {
            "similar_diseases": [],
            "common_features": "Lá hoàn toàn không có vết bệnh cơ học hay sinh học.",
            "key_distinctions": ["Lá bình thường, diệp lục đồng đều."]
        }
    },

    "Potato___Early_blight": {
        "crop": "Khoai tây (Potato)",
        "in_dataset": True,
        "vietnamese_name": "Bệnh đốm vòng / Cháy lá sớm trên khoai tây",
        "scientific_name": "Alternaria solani",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "Trung bình - Cao (Làm rụng lá sớm, giảm mạnh năng suất củ)",
        "symptoms": [
            "Xuất hiện đầu tiên trên các lá già ở tầng tán dưới gốc.",
            "Vết bệnh hình tròn hoặc bất định màu nâu sẫm đến đen.",
            "ĐẶC ĐIỂM ĐIỂN HÌNH: Có các VÒNG TRÒN ĐỒNG TÂM (concentric rings) giống như hình bia bắn (target board).",
            "Mô lá quanh vết đốm bị vàng dần (quầng vàng do độc tố của nấm Alternaria tiết ra).",
            "Các vết đốm lớn dần (lên đến 1-2 cm), liên kết lại làm toàn bộ lá khô ráp, giòn và rụng."
        ],
        "treatment_protocol": {
            "farming_practices": [
                "Thu gom và dọn sạch tàn dư lá già bị bệnh dưới mặt luống.",
                "Thực hiện luân canh cây trồng khác họ cà ít nhất 2-3 năm.",
                "Vun luống cao, thoát nước tốt sau mưa."
            ],
            "biological_control": ["Phun chế phẩm nấm đối kháng Chaetomium hoặc Chitosan kích kháng."],
            "chemical_control": [
                "Phun thuốc phòng ngừa: Mancozeb (Dithane M-45), Chlorothalonil (Daconil).",
                "Phun trị khi bệnh phát sinh: Difenoconazole (Score 250EC), Azoxystrobin (Amistar)."
            ]
        },
        "post_disease_recovery": [
            "Bổ sung Kali trắng (K2SO4) và Bo để hỗ trợ vận chuyển tinh bột nuôi củ.",
            "Phun amino acid hữu cơ giúp cây giải tỏa stress."
        ],
        "confusion_analysis": {
            "similar_diseases": ["Potato___Late_blight"],
            "common_features": "Đều gây cháy lá màu nâu đen trên khoai tây.",
            "key_distinctions": [
                "So với Late Blight: Early Blight có VÒNG ĐỒNG TÂM RÕ RỆT và khô ráo, phát triển chậm từ gốc lên. Late Blight là vết loang úng nước không có vòng đồng tâm và có mốc trắng mặt dưới."
            ]
        }
    },

    "Potato___healthy": {
        "crop": "Khoai tây (Potato)",
        "in_dataset": True,
        "vietnamese_name": "Khoai tây khỏe mạnh (Không phát hiện bệnh)",
        "scientific_name": "Solanum tuberosum",
        "pathogen_type": "Không có (Cây khỏe)",
        "danger_level": "An toàn",
        "symptoms": ["Tán lá xanh tươi, mép lá nguyên vẹn, cành mập mạp."],
        "treatment_protocol": {
            "farming_practices": ["Vun gốc định kỳ bảo vệ củ con không bị xanh hóa vỏ."],
            "biological_control": ["Bón lót phân vi sinh hữu cơ đối kháng nấm đất."],
            "chemical_control": ["Không cần dùng thuốc hóa học."]
        },
        "post_disease_recovery": ["Chăm sóc bình thường."],
        "confusion_analysis": {
            "similar_diseases": [],
            "common_features": "Lá lành lặn.",
            "key_distinctions": ["Không có tổn thương."]
        }
    },

    "Potato___Late_blight": {
        "crop": "Khoai tây (Potato)",
        "in_dataset": True,
        "vietnamese_name": "Bệnh mốc sương / Cháy lá muộn trên khoai tây",
        "scientific_name": "Phytophthora infestans",
        "pathogen_type": "Oomycete / Nấm thủy mi",
        "danger_level": "CỰC KỲ NGUY HIỂM - Phá hủy cả ruộng khoai trong 5-7 ngày",
        "symptoms": [
            "Vết bệnh ban đầu là các đốm ngậm nước màu xanh xám ở chóp hoặc mép lá.",
            "Vết bệnh nhanh chóng lan rộng thành mảng hoại tử lớn màu nâu đen úng nước.",
            "ĐẶC ĐIỂM: Sáng sớm hoặc khi trời ẩm có LỚP MỐC TRẮNG MỎNG NHƯ BÔNG TUYẾT ở mặt dưới rìa vết bệnh.",
            "Bệnh làm cháy khô toàn bộ thân cành, có mùi thối hăng khó chịu.",
            "Củ bị thối nâu khô từ vỏ ăn sâu vào ruột."
        ],
        "treatment_protocol": {
            "farming_practices": [
                "KHẨN CẤP: Cách ly và tiêu hủy ngay cây nhiễm bệnh nặng.",
                "Ngưng tưới phun mưa; giữ ruộng khô ráo."
            ],
            "biological_control": ["Chỉ dùng phòng ngừa đầu vụ; khi bùng dịch cần dùng thuốc hóa học đặc trị."],
            "chemical_control": [
                "Phun đặc trị nội hấp lưu dẫn mạnh: Metalaxyl + Mancozeb (Ridomil Gold 68WG).",
                "Hoạt chất phối hợp: Dimethomorph, Cymoxanil, Propamocarb (Previcur Energy)."
            ]
        },
        "post_disease_recovery": [
            "Cắt bỏ cành hỏng sau khi vết bệnh khô giòn.",
            "Phun vi lượng Kẽm (Zn) và Magie (Mg) tái tạo diệp lục."
        ],
        "confusion_analysis": {
            "similar_diseases": ["Potato___Early_blight", "Tomato_Late_blight"],
            "common_features": "Lá thâm đen, rụi cả tán.",
            "key_distinctions": [
                "So với Early Blight: Vết bệnh Late Blight dạng bỏng nước sôi, KHÔNG có vòng đồng tâm, có lớp mốc trắng ở mặt dưới khi ẩm."
            ]
        }
    },

    "Tomato_Bacterial_spot": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": True,
        "vietnamese_name": "Bệnh đốm vi khuẩn trên cà chua",
        "scientific_name": "Xanthomonas perforans / Xanthomonas vesicatoria",
        "pathogen_type": "Vi khuẩn (Bacteria)",
        "danger_level": "Cao - Gây rụng hoa, rụng lá và thâm nám quả",
        "symptoms": [
            "Đốm nhỏ 1-3 mm, hình tròn hoặc góc cạnh, viền ngậm nước.",
            "Tâm vết bệnh chuyển màu nâu đen khô giòn.",
            "ĐẶC ĐIỂM: Tâm đốm dễ rách rụng tạo thành các LỖ THỦNG NHỎ NHƯ ĐẠN BẮN (shot-hole).",
            "Mặt trên lá có quầng vàng sáng.",
            "Quả có các nốt mụn cóc nhỏ màu nâu gồ lên."
        ],
        "treatment_protocol": {
            "farming_practices": ["Không tỉa cành khi lá còn ướt sương.", "Khử trùng kéo tỉa bằng cồn."],
            "biological_control": ["Phun vi sinh Bacillus subtilis, Pseudomonas fluorescens."],
            "chemical_control": ["Phun Copper Hydroxide + Mancozeb hoặc Kasugamycin, Oxolinic acid (Starner)."]
        },
        "post_disease_recovery": ["Tưới kích rễ Axit Humic phục hồi tán lá.", "Bón canxi bo tăng độ dai tế bào."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato_Septoria_leaf_spot", "Tomato_Early_blight"],
            "common_features": "Đốm hoại tử màu nâu đen có quầng vàng.",
            "key_distinctions": [
                "So với Septoria: Đốm Septoria có tâm xám trắng với chấm đen li ti (ổ nấm) ở giữa và không bị thủng lỗ như đạn bắn của vi khuẩn."
            ]
        }
    },

    "Tomato_Early_blight": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": True,
        "vietnamese_name": "Bệnh đốm vòng / Cháy lá sớm trên cà chua",
        "scientific_name": "Alternaria solani",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "Trung bình - Phổ biến nhất trên cây cà chua",
        "symptoms": [
            "Bắt đầu từ lá già phía dưới gốc, vết bệnh ban đầu là đốm nâu đen nhỏ.",
            "Mở rộng nhanh thành đốm lớn có ĐƯỜNG VÂN ĐỒNG TÂM RÕ NÉT (target-like concentric rings).",
            "Quanh vết đốm bị biến vàng lan rộng.",
            "Lá úa vàng khô rủ xuống nhưng vẫn dính trên thân."
        ],
        "treatment_protocol": {
            "farming_practices": ["Vặt bỏ lá già dưới gốc bị bệnh.", "Tưới nhỏ giọt, phủ bạt nilon luống."],
            "biological_control": ["Phun định kỳ Trichoderma harzianum hoặc dịch chiết neem."],
            "chemical_control": ["Phun Difenoconazole (Score 250EC), Azoxystrobin (Amistar Top), Chlorothalonil."]
        },
        "post_disease_recovery": ["Phun vi lượng Bo, Kẽm và Amino acid dạng chelate.", "Bón NPK cân đối 1:1:1."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato__Target_Spot", "Tomato_Late_blight"],
            "common_features": "Đốm tròn nâu đen trên lá già làm vàng lá.",
            "key_distinctions": [
                "So với Target Spot: Early Blight có vân đồng tâm rất sắc nét và quầng vàng rộng; Target Spot đốm rải rác đều tán lá và dễ gây rụng lá nhanh hơn."
            ]
        }
    },

    "Tomato_healthy": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": True,
        "vietnamese_name": "Cà chua khỏe mạnh (Không phát hiện bệnh)",
        "scientific_name": "Solanum lycopersicum",
        "pathogen_type": "Không có (Cây khỏe)",
        "danger_level": "An toàn",
        "symptoms": ["Lá màu xanh đậm tự nhiên, phiến lá dày, mép răng cưa đều, không có đốm nấm."],
        "treatment_protocol": {
            "farming_practices": ["Tỉa chồi nách kịp thời, giữ vườn thông thoáng."],
            "biological_control": ["Duy trì phun phòng sinh học định kỳ."],
            "chemical_control": ["Không sử dụng thuốc hóa học."]
        },
        "post_disease_recovery": ["Chăm sóc bình thường."],
        "confusion_analysis": {
            "similar_diseases": [],
            "common_features": "Lá xanh đều.",
            "key_distinctions": ["Không có tổn thương."]
        }
    },

    "Tomato_Late_blight": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": True,
        "vietnamese_name": "Bệnh mốc sương / Cháy lá muộn trên cà chua",
        "scientific_name": "Phytophthora infestans",
        "pathogen_type": "Oomycete / Nấm thủy mi",
        "danger_level": "CỰC KỲ NGUY HIỂM - Làm hỏng cả vườn cà chua chỉ sau vài ngày",
        "symptoms": [
            "Đốm úng nước nhạt màu trên mép hoặc đầu ngọn lá.",
            "Chuyển nhanh sang màu nâu đậm hoặc đen, lan rộng cả phiến lá.",
            "ĐẶC ĐIỂM: Ẩm ướt sáng sớm MẶT DƯỚI LÁ NỞ LỚP MỐC TRẮNG MỎNG NHƯ BÔNG.",
            "Thân cây xuất hiện đoạn thâm đen tóp khô gãy gục.",
            "Quả bị nâu loang lổ, vỏ cứng sần sùi như bị luộc qua nước sôi."
        ],
        "treatment_protocol": {
            "farming_practices": ["Ngắt bỏ ngay lá, quả bệnh cho vào túi đem tiêu hủy xa vườn.", "Mở thoáng cửa nhà màng."],
            "biological_control": ["Trichoderma phòng ngừa đầu vụ."],
            "chemical_control": ["Phun khẩn cấp: Metalaxyl + Mancozeb (Ridomil Gold), Cymoxanil (Acrobat), Dimethomorph."]
        },
        "post_disease_recovery": ["Bổ sung đạm hữu cơ Amino acid + Kali humate.", "Tăng cường Canxi làm cứng vỏ quả."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato_Early_blight", "Tomato_Leaf_Mold"],
            "common_features": "Cháy lá mảng lớn, làm chết cả cành lá.",
            "key_distinctions": [
                "So với Early Blight: Late Blight có lớp mốc trắng mặt dưới, dạng úng nước lan cực nhanh, không có vân đồng tâm khô ráp."
            ]
        }
    },

    "Tomato_Leaf_Mold": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": True,
        "vietnamese_name": "Bệnh mốc lá cà chua",
        "scientific_name": "Passalora fulva (Cladosporium fulvum)",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "Trung bình - Rất phổ biến trong nhà màng ẩm thấp",
        "symptoms": [
            "Mặt trên lá xuất hiện ĐỐM VÀNG NHẠT HOẶC XANH LƠ NHẠT KHÔNG VIỀN.",
            "ĐẶC ĐIỂM: Mặt dưới đối diện mọc LỚP MỐC MÀU XANH Ô LIU ĐẾN NÂU NHUNG MỊN.",
            "Đốm vàng chuyển sang nâu sẫm, lá cuộn lại và rụng.",
            "Ít tấn công trực tiếp vào quả nhưng làm trơ trụi cành giảm năng suất."
        ],
        "treatment_protocol": {
            "farming_practices": ["Hạ độ ẩm không khí < 85% bằng quạt thông gió.", "Tỉa bớt lá già chân gốc."],
            "biological_control": ["Phun vi sinh đối kháng hoặc dầu khoáng sinh học."],
            "chemical_control": ["Phun Azoxystrobin, Difenoconazole, Chlorothalonil ướt đẫm MẶT DƯỚI CỦA LÁ."]
        },
        "post_disease_recovery": ["Bổ sung Magie (MgSO4) và Sắt (Fe) khôi phục diệp lục."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato_Late_blight", "Tomato_Spider_mites_Two_spotted_spider_mite"],
            "common_features": "Lá úa vàng và mặt dưới có màng bám.",
            "key_distinctions": [
                "So với Late Blight: Leaf Mold mốc màu xanh ô liu/nâu hung (không phải trắng bông tuyết) và mặt trên lá chỉ vàng loang lổ chứ không đen úng nước."
            ]
        }
    },

    "Tomato_Septoria_leaf_spot": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": True,
        "vietnamese_name": "Bệnh đốm lá Septoria trên cà chua",
        "scientific_name": "Septoria lycopersici",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "Cao - Gây trơ trụi cành lá khiến quả bị cháy nắng",
        "symptoms": [
            "Vô số ĐỐM NHỎ LI TI (đường kính 1.5 - 3 mm), hình tròn.",
            "ĐẶC ĐIỂM: VIỀN NGOÀI MÀU NÂU ĐẬM, TÂM VẾT BỆNH MÀU XÁM TRẮNG.",
            "TRONG TÂM MÀU XÁM CÓ CÁC CHẤM ĐEN NHỎ LI TI (ổ bào tử pycnidia).",
            "Mỗi lá có hàng chục đến hàng trăm đốm, lá vàng nhanh và rụng sớm."
        ],
        "treatment_protocol": {
            "farming_practices": ["Lót bạt nông nghiệp phủ luống.", "Cắt bỏ lá gốc có chớm đốm li ti."],
            "biological_control": ["Phun nấm đối kháng phòng định kỳ."],
            "chemical_control": ["Phun Chlorothalonil (Daconil 75WP), Mancozeb, Azoxystrobin + Difenoconazole."]
        },
        "post_disease_recovery": ["Che lưới đen giảm nắng tránh quả bị nám.", "Tưới đạm hữu cơ lỏng bật chồi đỉnh."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato_Bacterial_spot", "Tomato_Early_blight"],
            "common_features": "Đốm tròn nâu nhỏ trên lá.",
            "key_distinctions": [
                "So với Bacterial spot: Đốm Septoria có tâm xám trắng với chấm đen li ti ở giữa, không bị rách thủng thành lỗ đạn bắn."
            ]
        }
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": True,
        "vietnamese_name": "Nhện đỏ hai chấm hại cà chua",
        "scientific_name": "Tetranychus urticae",
        "pathogen_type": "Côn trùng / Nhện hại (Arachnid pest)",
        "danger_level": "Cao - Sinh sôi cực nhanh trong mùa khô nóng",
        "symptoms": [
            "Mặt trên lá có VẾT CHÂM KIM LI TI MÀU VÀNG HOẶC TRẮNG BẠC.",
            "Lá mất diệp lục loang lổ, khô ráp giòn như tờ giấy ráp.",
            "ĐẶC ĐIỂM: Ở MẶT DƯỚI LÁ VÀ KẼ NÁCH XUẤT HIỆN MẠNG TƠ MỎNG MỊN.",
            "Mặt dưới có các chấm li ti màu vàng/đỏ cam chuyển động.",
            "Ngọn non bị chùn, cây còi cọc."
        ],
        "treatment_protocol": {
            "farming_practices": ["Tăng độ ẩm bằng tưới phun sương tăng áp (nhện rất sợ nước).", "Cắt tỉa lá mạng tơ đem đốt."],
            "biological_control": ["Thả nhện bắt mồi thiên địch; phun dầu khoáng SK Enspray 99EC, dầu neem."],
            "chemical_control": ["Phun Abamectin, Spiromesifen, Fenpyroximate vào MẶT DƯỚI CỦA LÁ."]
        },
        "post_disease_recovery": ["Phun phân bón lá chứa Kẽm, Sắt và Amino Acid phục hồi diệp lục."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato_Leaf_Mold", "Tomato__Tomato_YellowLeaf__Curl_Virus"],
            "common_features": "Lá vàng loang lổ, biến dạng và khô giòn.",
            "key_distinctions": [
                "So với Leaf Mold: Nhện đỏ có vết châm kim bạc li ti và mạng tơ nhện ở mặt dưới, không phải lớp mốc nỉ nhung."
            ]
        }
    },

    "Tomato__Target_Spot": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": True,
        "vietnamese_name": "Bệnh đốm mắt cua / Đốm đích trên cà chua",
        "scientific_name": "Corynespora cassiicola",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "Cao - Gây rụng lá hàng loạt từ thân giữa lên ngọn",
        "symptoms": [
            "Đốm tròn màu nâu nhạt đến nâu sẫm.",
            "ĐẶC ĐIỂM: Có các VÒNG ĐỒNG TÂM NHƯ TÂM BIA nhưng vòng tròn mờ hơn Early Blight.",
            "Tâm vết bệnh mỏng giòn, RẤT DỄ NỨT NẺ HOẶC RỤNG RA KHI KHÔ.",
            "Xuất hiện đồng thời cả trên lá già lẫn lá bánh tẻ tầng giữa."
        ],
        "treatment_protocol": {
            "farming_practices": ["Tạo khoảng cách thông thoáng đón nắng gió.", "Không để tán lá chạm đất ẩm."],
            "biological_control": ["Phun nấm đối kháng Chaetomium định kỳ."],
            "chemical_control": ["Phun Pyraclostrobin, Boscalid, Azoxystrobin, Chlorothalonil."]
        },
        "post_disease_recovery": ["Bổ sung Canxi Nitrat và Bo tăng độ dai mô lá."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato_Early_blight", "Tomato__Anthracnose"],
            "common_features": "Đốm tròn nâu sẫm trên lá với quầng vàng.",
            "key_distinctions": [
                "So với Early Blight: Target Spot tâm màu nâu nhạt hơn, dễ nứt rách tâm hơn, phân bố cả ở tầng lá giữa."
            ]
        }
    },

    "Tomato__Tomato_mosaic_virus": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": True,
        "vietnamese_name": "Bệnh virus khảm cà chua (ToMV)",
        "scientific_name": "Tomato mosaic virus (ToMV)",
        "pathogen_type": "Virus (Siêu vi trùng)",
        "danger_level": "RẤT CAO - Lây lan cơ học cực mạnh, không có thuốc trị trực tiếp",
        "symptoms": [
            "Lá có hoa văn LOANG LỔ KHẢM HOA: Mảng xanh đậm xen lẫn mảng vàng hoặc xanh nhạt.",
            "Phiến lá biến dạng gồ ghề, nhăn nheo, mép lá co rút hoặc biến thành hình sợi dải hẹp (lá liễu).",
            "Cây phát triển còi cọc, quả chín loang lổ có vệt hoại tử nâu trong ruột."
        ],
        "treatment_protocol": {
            "farming_practices": [
                "NHỔ BỎ VÀ TIÊU HỦY NGAY LẬP TỨC: Cây đã nhiễm virus khảm KHÔNG THỂ CHỮA KHỎI.",
                "Khử trùng tay/dụng cụ bằng sữa tươi không đường 20% hoặc xà phòng TSP trước khi tỉa cây.",
                "Người hút thuốc lá không được chạm vào cây (virus khảm thuốc lá lây truyền qua tay)."
            ],
            "biological_control": ["Phun Chitosan hoặc Oligosaccharide kích kháng miễn dịch tự nhiên (SAR)."],
            "chemical_control": ["KHÔNG CÓ THUỐC HÓA HỌC NÀO DIỆT ĐƯỢC VIRUS ĐANG Ở TRONG CÂY."]
        },
        "post_disease_recovery": ["Bón tăng cường Kali và Lân cho các cây còn lại trong vườn để tăng độ cứng cáp."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato__Tomato_YellowLeaf__Curl_Virus"],
            "common_features": "Lá biến dạng và cây còi cọc do virus.",
            "key_distinctions": [
                "So với TYLCV: Mosaic Virus tạo hoa văn khảm loang lổ xanh đậm - xanh nhạt; TYLCV làm lá non cong hình chén và vàng rực toàn phần."
            ]
        }
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": True,
        "vietnamese_name": "Bệnh virus xoăn vàng lá cà chua (TYLCV)",
        "scientific_name": "Tomato yellow leaf curl virus (TYLCV)",
        "pathogen_type": "Virus do Bọ phấn trắng (Bemisia tabaci) truyền",
        "danger_level": "CỰC KỲ NGUY HIỂM - Có thể gây mất trắng 100% vụ mùa nếu nhiễm sớm",
        "symptoms": [
            "Xuất hiện rõ nhất ở CÁC ĐỌT NON VÀ TẦNG LÁ TRÊN CÙNG.",
            "ĐẶC ĐIỂM: Mép lá non CONG UỐN LÊN TRÊN HÌNH CHIẾC THUYỀN HOẶC CHIẾC CHÉN (cupping/curling).",
            "Phiến lá non nhỏ lại, dày và giòn hơn bình thường.",
            "TOÀN BỘ PHIẾN LÁ VÀ VIỀN LÁ CHUYỂN SANG MÀU VÀNG CHANH SÁNG RỰC.",
            "Cây bị lùn tịt, lóng ngọn sít lại thành chùm; rụng hoa không đậu quả."
        ],
        "treatment_protocol": {
            "farming_practices": ["Nhổ bỏ ngay cây bệnh đem đốt.", "Căng lưới chắn côn trùng 50 mesh.", "Treo BẪY DÍNH MÀU VÀNG."],
            "biological_control": ["Phun nấm ký sinh Verticillium lecanii hoặc Beauveria bassiana diệt bọ phấn trắng."],
            "chemical_control": [
                "TIÊU DIỆT VÉC-TƠ TRUYỀN BỆNH (BỌ PHẤN TRẮNG) LÀ YẾU TỐ QUYẾT ĐỊNH:",
                "Phun luân phiên: Thiamethoxam, Pymetrozine, Dinotefuran, Spirotetramat (Movento 150OD)."
            ]
        },
        "post_disease_recovery": ["Bổ sung Canxi, Kẽm, Silic cho cây chưa nhiễm để biểu bì lá dày hơn."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato__Tomato_mosaic_virus"],
            "common_features": "Lá ngọn biến dạng, màu vàng, cây còi cọc.",
            "key_distinctions": [
                "So với Mosaic virus: TYLCV có biểu hiện uốn cong mép lá hình chén ngửa lên và màu vàng chanh sáng rực toàn phiến."
            ]
        }
    },

    # =========================================================================
    # PHẦN 2: CÁC BỆNH BỔ SUNG TRÊN LÁ CHƯA CÓ TRONG TẬP DỮ LIỆU (in_dataset: False)
    # =========================================================================

    # --- ỚT CHUÔNG (BELL PEPPER) MỞ RỘNG ---
    "Pepper__Anthracnose": {
        "crop": "Ớt chuông (Bell Pepper)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh thán thư trên ớt (Thối đuôi ớt / Bệnh nổ mắt cua)",
        "scientific_name": "Colletotrichum capsici / Colletotrichum gloeosporioides",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "RẤT CAO - Bệnh nguy hiểm số 1 trên cây ớt tại Việt Nam",
        "symptoms": [
            "Trên lá: Xuất hiện các vết đốm tròn màu nâu xám, hơi trũng lõm xuống, có các vòng tròn đồng tâm.",
            "Trên thân và quả: Vết bệnh ban đầu là đốm ngậm nước, sau lõm sâu vào trong thịt quả, hình bầu dục hoặc tròn.",
            "ĐẶC ĐIỂM ĐIỂN HÌNH: Trong điều kiện ẩm ướt, TRÊN VẾT BỆNH NỞ CÁC CHẤM NHẦY MÀU HỒNG CAM HOẶC ĐEN xếp thành vòng tròn đồng tâm (chính là ổ bào tử nấm acervuli).",
            "Quả bị thối khô, vỏ teo tóp lại, rụng hàng loạt khi trời mưa nhiều."
        ],
        "treatment_protocol": {
            "farming_practices": [
                "Thu gom và tiêu hủy toàn bộ quả và lá bị thán thư, không vứt bừa bãi tại rãnh luống.",
                "Tạo luống cao, thoát nước cực tốt trong mùa mưa, tránh để nước ngập úng gốc.",
                "Lót màng phủ nông nghiệp để ngăn nấm bào tử từ đất bắn lên tán cây khi mưa rào."
            ],
            "biological_control": [
                "Phun phòng chế phẩm sinh học Trichoderma harzianum hoặc dịch chiết tỏi ớt.",
                "Sử dụng Chitosan liều lượng 0.2% kích kháng vách tế bào."
            ],
            "chemical_control": [
                "Phun thuốc phòng trừ khi chớm thấy vết thán thư:",
                "+ Nhóm tiếp xúc bảo vệ: Mancozeb (Dithane M-45), Propineb (Antracol 70WP), Chlorothalonil.",
                "+ Nhóm đặc trị nội hấp lưu dẫn: Difenoconazole + Azoxystrobin (Amistar Top 325SC), Tebuconazole (Nativo 750WG), Hexaconazole (Anvil 5SC).",
                "+ Phun ướt đều cả lá, hoa và quả; phun lặp lại sau 5-7 ngày trong mùa mưa."
            ]
        },
        "post_disease_recovery": [
            "Bón tăng cường phân Kali và Canxi để làm cứng vỏ quả và dày phiến lá, chống nấm tái xâm nhập.",
            "Tưới phân Axit Humic kích thích bộ rễ mới, giảm bón phân đạm (N) làm mô quả mềm xốp."
        ],
        "confusion_analysis": {
            "similar_diseases": ["Pepper__bell___Bacterial_spot", "Tomato__Target_Spot"],
            "common_features": "Đều gây đốm tròn màu nâu có vân đồng tâm trên lá và quả.",
            "key_distinctions": [
                "So với Bacterial spot: Vết thán thư trũng lõm sâu xuống bề mặt mô và có chất nhầy màu hồng cam/đen khi ẩm; đốm vi khuẩn nhỏ hơn và mặt dưới gồ nốt mụn chứ không lõm.",
                "So với Target Spot: Bệnh thán thư tấn công cực mạnh lên quả ớt tạo vết thối trũng lõm đặc trưng."
            ]
        }
    },

    "Pepper__Powdery_mildew": {
        "crop": "Ớt chuông (Bell Pepper)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh phấn trắng trên ớt",
        "scientific_name": "Leveillula taurica",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "Trung bình - Cao (Làm rụng lá hàng loạt trong mùa khô ấm)",
        "symptoms": [
            "Mặt trên lá xuất hiện các đốm màu vàng sáng loang lổ.",
            "ĐẶC ĐIỂM: MẶT DƯỚI LÁ PHỦ MỘT LỚP PHẤN TRẮNG MỊN NHƯ BỘT MÌ (bào tử nấm conidia).",
            "Lá bị nhiễm bệnh có xu hướng cuộn mép lên trên, khô giòn và rụng từ dưới gốc rụng lên.",
            "Bệnh làm trơ trụi cành khiến quả non bị cháy nắng."
        ],
        "treatment_protocol": {
            "farming_practices": ["Tăng độ ẩm bằng tưới phun nhẹ vào sáng sớm.", "Cắt tỉa lá già chân gốc."],
            "biological_control": ["Phun chế phẩm vi khuẩn Bacillus amyloliquefaciens hoặc dung dịch baking soda 0.5%."],
            "chemical_control": [
                "Phun thuốc đặc trị phấn trắng: Hexaconazole (Anvil), Dinocap, Azoxystrobin, Difenoconazole.",
                "Thuốc lưu huỳnh hữu cơ vi sinh (Sulfur) phun ngừa rất hiệu quả."
            ]
        },
        "post_disease_recovery": ["Phun phân bón lá giàu Magie và Silic giúp phục hồi độ xanh của lá non."],
        "confusion_analysis": {
            "similar_diseases": ["Pepper__bell___healthy", "Tomato_Leaf_Mold"],
            "common_features": "Mặt dưới lá có lớp bột mịn bao phủ.",
            "key_distinctions": [
                "Phấn trắng có lớp bột MÀU TRẮNG TINH như bột mì; trong khi Leaf Mold lớp mốc có màu xanh ô liu hoặc nâu nhung."
            ]
        }
    },

    "Pepper__Bacterial_wilt": {
        "crop": "Ớt chuông (Bell Pepper)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh héo xanh vi khuẩn trên ớt",
        "scientific_name": "Ralstonia solanacearum",
        "pathogen_type": "Vi khuẩn trong đất (Soil-borne Bacteria)",
        "danger_level": "CỰC KỲ NGUY HIỂM - Cây chết rũ khi lá vẫn còn xanh tươi",
        "symptoms": [
            "ĐẶC ĐIỂM ĐIỂN HÌNH: CÂY HÉO RŨ ĐỘT NGỘT BAN NGÀY KHI TRỜI NẮNG, BAN ĐÊM TƯƠI LẠI; sau 2-3 ngày cây chết héo hoàn toàn NHƯNG LÁ VẪN GIỮ MÀU XANH.",
            "Phần gốc sát mặt đất bị thâm nâu, bó mạch dẫn bên trong thân chuyển màu nâu đen sũng nước.",
            "Thử nghiệm dòng vi khuẩn: Cắt một đoạn thân nhúng vào cốc nước thủy tinh trong suốt, sau vài phút sẽ thấy DÒNG MỦ VI KHUẨN MÀU TRẮNG ĐỤC NHƯ KHÓI CHẢY RA TỪ VẾT CẮT."
        ],
        "treatment_protocol": {
            "farming_practices": [
                "NHỔ BỎ NGAY LẬP TỨC cây bị bệnh và rắc vôi bột vào hố cây để khử trùng đất, không tưới nước tràn qua hố bệnh.",
                "Luân canh với cây trồng nước (lúa) hoặc ngô ít nhất 2 vụ.",
                "Ghép cây ớt lên gốc cà tím kháng bệnh héo xanh."
            ],
            "biological_control": ["Tưới nấm đối kháng Trichoderma và xạ khuẩn Streptomyces định kỳ từ khi ươm cây giống."],
            "chemical_control": [
                "Bệnh vi khuẩn mạch dẫn không thể chữa khỏi khi cây đã héo.",
                "Phun/tưới phòng ngừa các cây xung quanh bằng Bismerthiazol, Oxolinic acid (Starner), Copper Hydroxide."
            ]
        },
        "post_disease_recovery": ["Không thể phục hồi cây đã héo xanh, cần chú trọng xử lý đất vụ sau bằng vôi bột 100kg/1000m2."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato_Late_blight"],
            "common_features": "Cây héo rũ nhanh chóng và chết.",
            "key_distinctions": [
                "Héo xanh chết khi LÁ VẪN CÒN XANH NGUYÊN và có dòng mủ vi khuẩn trong thân; Late Blight lá bị thâm đen ủng nước trước khi héo."
            ]
        }
    },

    "Pepper__Cercospora_leaf_spot": {
        "crop": "Ớt chuông (Bell Pepper)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh đốm mắt ếch Cercospora trên ớt",
        "scientific_name": "Cercospora capsici",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "Trung bình - Cao (Gây rụng lá hàng loạt mùa mưa)",
        "symptoms": [
            "Hình thành các đốm tròn lớn (5-10 mm) trên phiến lá.",
            "ĐẶC ĐIỂM: VIỀN NGOÀI MÀU NÂU ĐỎ ĐẬM, TÂM VẾT BỆNH MÀU TRẮNG XÁM HOẶC XÁM NHẠT trông giống như MẮT CON ẾCH (frogeye spot).",
            "Mô lá quanh vết đốm bị biến vàng rõ rệt.",
            "Lá bị nhiều đốm sẽ vàng toàn bộ và rụng rất nhanh, chỉ còn trơ lại cành và quả non."
        ],
        "treatment_protocol": {
            "farming_practices": ["Thu gom dọn sạch lá rụng dưới đất.", "Tưới gốc nhỏ giọt, tránh văng nước lên lá."],
            "biological_control": ["Phun phòng nấm Chaetomium định kỳ."],
            "chemical_control": ["Phun thuốc nhóm Triazole: Difenoconazole, Tebuconazole hoặc Mancozeb, Chlorothalonil."]
        },
        "post_disease_recovery": ["Phun đạm cá hữu cơ và phân bón vi lượng kích thích ra chồi mới."],
        "confusion_analysis": {
            "similar_diseases": ["Pepper__bell___Bacterial_spot", "Tomato_Septoria_leaf_spot"],
            "common_features": "Đều có tâm màu xám sáng và viền màu đậm.",
            "key_distinctions": [
                "Đốm Cercospora to hơn (5-10mm) và tròn trịa hình mắt ếch rõ nét, không bị thủng lỗ như Bacterial spot."
            ]
        }
    },

    # --- CÀ CHUA (TOMATO) MỞ RỘNG ---
    "Tomato__Anthracnose": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh thán thư trên quả và lá cà chua",
        "scientific_name": "Colletotrichum coccodes",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "Cao - Gây thối hỏng quả chín hàng loạt",
        "symptoms": [
            "Trên lá: Tạo các đốm nhỏ màu nâu có quầng vàng hẹp.",
            "Trên quả chín: Xuất hiện các vết đốm tròn trũng lõm xuống (sunken circular lesions).",
            "ĐẶC ĐIỂM: Tâm vết bệnh trũng nước, sau đó xuất hiện VÒNG TRÒN CÁC HẠT NHỎ MÀU ĐEN (microsclerotia) hoặc nhầy hồng cam.",
            "Các vết đốm lớn dần liên kết làm quả thối nhũn và rụng trước khi thu hoạch."
        ],
        "treatment_protocol": {
            "farming_practices": ["Thu hoạch quả đúng độ chín, không để quả chín quá lâu trên cây.", "Lót màng phủ ngăn đất bắn lên quả."],
            "biological_control": ["Phun dịch chiết tỏi ớt hoặc Trichoderma phòng ngừa."],
            "chemical_control": ["Phun Azoxystrobin (Amistar), Difenoconazole, Chlorothalonil (Daconil), Pyraclostrobin."]
        },
        "post_disease_recovery": ["Tăng cường bón Kali và Canxi giúp vỏ quả cà chua dày dặn, hạn chế nứt quả nấm xâm nhập."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato__Target_Spot", "Tomato_Early_blight"],
            "common_features": "Đốm tròn nâu có vân đồng tâm.",
            "key_distinctions": [
                "Thán thư tạo vết trũng lõm sâu đặc trưng trên quả chín; Target Spot và Early Blight chủ yếu gây hại mạnh trên phiến lá."
            ]
        }
    },

    "Tomato__Powdery_mildew": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh phấn trắng trên cà chua",
        "scientific_name": "Leveillula taurica / Oidium neolycopersici",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "Trung bình - Phổ biến trong nhà màng mùa hanh khô",
        "symptoms": [
            "Mặt trên lá xuất hiện các đám bột màu trắng như rắc bột mì.",
            "Các mảng bột trắng lan rộng phủ kín cả mặt trên và mặt dưới lá.",
            "Mô lá dưới lớp bột chuyển sang màu vàng, sau đó khô cháy màu nâu và rụng."
        ],
        "treatment_protocol": {
            "farming_practices": ["Tăng thông gió trong nhà màng, giảm mật độ trồng quá dày."],
            "biological_control": ["Phun dung dịch sữa tươi pha nước tỉ lệ 1:9 hoặc dầu khoáng sinh học."],
            "chemical_control": ["Phun Hexaconazole (Anvil), Azoxystrobin, Difenoconazole, Sulfur vi sinh."]
        },
        "post_disease_recovery": ["Bổ sung Magie và Phân bón lá vi lượng giúp lá xanh quang hợp trở lại."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato_Leaf_Mold"],
            "common_features": "Lá có lớp màng bao phủ.",
            "key_distinctions": [
                "Phấn trắng có màu TRẮNG BỘT ở cả 2 mặt lá; Leaf Mold có màu xanh ô liu/nâu hung mịn ở mặt dưới lá."
            ]
        }
    },

    "Tomato__Bacterial_wilt": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh héo xanh vi khuẩn trên cà chua",
        "scientific_name": "Ralstonia solanacearum",
        "pathogen_type": "Vi khuẩn trong đất (Soil-borne Bacteria)",
        "danger_level": "CỰC KỲ NGUY HIỂM - Cây chết rũ nhanh chóng khi lá vẫn xanh",
        "symptoms": [
            "Lá ngọn héo rũ rụt xuống vào buổi trưa nắng gắt, hồi phục vào ban đêm.",
            "Chỉ sau 2-4 ngày, toàn bộ cây héo rũ hoàn toàn nhưng LÁ VẪN GIỮ MÀU XANH KHÔNG KỊP VÀNG ÚA.",
            "Cắt ngang gốc thân nhúng vào nước trong sẽ thấy DÒNG MỦ TRẮNG ĐỤC TIẾT RA TỪ BÓ MẠCH DẪN."
        ],
        "treatment_protocol": {
            "farming_practices": [
                "Nhổ bỏ ngay cây bệnh cho vào bao nilon đem đốt, rắc vôi bột vào gốc.",
                "Ghép cà chua trên gốc cà tím kháng héo xanh EG203."
            ],
            "biological_control": ["Tưới phòng Trichoderma và Bacillus subtilis từ khi làm đất."],
            "chemical_control": ["Tưới gốc Starner 20WP, Kasumin hoặc Bismerthiazol phòng các cây lân cận."]
        },
        "post_disease_recovery": ["Không thể phục hồi cây đã héo xanh."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato_Late_blight"],
            "common_features": "Cây héo rũ chết nhanh.",
            "key_distinctions": [
                "Héo xanh chết khi lá vẫn xanh mướt; Late Blight lá bị thâm đen úng nước trước khi chết."
            ]
        }
    },

    "Tomato__Fusarium_wilt": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh héo vàng Fusarium trên cà chua",
        "scientific_name": "Fusarium oxysporum f. sp. lycopersici",
        "pathogen_type": "Nấm đất (Soil-borne Fungus)",
        "danger_level": "Cao - Gây vàng lá một bên thân và chết dần",
        "symptoms": [
            "ĐẶC ĐIỂM: VÀNG LÁ BẮT ĐẦU TỪ MỘT BÊN CỦA CÂY HOẶC MỘT BÊN CỦA PHIẾN LÁ (héo một bên).",
            "Bắt đầu từ các lá già dưới gốc rồi tiến dần lên ngọn.",
            "Lá vàng úa héo rủ từ từ chứ không héo nhanh như vi khuẩn.",
            "Chẻ dọc thân cây thấy bó mạch dẫn hóa nâu sẫm kéo dài từ gốc lên thân."
        ],
        "treatment_protocol": {
            "farming_practices": ["Nâng độ pH đất lên 6.5 - 7.0 bằng vôi bột (nấm Fusarium rất thích đất chua pH < 5.5)."],
            "biological_control": ["Tưới nấm đối kháng Trichoderma viride vùng rễ."],
            "chemical_control": ["Tưới gốc thuốc trừ nấm: Hymexazol, Carbendazim, Metalaxyl + Mancozeb."]
        },
        "post_disease_recovery": ["Bón phân lân và kali hữu cơ giúp rễ phục hồi."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato__Bacterial_wilt", "Tomato_Early_blight"],
            "common_features": "Lá vàng úa và héo.",
            "key_distinctions": [
                "Fusarium lá bị VÀNG RỰC MỘT BÊN THÂN TRƯỚC rồi mới héo từ từ; Bacterial wilt lá vẫn xanh mà héo rũ cực nhanh trong 2 ngày."
            ]
        }
    },

    "Tomato__Gray_leaf_spot": {
        "crop": "Cà chua (Tomato)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh đốm xám lá cà chua",
        "scientific_name": "Stemphylium solani / Stemphylium lycopersici",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "Cao - Làm rụng trụi lá già đến lá non",
        "symptoms": [
            "Xuất hiện các đốm nhỏ (1-2 mm) màu nâu sẫm ở cả hai mặt lá.",
            "Vết bệnh mở rộng thành đốm xám bóng, hình tròn hoặc góc cạnh.",
            "ĐẶC ĐIỂM: TÂM ĐỐM MÀU XÁM TRO RẤT MỎNG VÀ DỄ RÁCH RỤNG TẠO LỖ HỔNG.",
            "Lá vàng nhanh và rụng ồ ạt, trơ lại quả cà chua trên giàn."
        ],
        "treatment_protocol": {
            "farming_practices": ["Vệ sinh đồng ruộng, dọn sạch tàn dư cây trồng họ cà."],
            "biological_control": ["Phun phòng nấm đối kháng định kỳ."],
            "chemical_control": ["Phun thuốc: Azoxystrobin, Chlorothalonil, Mancozeb, Difenoconazole."]
        },
        "post_disease_recovery": ["Phun phân bón lá kích rễ và tăng sinh diệp lục."],
        "confusion_analysis": {
            "similar_diseases": ["Tomato_Septoria_leaf_spot", "Tomato_Bacterial_spot"],
            "common_features": "Đốm nhỏ li ti màu xám trên lá.",
            "key_distinctions": [
                "Đốm xám Stemphylium không có các chấm đen li ti (ổ nấm pycnidia) như Septoria."
            ]
        }
    },

    # --- KHOAI TÂY (POTATO) MỞ RỘNG ---
    "Potato__Anthracnose_Black_dot": {
        "crop": "Khoai tây (Potato)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh thán thư / Chấm đen khoai tây (Black Dot)",
        "scientific_name": "Colletotrichum coccodes",
        "pathogen_type": "Nấm (Fungus)",
        "danger_level": "Cao - Gây chết rụi thân ngọn và thối củ",
        "symptoms": [
            "Trên lá: Phiến lá chuyển sang màu vàng xám, héo rũ tương tự như thiếu nước.",
            "ĐẶC ĐIỂM: TRÊN THÂN VÀ RỄ GỐC XUẤT HIỆN VÔ SỐ CHẤM ĐEN NHỎ LI TI NHƯ HẠT TIÊU (microsclerotia).",
            "Vỏ thân cây bị bong tróc, phần lõi thân rỗng và chuyển màu xám bạc.",
            "Trên củ khoai tây có các vết nám màu nâu xám bạc có nhiều chấm đen nhỏ li ti."
        ],
        "treatment_protocol": {
            "farming_practices": ["Luân canh cây trồng ít nhất 3-4 năm.", "Chọn củ giống khỏe không có chấm đen."],
            "biological_control": ["Xử lý đất trước khi trồng bằng nấm đối kháng Trichoderma."],
            "chemical_control": ["Phun Azoxystrobin, Boscalid, Difenoconazole khi cây bắt đầu ra hoa tạo củ."]
        },
        "post_disease_recovery": ["Tăng cường phân Kali giúp củ chắc ruột và vỏ củ nhanh lành lặn."],
        "confusion_analysis": {
            "similar_diseases": ["Potato___Early_blight"],
            "common_features": "Lá vàng úa và khô cháy.",
            "key_distinctions": [
                "Black Dot có các chấm đen li ti hạt tiêu phủ kín gốc thân và rễ; Early Blight có vân đồng tâm trên phiến lá."
            ]
        }
    },

    "Potato__Virus_Y": {
        "crop": "Khoai tây (Potato)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh virus Y trên khoai tây (PVY)",
        "scientific_name": "Potato virus Y (PVY)",
        "pathogen_type": "Virus do Rệp muội (Aphids) truyền",
        "danger_level": "RẤT CAO - Giảm 50-80% năng suất củ khoai tây",
        "symptoms": [
            "Gân lá ở mặt dưới bị hoại tử thâm đen (dấu hiệu gân đen hình xương cá).",
            "Mặt trên lá có hoa văn khảm loang lổ xanh đậm xanh nhạt, phiến lá nhăn nheo, gồ ghề.",
            "Lá già phía dưới bị rụng rũ xuống nhưng vẫn dính trên thân (dấu hiệu lá cụp).",
            "Cây phát triển lùn còi cọc, củ nhỏ và ruột củ có các vòng hoại tử nâu."
        ],
        "treatment_protocol": {
            "farming_practices": ["Sử dụng củ giống sạch bệnh có chứng nhận.", "Nhổ bỏ cây bệnh tiêu hủy ngay."],
            "biological_control": ["Phun dầu khoáng nông nghiệp để ngăn rệp muội chích hút truyền virus."],
            "chemical_control": [
                "Diệt rệp muội môi giới truyền bệnh: Pymetrozine, Imidacloprid, Thiamethoxam, Spirotetramat."
            ]
        },
        "post_disease_recovery": ["Cây đã nhiễm virus PVY không thể chữa khỏi; bảo vệ các cây chưa nhiễm bằng cách tiêu diệt rệp muội."],
        "confusion_analysis": {
            "similar_diseases": ["Potato___Early_blight"],
            "common_features": "Lá có vết thâm đen và rụng lá.",
            "key_distinctions": [
                "PVY hoại tử chạy dọc theo GÂN LÁ hình xương cá và lá nhăn nheo dạng virus; Early Blight là đốm tròn có vòng đồng tâm trên phiến thịt lá."
            ]
        }
    },

    "Potato__Bacterial_wilt": {
        "crop": "Khoai tây (Potato)",
        "in_dataset": False,
        "vietnamese_name": "Bệnh héo xanh vi khuẩn / Thối nâu khoai tây",
        "scientific_name": "Ralstonia solanacearum",
        "pathogen_type": "Vi khuẩn trong đất (Bacteria)",
        "danger_level": "CỰC KỲ NGUY HIỂM - Cây chết rũ khi lá còn xanh, củ thối chảy dịch trắng",
        "symptoms": [
            "Cây héo rũ nhanh chóng khi lá vẫn còn xanh mướt.",
            "Củ khoai tây bị thối nâu vòng mạch dẫn: Cắt ngang củ thấy VÒNG BÓ MẠCH CỦ CHUYỂN MÀU NÂU VÀ CHẢY RA DỊCH MỦ VI KHUẨN MÀU TRẮNG ĐỤC khi bóp nhẹ.",
            "Mắt củ khoai tây bị thâm đen và dính đất cát bết lại."
        ],
        "treatment_protocol": {
            "farming_practices": ["Tuyệt đối không dùng củ từ ruộng bệnh làm giống.", "Luân canh ngô/lúa nước."],
            "biological_control": ["Xử lý đất bằng vôi bột và nấm đối kháng."],
            "chemical_control": ["Tưới phòng thuốc gốc đồng hoặc Starner 20WP quanh vùng rễ cây lân cận."]
        },
        "post_disease_recovery": ["Không thể phục hồi cây bệnh."],
        "confusion_analysis": {
            "similar_diseases": ["Potato___Late_blight"],
            "common_features": "Cây chết và củ bị thối nâu.",
            "key_distinctions": [
                "Bacterial wilt cắt củ bóp nhẹ ra dịch mủ trắng đục của vi khuẩn; Late Blight thối khô từ ngoài vỏ ăn vào và lá có mốc trắng sương mai."
            ]
        }
    }
}

# =========================================================================
# BẢNG TRA CỨU NHÓM BỆNH TƯƠNG ĐỒNG DỄ NHẦM LẪN (Confusion Differentiation)
# =========================================================================
# Nhóm ớt cay riêng biệt: các lớp dưới đây mới ở mức kiến thức, chưa được train.
# Tên khóa phải trùng CHÍNH XÁC tên thư mục ảnh khi bổ sung dữ liệu huấn luyện.
DISEASE_DATABASE.update({
    "Chili___healthy": {
        "crop": "Ớt cay (Chili Pepper)", "in_dataset": False,
        "vietnamese_name": "Lá ớt cay khỏe mạnh", "scientific_name": "Capsicum spp.",
        "pathogen_type": "Không có", "danger_level": "Không phát hiện dấu hiệu bệnh",
        "symptoms": ["Lá có màu xanh tương đối đồng đều, hình dạng tùy giống ớt.",
                     "Không có đốm hoại tử, vệt khảm hay xoăn biến dạng bất thường."],
        "treatment_protocol": {
            "farming_practices": ["Duy trì tưới tiêu phù hợp, tránh úng và quan sát hai mặt lá thường xuyên."],
            "biological_control": ["Theo dõi sâu bệnh theo nguyên tắc quản lý dịch hại tổng hợp."],
            "chemical_control": ["Không cần thuốc bảo vệ thực vật nếu cây khỏe."]},
        "post_disease_recovery": ["Chăm sóc và bón phân cân đối theo giai đoạn sinh trưởng."],
        "confusion_analysis": {"similar_diseases": ["Chili___Mosaic_virus"],
                               "common_features": "Màu lá có thể không đồng đều.",
                               "key_distinctions": ["Khảm lá bất thường cần đánh giá thêm, không thể khẳng định virus chỉ từ ảnh."]}
    },
    "Chili___Bacterial_spot": {
        "crop": "Ớt cay (Chili Pepper)", "in_dataset": False,
        "vietnamese_name": "Bệnh đốm vi khuẩn trên ớt cay", "scientific_name": "Xanthomonas spp.",
        "pathogen_type": "Vi khuẩn", "danger_level": "Có thể gây rụng lá khi nặng",
        "symptoms": ["Đốm nhỏ màu nâu sẫm, đôi khi có quầng vàng.",
                     "Nhiều đốm liên kết có thể làm lá vàng và rụng."],
        "treatment_protocol": {
            "farming_practices": ["Dùng giống sạch bệnh, tránh tưới phun lên lá, vệ sinh dụng cụ.",
                                  "Thu gom tàn dư bệnh và giảm ẩm trên tán lá."],
            "biological_control": ["Ưu tiên quản lý dịch hại tổng hợp; chọn chế phẩm được phép dùng cho ớt."],
            "chemical_control": ["Chỉ dùng thuốc đăng ký phù hợp với cây và bệnh; tuân thủ nhãn, liều và thời gian cách ly."]},
        "post_disease_recovery": ["Giữ cây thông thoáng và theo dõi lá non."],
        "confusion_analysis": {"similar_diseases": ["Chili___Cercospora_leaf_spot"],
                               "common_features": "Đều tạo đốm nâu trên lá.",
                               "key_distinctions": ["Cercospora thường có tâm đốm xám nhạt, viền sẫm; có thể cần xét nghiệm phân biệt."]}
    },
    "Chili___Cercospora_leaf_spot": {
        "crop": "Ớt cay (Chili Pepper)", "in_dataset": False,
        "vietnamese_name": "Đốm mắt ếch Cercospora trên ớt cay", "scientific_name": "Cercospora capsici",
        "pathogen_type": "Nấm", "danger_level": "Có thể gây rụng lá khi nặng",
        "symptoms": ["Đốm tròn có tâm xám/trắng nhạt và viền nâu sẫm.",
                     "Lá nhiễm nặng có thể vàng và rụng."],
        "treatment_protocol": {
            "farming_practices": ["Dọn lá bệnh, luân canh và hạn chế làm ướt lá."],
            "biological_control": ["Áp dụng quản lý dịch hại tổng hợp."],
            "chemical_control": ["Nếu cần thuốc, kiểm tra đăng ký sử dụng và tuân thủ nhãn sản phẩm."]},
        "post_disease_recovery": ["Theo dõi lá non và duy trì dinh dưỡng cân đối."],
        "confusion_analysis": {"similar_diseases": ["Chili___Bacterial_spot"],
                               "common_features": "Đều có đốm hoại tử trên lá.",
                               "key_distinctions": ["Đốm mắt ếch thường có tâm sáng và viền nâu rõ."]}
    },
    "Chili___Mosaic_virus": {
        "crop": "Ớt cay (Chili Pepper)", "in_dataset": False,
        "vietnamese_name": "Triệu chứng khảm lá nghi do virus trên ớt cay",
        "scientific_name": "Nhiều virus có thể gây triệu chứng tương tự",
        "pathogen_type": "Virus (chưa xác định)", "danger_level": "Tùy tác nhân",
        "symptoms": ["Mảng xanh đậm và xanh nhạt loang bất thường, lá có thể nhăn hoặc biến dạng.",
                     "Dễ nhầm với tổn thương do dinh dưỡng, thuốc hoặc côn trùng."],
        "treatment_protocol": {
            "farming_practices": ["Cách ly cây nghi bệnh, vệ sinh dụng cụ, kiểm soát môi giới truyền bệnh.",
                                  "Gửi mẫu xét nghiệm nếu cần xác nhận virus."],
            "biological_control": ["Theo dõi côn trùng môi giới và quản lý tổng hợp."],
            "chemical_control": ["Thuốc trừ nấm và vi khuẩn không chữa khỏi bệnh do virus."]},
        "post_disease_recovery": ["Cây bị nhiễm virus thường không chữa khỏi; đánh giá loại bỏ cây bệnh nếu cần."],
        "confusion_analysis": {"similar_diseases": ["Chili___healthy"],
                               "common_features": "Màu lá thay đổi theo giống và điều kiện.",
                               "key_distinctions": ["Không thể xác định chính xác loại virus chỉ bằng ảnh lá."]}
    }
})

SIMILARITY_GROUPS = [
    {
        "group_name": "Nhóm 1: Bệnh nấm đốm hoại tử lá cà chua (Cháy lá sớm vs Cháy lá muộn vs Đốm Septoria vs Đốm đích vs Thán thư)",
        "classes": [
            "Tomato_Early_blight",
            "Tomato_Late_blight",
            "Tomato_Septoria_leaf_spot",
            "Tomato__Target_Spot",
            "Tomato__Anthracnose"
        ],
        "confusion_reason": "Cả 5 bệnh đều gây ra các vết đốm màu nâu sẫm/đen trên phiến lá, có quầng vàng xung quanh và làm lá khô héo rụng sớm. Khi mô hình AI trích xuất đặc trưng bậc thấp (cạnh, màu sắc), chúng rất dễ bị phạt nhầm lẫn nếu không học được đặc trưng vân (texture) tinh vi.",
        "differentiation_guide": (
            "- Early Blight (Đốm vòng): Đốm to (1-1.5cm), có VÂN ĐỒNG TÂM HÌNH TÂM BIA rõ nét, khô ráo, từ lá già dưới gốc.\n"
            "- Late Blight (Mốc sương): Đốm úng nước loang lổ, KHÔNG có vân đồng tâm, có LỚP MỐC TRẮNG NHƯ BÔNG ở mặt dưới lá khi ẩm, lây lan siêu tốc.\n"
            "- Septoria (Đốm Septoria): Vô số đốm NHỎ LI TI (1-3mm), viền nâu đậm nhưng TÂM MÀU XÁM TRẮNG có CHẤM ĐEN LI TI (ổ bào tử) ở giữa.\n"
            "- Target Spot (Đốm mắt cua): Đốm tròn vừa (4-10mm), vân đồng tâm ít sắc nét hơn Early Blight, TÂM RẤT DỄ NỨT RÁCH THỦNG.\n"
            "- Anthracnose (Thán thư): Vết bệnh hơi TRŨNG LÕM XUỐNG, đặc biệt tấn công mạnh lên quả tạo vết thối tròn lõm với hạt nhầy hồng cam."
        )
    },
    {
        "group_name": "Nhóm 2: Bệnh nấm cháy lá trên khoai tây (Cháy lá sớm vs Cháy lá muộn vs Thán thư Black Dot)",
        "classes": [
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato__Anthracnose_Black_dot"
        ],
        "confusion_reason": "Đều làm cháy lá khoai tây thành màu nâu đen, làm sụt giảm nghiêm trọng năng suất củ.",
        "differentiation_guide": (
            "- Potato Early Blight: Vết bệnh khô, hình tròn có các vân đồng tâm nổi rõ, phát triển chậm từ gốc lên.\n"
            "- Potato Late Blight: Vết bệnh dạng bỏng nước sôi thâm đen, mép loang bất định, có viền mốc trắng sáng sớm mặt dưới, cháy rụi cả giàn chỉ trong vài ngày.\n"
            "- Potato Black Dot (Thán thư chấm đen): Thân và rễ có vô số CHẤM ĐEN NHỎ LI TI NHƯ HẠT TIÊU, vỏ thân bong tróc bạc màu."
        )
    },
    {
        "group_name": "Nhóm 3: Bệnh virus gây biến dạng lá cà chua & khoai tây (Virus khảm ToMV vs Virus xoăn vàng lá TYLCV vs Virus Y)",
        "classes": [
            "Tomato__Tomato_mosaic_virus",
            "Tomato__Tomato_YellowLeaf__Curl_Virus",
            "Potato__Virus_Y"
        ],
        "confusion_reason": "Đều do virus gây ra, làm lá cong queo, nhăn nhúm, cây lùn còi cọc và không chữa được bằng thuốc diệt nấm/vi khuẩn.",
        "differentiation_guide": (
            "- Tomato Mosaic Virus (ToMV): Mặt lá có hoa văn khảm loang lổ (xanh đậm xen xanh vàng), phiến lá hẹp dài như lá liễu.\n"
            "- Tomato Yellow Leaf Curl Virus (TYLCV): Mép lá cuộn ngửa lên tạo HÌNH CHIẾC CHÉN/CHIẾC THUYỀN, màu vàng rực từ ngọn non, cây lùn sít ngọn.\n"
            "- Potato Virus Y (PVY): GÂN LÁ MẶT DƯỚI BỊ HOẠI TỬ ĐEN HÌNH XƯƠNG CÁ, lá già cụp rũ dính thân."
        )
    },
    {
        "group_name": "Nhóm 4: Bệnh trên lá ớt chuông (Đốm vi khuẩn vs Thán thư ớt vs Đốm mắt ếch Cercospora)",
        "classes": [
            "Pepper__bell___Bacterial_spot",
            "Pepper__Anthracnose",
            "Pepper__Cercospora_leaf_spot"
        ],
        "confusion_reason": "Đều tạo ra các đốm nâu đen hoại tử trên phiến lá ớt và làm rụng lá hàng loạt.",
        "differentiation_guide": (
            "- Pepper Bacterial Spot: Đốm nhỏ góc cạnh 1-3mm, mặt dưới hơi gồ nốt mụn, không bị trũng lõm.\n"
            "- Pepper Anthracnose (Thán thư): Vết bệnh tròn hơi LÕM XUỐNG, có vân đồng tâm và có hạt nhầy màu hồng cam/đen khi ẩm; gây thối lõm nặng trên quả.\n"
            "- Cercospora (Đốm mắt ếch): Đốm to (5-10mm) hình tròn trịa, VIỀN NÂU ĐỎ ĐẬM, TÂM TRẮNG XÁM rõ rệt như mắt con ếch."
        )
    }
]

# ============================================================
# PHAN 4: KIEN THUC 6 LOAI CAY MOI - 26 LOP
# Chili, Coffee, Corn, Rubber, Mango, Rice
#
# Giu nguyen cac lop cu cua:
# - Tomato
# - Potato
# - Pepper bell
# - Cac benh tham khao chua train
#
# Khong thay doi schema DISEASE_DATABASE hien tai.
# ============================================================

def _add_plant_disease(
    key, crop, vietnamese_name, scientific_name,
    pathogen_type, danger_level, symptoms,
    farming_practices, biological_control,
    post_disease_recovery, similar_diseases,
    key_distinctions, chemical_control=None
):
    """Tao mot ban ghi kien thuc tuong thich Streamlit."""

    healthy = key.endswith("___healthy")
    virus = "virus" in pathogen_type.lower()

    if chemical_control is None:
        if healthy:
            chemical_control = [
                "Không cần sử dụng thuốc bảo vệ thực vật "
                "khi cây không có dấu hiệu dịch hại.",
                "Chỉ can thiệp khi đã xác định được "
                "một vấn đề cụ thể cần xử lý."
            ]
        elif virus:
            chemical_control = [
                "Không có thuốc bảo vệ thực vật nào "
                "loại bỏ được virus đã nhiễm bên trong cây.",
                "Nếu có côn trùng môi giới, cần xác định "
                "đúng đối tượng và áp dụng biện pháp "
                "quản lý được cơ quan chuyên môn khuyến cáo."
            ]
        else:
            chemical_control = [
                "Nếu bệnh tiếp tục phát triển, cần xác định "
                "đúng tác nhân trước khi cân nhắc thuốc bảo "
                "vệ thực vật được phép sử dụng cho cây này.",
                "Việc lựa chọn và sử dụng thuốc phải do "
                "người có chuyên môn thực hiện theo nhãn "
                "đăng ký và quy định an toàn; không lựa "
                "chọn thuốc chỉ từ kết quả AI."
            ]

    # Khong thay the cac khoa cua cay cu.
    # Ba lop Chili cu duoc phep cap nhat de co them
    # kien thuc va dung trang thai in_dataset.
    allowed_updates = {
        "Chili___healthy",
        "Chili___Bacterial_spot",
        "Chili___Cercospora_leaf_spot",
    }

    if key in DISEASE_DATABASE and key not in allowed_updates:
        raise ValueError(
            f"Khong the ghi de benh cu: {key}"
        )

    DISEASE_DATABASE[key] = {
        "crop": crop,
        "in_dataset": True,
        "vietnamese_name": vietnamese_name,
        "scientific_name": scientific_name,
        "pathogen_type": pathogen_type,
        "danger_level": danger_level,
        "symptoms": symptoms,
        "treatment_protocol": {
            "farming_practices": farming_practices,
            "biological_control": biological_control,
            "chemical_control": chemical_control,
        },
        "post_disease_recovery": post_disease_recovery,
        "confusion_analysis": {
            "similar_diseases": similar_diseases,
            "common_features": (
                "Một số triệu chứng trên lá có thể giống "
                "với bệnh khác hoặc tổn thương sinh lý."
            ),
            "key_distinctions": key_distinctions,
        }
    }


# ============================================================
# A. OT CAY - CHILI - 4 LOP
# ============================================================

_add_plant_disease(
    "Chili___Bacterial_spot",
    "Ớt cay (Chili Pepper)",
    "Bệnh đốm vi khuẩn trên lá ớt cay",
    "Xanthomonas spp.",
    "Vi khuẩn",
    "Cao - Có thể gây vàng lá, rụng lá và giảm năng suất.",
    [
        "Ban đầu xuất hiện các đốm nhỏ ngậm nước "
        "trên mặt lá, sau chuyển màu nâu sẫm hoặc đen.",
        "Đốm bệnh có thể có hình góc cạnh và quầng "
        "vàng bao quanh.",
        "Khi bị nặng, nhiều đốm liên kết khiến "
        "lá vàng, khô và rụng trước tuổi.",
        "Cây mất nhiều lá sẽ giảm quang hợp, "
        "hoa và quả phát triển kém."
    ],
    [
        "Tách cây bị bệnh nặng khỏi khu vực cây giống "
        "hoặc cây khỏe khi có thể.",
        "Loại bỏ những lá hư hại nặng, thu gom tàn dư "
        "không để lẫn vào luống trồng.",
        "Chuyển sang tưới gốc, hạn chế nước bắn "
        "từ lá bệnh sang lá khỏe.",
        "Vệ sinh dụng cụ chăm sóc và luân canh "
        "với cây khác họ cà.",
        "Sử dụng giống và cây con khỏe mạnh."
    ],
    [
        "Có thể cân nhắc chế phẩm vi sinh đối kháng "
        "được đăng ký phù hợp với cây ớt.",
        "Ưu tiên quản lý tổng hợp, duy trì thông thoáng "
        "và giảm nguồn lây hơn là chỉ dựa vào chế phẩm."
    ],
    [
        "Nếu bệnh mới xuất hiện trên một phần nhỏ tán, "
        "cây còn khỏe có thể tiếp tục phát triển sau "
        "khi kiểm soát được nguồn lây.",
        "Duy trì độ ẩm đất ổn định nhưng không để úng.",
        "Bón phân cân đối theo nhu cầu cây, tránh "
        "bón quá nhiều đạm làm tán non rậm và mềm.",
        "Theo dõi lá mới; lá đã hoại tử sẽ không "
        "phục hồi màu xanh như ban đầu."
    ],
    ["Chili___Cercospora_leaf_spot"],
    [
        "Đốm vi khuẩn thường nhỏ, sẫm màu và có "
        "thể xuất hiện quầng vàng.",
        "Cercospora thường tạo đốm tròn có tâm "
        "xám sáng và viền nâu tương đối rõ."
    ]
)


_add_plant_disease(
    "Chili___Cercospora_leaf_spot",
    "Ớt cay (Chili Pepper)",
    "Bệnh đốm mắt ếch Cercospora trên ớt",
    "Cercospora capsici và các loài liên quan",
    "Nấm",
    "Trung bình đến cao - Có thể làm rụng nhiều lá.",
    [
        "Xuất hiện các đốm tròn hoặc gần tròn.",
        "Tâm đốm chuyển màu xám trắng hoặc nâu nhạt, "
        "bao quanh là viền nâu đỏ.",
        "Mô lá xung quanh vết bệnh có thể vàng.",
        "Khi nhiều đốm hợp lại, lá khô và rụng, "
        "làm giảm khả năng nuôi quả."
    ],
    [
        "Dọn lá bệnh và tàn dư trong vườn.",
        "Tạo khoảng cách thông thoáng giữa các cây.",
        "Tưới gốc thay vì làm ướt toàn bộ tán.",
        "Hạn chế trồng liên tục ớt hoặc các cây "
        "cùng ký chủ tại khu đất thường xuyên bị bệnh."
    ],
    [
        "Có thể xem xét các chế phẩm sinh học đối kháng "
        "nấm có đăng ký và bằng chứng phù hợp.",
        "Kết hợp vệ sinh vườn, theo dõi thời tiết "
        "và quản lý dịch hại tổng hợp."
    ],
    [
        "Nếu phần lớn tán vẫn xanh, cây có khả năng "
        "ra lá mới sau khi bệnh được khống chế.",
        "Giữ đất đủ ẩm, không để rễ bị thiếu oxy do úng.",
        "Duy trì kali, đạm và các chất dinh dưỡng "
        "theo nhu cầu thực tế, không bón dồn quá mức.",
        "Theo dõi lá mới để phát hiện bệnh tái phát."
    ],
    ["Chili___Bacterial_spot"],
    [
        "Tâm đốm Cercospora thường sáng hơn và "
        "viền sẫm rõ hơn.",
        "Nên kiểm tra thêm mẫu khi triệu chứng hai "
        "bệnh đốm lá không điển hình."
    ]
)


_add_plant_disease(
    "Chili___Curl_virus",
    "Ớt cay (Chili Pepper)",
    "Triệu chứng xoăn lá ớt nghi do virus",
    "Chưa xác định được loài virus từ nhãn Curl_Virus",
    "Virus nghi ngờ",
    "Cao nếu đúng là virus - Có nguy cơ lây lan và giảm đậu quả.",
    [
        "Lá non co lại, cong lên hoặc xoắn bất thường.",
        "Một số lá có thể nhỏ, dày, nhăn hoặc vàng.",
        "Đọt sinh trưởng chậm, cây có thể bị lùn.",
        "Triệu chứng tương tự đôi khi do bọ trĩ, "
        "nhện, thuốc hoặc mất cân bằng sinh lý."
    ],
    [
        "Theo dõi cây nghi bệnh và tránh dùng cây "
        "đó làm nguồn nhân giống.",
        "Kiểm tra côn trùng chích hút như bọ phấn "
        "và bọ trĩ; mỗi virus có thể có môi giới khác nhau.",
        "Dùng lưới chắn côn trùng trong khu vực cây giống "
        "nếu điều kiện sản xuất cho phép.",
        "Vệ sinh dụng cụ, loại bỏ cây nhiễm được xác nhận "
        "nặng theo hướng dẫn của cán bộ kỹ thuật.",
        "Xét nghiệm khi cần xác định chính xác virus."
    ],
    [
        "Bảo vệ thiên địch của côn trùng chích hút.",
        "Dùng bẫy theo dõi quần thể côn trùng và "
        "quản lý tổng hợp để hạn chế lây lan."
    ],
    [
        "Cây đã nhiễm virus không thể chữa khỏi "
        "virus bằng phân bón hay thuốc trừ nấm.",
        "Nếu cây còn sinh trưởng, có thể duy trì nước "
        "và dinh dưỡng phù hợp để hạn chế suy kiệt.",
        "Theo dõi khả năng ra hoa, đậu quả thực tế.",
        "Ưu tiên bảo vệ cây khỏe xung quanh; cây "
        "nhiễm nặng có thể cần loại bỏ để giảm nguồn lây."
    ],
    ["Chili___healthy", "Chili___Mosaic_virus"],
    [
        "Không thể kết luận một loại virus cụ thể "
        "chỉ qua hình dạng xoăn lá.",
        "Cần phân biệt triệu chứng virus với tác hại "
        "côn trùng và điều kiện môi trường."
    ],
    [
        "Không có thuốc hóa học nào chữa khỏi "
        "virus đã tồn tại trong cây.",
        "Nếu có côn trùng môi giới, cần quản lý chúng "
        "bằng biện pháp phù hợp và hướng dẫn chuyên môn."
    ]
)


_add_plant_disease(
    "Chili___healthy",
    "Ớt cay (Chili Pepper)",
    "Lá ớt cay khỏe mạnh",
    "Capsicum spp.",
    "Không phát hiện tác nhân gây bệnh rõ ràng",
    "An toàn - Lá không có dấu hiệu bệnh nổi bật.",
    [
        "Lá xanh tương đối đồng đều và phát triển bình thường.",
        "Không có vết hoại tử hoặc đốm bệnh rõ.",
        "Đọt non phát triển bình thường, ít biến dạng."
    ],
    [
        "Tưới nước đều, giữ đất thoát nước tốt.",
        "Trồng với mật độ phù hợp để tán thông thoáng.",
        "Theo dõi cả mặt trên lẫn mặt dưới lá."
    ],
    [
        "Duy trì hệ sinh thái vườn và bảo vệ thiên địch.",
        "Áp dụng quản lý dịch hại tổng hợp để phòng bệnh."
    ],
    [
        "Duy trì dinh dưỡng cân đối trong giai đoạn "
        "sinh trưởng, ra hoa và nuôi quả.",
        "Theo dõi màu lá và sự phát triển của đọt non.",
        "Không dùng thuốc chỉ vì hệ thống nhận diện "
        "kết luận cây khỏe."
    ],
    ["Chili___Curl_virus"],
    [
        "Lá khỏe thường không có kiểu biến dạng "
        "xoăn bất thường kéo dài qua nhiều đợt lá."
    ]
)


# ============================================================
# B. CA PHE - COFFEE - 3 LOP
# ============================================================

_add_plant_disease(
    "Coffee___Rust",
    "Cà phê (Coffee)",
    "Bệnh gỉ sắt lá cà phê",
    "Hemileia vastatrix",
    "Nấm",
    "Cao - Có thể làm rụng lá mạnh và giảm năng suất vụ sau.",
    [
        "Mặt trên lá có những đốm vàng nhạt.",
        "Mặt dưới lá xuất hiện ổ bào tử dạng bột "
        "màu vàng cam hoặc cam nâu.",
        "Nhiều ổ bệnh hợp lại làm lá vàng và rụng sớm.",
        "Mất lá nhiều khiến cành mang quả suy yếu "
        "và giảm khả năng tích lũy dinh dưỡng."
    ],
    [
        "Theo dõi mặt dưới lá, nhất là trong thời kỳ "
        "ẩm ướt thuận lợi cho bệnh.",
        "Tỉa cành và quản lý tán để ánh sáng, không khí "
        "phân bố hợp lý.",
        "Dọn cỏ, tàn dư lá bệnh khi phù hợp và "
        "chăm sóc cây theo sức mang quả.",
        "Ưu tiên giống có khả năng kháng gỉ sắt "
        "khi trồng mới hoặc tái canh."
    ],
    [
        "Có thể xem xét các tác nhân sinh học có "
        "khả năng ức chế nấm gỉ sắt đã được đánh giá.",
        "Kết hợp quản lý tán và cân bằng dinh dưỡng; "
        "không coi chế phẩm sinh học là biện pháp "
        "chữa khỏi toàn bộ các vết bệnh."
    ],
    [
        "Cây vẫn có thể hồi phục nếu bộ rễ, cành "
        "và chồi sinh trưởng còn khỏe.",
        "Tưới nước phù hợp giai đoạn sinh trưởng, "
        "không gây khô hạn kéo dài.",
        "Bón phân cân đối dựa trên tình trạng cây, "
        "sản lượng và điều kiện đất.",
        "Để cây phát triển lứa lá mới, theo dõi "
        "bệnh trên mặt dưới lá non.",
        "Nếu cây suy kiệt vì mất lá nặng, cần đánh "
        "giá sức cây trước khi tiếp tục mang nhiều quả."
    ],
    ["Coffee___Phoma"],
    [
        "Gỉ sắt có ổ bột vàng cam nổi bật ở mặt dưới lá.",
        "Phoma gây các vùng hoại tử nâu nhưng không "
        "có đặc trưng ổ bột gỉ sắt như vậy."
    ]
)


_add_plant_disease(
    "Coffee___Phoma",
    "Cà phê (Coffee)",
    "Bệnh đốm lá hoặc cháy lá Phoma trên cà phê",
    "Phoma spp. và các loài liên quan",
    "Nấm",
    "Trung bình đến cao - Có thể gây hại lá và đọt non.",
    [
        "Vết bệnh màu nâu sẫm thường xuất hiện "
        "trên lá hoặc phần mô non.",
        "Vết hoại tử có thể lan từ mép hoặc chóp lá.",
        "Mô bị bệnh khô lại, nhiều vết có thể hợp thành "
        "mảng lớn.",
        "Bệnh nặng làm giảm diện tích lá quang hợp "
        "và có thể ảnh hưởng sinh trưởng đọt."
    ],
    [
        "Theo dõi lá non, cành non và phần tán "
        "thường xuyên ẩm kéo dài.",
        "Tỉa cành hợp lý, hạn chế bộ phận cây "
        "bị che kín và thiếu thông thoáng.",
        "Thu gom bộ phận nhiễm nặng khi phù hợp.",
        "Tránh tạo thêm vết thương cơ học không cần thiết."
    ],
    [
        "Xem xét các biện pháp vi sinh đối kháng nấm "
        "phù hợp theo khuyến cáo địa phương.",
        "Ưu tiên phối hợp quản lý ẩm độ, chăm sóc "
        "vườn và giám sát bệnh."
    ],
    [
        "Nếu chỉ một phần lá bị bệnh, cành khỏe "
        "vẫn có thể phát triển chồi mới.",
        "Duy trì đủ nước nhưng hạn chế ẩm kéo dài "
        "trên bề mặt lá.",
        "Cung cấp dinh dưỡng cân đối, không kích "
        "đọt quá mạnh khi điều kiện còn thuận lợi cho bệnh.",
        "Theo dõi đợt lá mới và kiểm tra xem vết bệnh "
        "còn xuất hiện hay không."
    ],
    ["Coffee___Rust"],
    [
        "Phoma thường biểu hiện dạng đốm hoặc "
        "mảng hoại tử nâu.",
        "Gỉ sắt dễ nhận ra hơn khi mặt dưới lá "
        "có ổ bào tử màu vàng cam."
    ]
)


_add_plant_disease(
    "Coffee___healthy",
    "Cà phê (Coffee)",
    "Lá cà phê khỏe mạnh",
    "Coffea spp.",
    "Không phát hiện tác nhân gây bệnh rõ ràng",
    "An toàn",
    [
        "Lá xanh, hình dạng tương đối bình thường.",
        "Không có đốm hoại tử hoặc ổ gỉ sắt rõ.",
        "Lá có thể khác màu tùy giống và tuổi lá."
    ],
    [
        "Tạo tán và duy trì độ che bóng phù hợp giống.",
        "Tưới tiêu theo điều kiện đất và thời tiết.",
        "Theo dõi cả hai mặt lá định kỳ."
    ],
    [
        "Duy trì quản lý dịch hại tổng hợp.",
        "Bảo vệ sinh vật có ích trong vườn."
    ],
    [
        "Bón phân theo tuổi cây, năng suất và đất.",
        "Giữ bộ rễ khỏe bằng thoát nước phù hợp.",
        "Kiểm tra đợt lá non để nhận biết bệnh sớm."
    ],
    ["Coffee___Rust", "Coffee___Phoma"],
    [
        "Lá khỏe không có ổ bột vàng cam "
        "hoặc vết hoại tử lan rộng."
    ]
)


# ============================================================
# C. NGO - CORN - 4 LOP
# ============================================================

_add_plant_disease(
    "Corn___Blight",
    "Ngô (Corn)",
    "Bệnh cháy lá ngô",
    "Nhãn Blight chưa xác định tác nhân cụ thể",
    "Thường do nấm; cần xác minh tác nhân",
    "Cao nếu bệnh xuất hiện sớm và lan rộng trên nhiều lá.",
    [
        "Lá có vết bệnh dài hoặc hình bầu dục, "
        "màu xám xanh lúc đầu rồi chuyển nâu.",
        "Các vết có thể lớn lên và liên kết, "
        "làm khô diện tích lá đáng kể.",
        "Bệnh nặng trước khi ngô phát triển hạt "
        "có thể làm giảm năng suất.",
        "Tùy tác nhân, hình dạng vết bệnh có "
        "thể khác nhau."
    ],
    [
        "Sử dụng giống ngô có khả năng chống chịu.",
        "Luân canh cây trồng và quản lý tàn dư ngô "
        "từ vụ trước.",
        "Theo dõi bệnh từ lá dưới lên lá phía trên.",
        "Giữ mật độ gieo trồng và dinh dưỡng hợp lý.",
        "Xác định loại cháy lá trước khi chọn biện pháp xử lý."
    ],
    [
        "Áp dụng biện pháp sinh học phù hợp với "
        "loại nấm đã được xác định.",
        "Ưu tiên phòng bệnh bằng quản lý đồng ruộng "
        "và giống chống chịu."
    ],
    [
        "Nếu ngô còn nhiều lá xanh, vẫn có khả năng "
        "quang hợp để tiếp tục nuôi bắp.",
        "Giữ đủ nước trong các giai đoạn phát triển "
        "quan trọng, tránh để cây thiếu nước kéo dài.",
        "Không cắt bỏ hàng loạt lá còn xanh chỉ "
        "vì có vài đốm bệnh.",
        "Theo dõi tốc độ phát triển của vết bệnh "
        "và mức độ xanh của lá phía trên."
    ],
    ["Corn___Gray_leaf_spot", "Corn___Common_rust"],
    [
        "Nhãn Blight là nhóm triệu chứng rộng.",
        "Vết cháy lá thường lớn và kéo dài hơn "
        "đốm xám nhỏ ban đầu."
    ]
)


_add_plant_disease(
    "Corn___Common_rust",
    "Ngô (Corn)",
    "Bệnh gỉ sắt thông thường trên ngô",
    "Puccinia sorghi",
    "Nấm",
    "Trung bình - Có thể gây hại đáng kể trên giống mẫn cảm.",
    [
        "Xuất hiện các ổ bào tử nổi màu nâu đỏ "
        "trên mặt lá.",
        "Ổ bào tử có thể xuất hiện cả hai mặt lá.",
        "Khi bệnh nặng, mô lá vàng và có thể khô.",
        "Bệnh thường dễ phát hiện nhờ các ổ "
        "gỉ nổi trên bề mặt lá."
    ],
    [
        "Ưu tiên giống có khả năng kháng bệnh.",
        "Theo dõi đồng ruộng trong thời tiết mát và ẩm.",
        "Đánh giá mức độ bệnh trên lá, không kết luận "
        "chỉ từ một vài ổ gỉ nhỏ.",
        "Giữ dinh dưỡng và mật độ cây phù hợp."
    ],
    [
        "Ưu tiên giống chống chịu và quản lý "
        "dịch hại tổng hợp.",
        "Biện pháp sinh học chỉ nên được áp dụng "
        "khi có bằng chứng phù hợp với bệnh gỉ sắt."
    ],
    [
        "Nếu mức độ bệnh nhẹ, cây có thể tiếp tục "
        "sinh trưởng và tạo hạt bình thường.",
        "Duy trì nước tưới phù hợp để cây tránh stress.",
        "Theo dõi lá phía trên vì chúng rất quan trọng "
        "cho quang hợp trong giai đoạn nuôi hạt."
    ],
    ["Corn___Gray_leaf_spot", "Corn___Blight"],
    [
        "Gỉ sắt tạo ổ bào tử màu nâu đỏ nổi lên.",
        "Đốm xám chủ yếu tạo vết phẳng dài "
        "theo giới hạn gân lá."
    ]
)


_add_plant_disease(
    "Corn___Gray_leaf_spot",
    "Ngô (Corn)",
    "Bệnh đốm xám lá ngô",
    "Cercospora zeae-maydis và loài liên quan",
    "Nấm",
    "Cao trên giống mẫn cảm trong điều kiện ẩm kéo dài.",
    [
        "Đốm ban đầu nhỏ, màu nâu hoặc xám.",
        "Vết bệnh phát triển thành hình chữ nhật "
        "hoặc thuôn dài theo gân lá.",
        "Tâm vết bệnh màu xám, bờ vết tương đối thẳng.",
        "Nhiều đốm liên kết làm lá cháy thành mảng."
    ],
    [
        "Ưu tiên giống chống chịu.",
        "Luân canh cây trồng để hạn chế tích lũy nguồn bệnh.",
        "Quản lý tàn dư lá ngô sau thu hoạch.",
        "Theo dõi bệnh khi thời tiết ẩm kéo dài.",
        "Duy trì mật độ gieo trồng hợp lý."
    ],
    [
        "Áp dụng biện pháp sinh học có bằng chứng "
        "phù hợp khi được khuyến cáo.",
        "Kết hợp vệ sinh đồng ruộng và quản lý giống."
    ],
    [
        "Cây còn nhiều lá xanh có thể tiếp tục "
        "nuôi bắp nếu bệnh được kiểm soát.",
        "Giữ đủ nước và dinh dưỡng trong giai đoạn "
        "trổ cờ, phun râu và hình thành hạt.",
        "Theo dõi các lá gần bắp, hạn chế để "
        "bệnh lan rộng lên tầng lá quan trọng."
    ],
    ["Corn___Blight", "Corn___Common_rust"],
    [
        "Đốm xám thường có hai cạnh gần song song "
        "do bị giới hạn bởi gân lá.",
        "Gỉ sắt xuất hiện các ổ bào tử nổi."
    ]
)


_add_plant_disease(
    "Corn___healthy",
    "Ngô (Corn)",
    "Lá ngô khỏe mạnh",
    "Zea mays",
    "Không phát hiện bệnh rõ ràng",
    "An toàn",
    [
        "Lá xanh tương đối đồng đều theo giống.",
        "Không có vùng hoại tử hoặc ổ gỉ sắt nổi.",
        "Phiến lá phát triển bình thường."
    ],
    [
        "Duy trì mật độ gieo trồng hợp lý.",
        "Đảm bảo nước tưới và dinh dưỡng phù hợp.",
        "Kiểm tra các lá dưới để phát hiện bệnh sớm."
    ],
    [
        "Quản lý dịch hại tổng hợp.",
        "Bảo vệ các sinh vật có ích trong ruộng."
    ],
    [
        "Duy trì dinh dưỡng phù hợp giai đoạn phát triển.",
        "Chú ý nhu cầu nước khi ngô trổ cờ và nuôi hạt.",
        "Không cần xử lý thuốc khi không có dịch hại."
    ],
    ["Corn___Common_rust", "Corn___Gray_leaf_spot"],
    [
        "Lá khỏe không có ổ gỉ sắt nâu đỏ "
        "hoặc đốm xám hình chữ nhật."
    ]
)


# ============================================================
# D. CAO SU - RUBBER - 4 LOP
# ============================================================

_add_plant_disease(
    "Rubber___Anthracnose",
    "Cao su (Rubber)",
    "Bệnh thán thư lá cao su",
    "Colletotrichum spp.",
    "Nấm",
    "Cao nếu gây rụng nhiều lá non hoặc lặp lại nhiều đợt.",
    [
        "Lá non có đốm nhỏ màu nâu, đôi khi có quầng vàng.",
        "Vết bệnh lớn lên tạo mảng hoại tử không đều.",
        "Lá non có thể nhăn, cong hoặc biến dạng.",
        "Nhiễm nặng làm lá rụng sớm, giảm tán lá."
    ],
    [
        "Theo dõi các đợt ra lá non và giai đoạn thay lá.",
        "Giữ vườn ươm thông thoáng.",
        "Loại bỏ bộ phận bệnh nặng khi phù hợp.",
        "Không dùng vật liệu nhân giống có triệu chứng bệnh.",
        "Quản lý tàn dư lá bệnh và nước đọng."
    ],
    [
        "Có thể cân nhắc chế phẩm sinh học đối kháng "
        "nấm đã được đánh giá phù hợp.",
        "Quản lý nguồn bệnh và điều kiện vườn "
        "là phần quan trọng của phòng bệnh."
    ],
    [
        "Nếu cây vẫn còn chồi và bộ rễ khỏe, "
        "có khả năng tái tạo lá sau đợt bệnh.",
        "Bảo đảm đất thoát nước và dinh dưỡng cân đối.",
        "Tránh gây thêm stress cho cây đang mất lá nhiều.",
        "Đánh giá mật độ tán mới trước khi khôi phục "
        "các hoạt động khai thác thông thường."
    ],
    ["Rubber___Leaf_spot", "Rubber___Dry_leaf"],
    [
        "Thán thư thường gây hoại tử và biến dạng "
        "rõ trên lá non.",
        "Nhãn Leaf_spot có thể do nhiều loài nấm "
        "nên cần kiểm tra thêm để phân biệt."
    ]
)


_add_plant_disease(
    "Rubber___Dry_leaf",
    "Cao su (Rubber)",
    "Hiện tượng khô lá cao su",
    "Chưa xác định tác nhân từ nhãn Dry_Leaf",
    "Triệu chứng sinh lý hoặc bệnh",
    "Tùy nguyên nhân - Không nên mặc định là bệnh do nấm.",
    [
        "Méplá hoặc một phần phiến lá chuyển màu nâu khô.",
        "Lá có thể giòn, mất nước hoặc rụng.",
        "Hiện tượng có thể liên quan thiếu nước, "
        "rễ hư, dinh dưỡng hoặc sinh vật gây bệnh.",
        "Cần quan sát cả cây, rễ, đất và vết bệnh."
    ],
    [
        "Kiểm tra độ ẩm đất và khả năng thoát nước.",
        "Đánh giá bộ rễ, tình trạng đất và lượng dinh dưỡng.",
        "Theo dõi khô lá có xuất hiện đồng loạt "
        "sau thời tiết bất lợi hay không.",
        "Kiểm tra dấu hiệu nấm hoặc côn trùng trước "
        "khi quyết định cách xử lý."
    ],
    [
        "Chỉ sử dụng biện pháp sinh học đặc hiệu "
        "nếu đã có bằng chứng về tác nhân gây bệnh.",
        "Đối với stress sinh lý, tập trung khắc phục "
        "môi trường thay vì dùng chế phẩm diệt nấm."
    ],
    [
        "Nếu cây chỉ bị stress nhẹ, cải thiện nước "
        "và điều kiện rễ có thể giúp cây ra lá mới.",
        "Không cố phục hồi lá đã khô hoàn toàn.",
        "Bón dinh dưỡng sau khi xác định tình trạng đất, "
        "tránh bón quá mức khi rễ đang suy yếu.",
        "Theo dõi sự hình thành lá mới và sức sống của chồi."
    ],
    ["Rubber___Leaf_spot", "Rubber___Anthracnose"],
    [
        "Khô lá là triệu chứng chung, không phải "
        "tên một tác nhân gây bệnh cụ thể.",
        "Đốm lá do nấm thường có vết bệnh riêng biệt "
        "trước khi lan thành vùng hoại tử."
    ],
    [
        "Không nên dùng thuốc bảo vệ thực vật chỉ "
        "vì thấy lá cao su khô.",
        "Cần xác định nguyên nhân trước khi cân nhắc "
        "biện pháp xử lý chuyên môn."
    ]
)


_add_plant_disease(
    "Rubber___Leaf_spot",
    "Cao su (Rubber)",
    "Bệnh đốm lá cao su",
    "Nhiều loài nấm có thể gây triệu chứng tương tự",
    "Nấm hoặc tác nhân khác chưa xác định",
    "Trung bình đến cao, tùy mức độ rụng lá.",
    [
        "Trên lá xuất hiện đốm nâu, đen hoặc xám.",
        "Vết bệnh có thể có viền sẫm và quầng vàng.",
        "Nhiều đốm liên kết tạo mảng lá khô.",
        "Bệnh nặng có thể làm giảm mật độ tán lá."
    ],
    [
        "Thu gom và quản lý lá rụng có triệu chứng bệnh.",
        "Quan sát sự phát triển của đốm trên lá non.",
        "Duy trì độ thông thoáng và chăm sóc vườn phù hợp.",
        "Lấy mẫu xác định tác nhân khi bệnh lan rộng."
    ],
    [
        "Chỉ lựa chọn chế phẩm sinh học phù hợp "
        "sau khi xác định nhóm tác nhân gây bệnh.",
        "Kết hợp vệ sinh vườn và duy trì sức sinh trưởng."
    ],
    [
        "Giữ cây đủ dinh dưỡng để tái tạo tán lá.",
        "Theo dõi số lượng lá mới và mức độ rụng lá.",
        "Cây bị mất lá nặng cần được đánh giá sức khỏe "
        "trước khi tiếp tục khai thác.",
        "Ưu tiên duy trì bộ rễ và thân khỏe."
    ],
    ["Rubber___Anthracnose", "Rubber___Dry_leaf"],
    [
        "Nhãn Leaf_spot là nhóm triệu chứng rộng.",
        "Có thể cần giám định tác nhân để phân biệt "
        "thán thư với các bệnh đốm lá khác."
    ]
)


_add_plant_disease(
    "Rubber___healthy",
    "Cao su (Rubber)",
    "Lá cao su khỏe mạnh",
    "Hevea brasiliensis",
    "Không phát hiện bệnh rõ ràng",
    "An toàn",
    [
        "Lá phát triển bình thường theo tuổi lá.",
        "Không có đốm hoại tử rõ hoặc dấu hiệu rụng bất thường.",
        "Lá non có thể khác màu so với lá trưởng thành."
    ],
    [
        "Theo dõi các đợt thay lá và ra lá mới.",
        "Duy trì đất thoát nước và dinh dưỡng phù hợp.",
        "Giữ vệ sinh vườn và khu vực cây giống."
    ],
    [
        "Duy trì quản lý dịch hại tổng hợp.",
        "Bảo vệ sinh vật có ích trong vườn."
    ],
    [
        "Chăm sóc theo tuổi cây và mùa vụ.",
        "Theo dõi mật độ tán lá và sức sinh trưởng.",
        "Không dùng thuốc khi không có dịch hại."
    ],
    ["Rubber___Anthracnose", "Rubber___Leaf_spot"],
    [
        "Lá không có vùng đốm hoại tử đặc trưng "
        "của các bệnh đã biết."
    ]
)


# ============================================================
# E. XOAI - MANGO - 8 LOP
# ============================================================

_add_plant_disease(
    "Mango___Anthracnose",
    "Xoài (Mango)",
    "Bệnh thán thư trên xoài",
    "Colletotrichum spp.",
    "Nấm",
    "Cao - Có thể gây hại lá, hoa và quả.",
    [
        "Trên lá có đốm hoại tử nâu đen.",
        "Đốm lan rộng hoặc liên kết thành mảng.",
        "Trên chùm hoa, bệnh có thể gây khô hoa.",
        "Trên quả, một số vết bệnh có thể xuất hiện "
        "rõ hơn khi quả chín."
    ],
    [
        "Tỉa cành để tán cây thông thoáng.",
        "Thu gom quả và bộ phận nhiễm bệnh nặng.",
        "Hạn chế tổn thương cơ học khi chăm sóc.",
        "Theo dõi bệnh trước và trong mùa ra hoa.",
        "Quản lý ẩm độ và vệ sinh vườn."
    ],
    [
        "Ưu tiên quản lý tổng hợp từ khâu vệ sinh vườn.",
        "Có thể cân nhắc chế phẩm đối kháng nấm "
        "có cơ sở phù hợp với điều kiện thực tế."
    ],
    [
        "Cành và lá còn khỏe vẫn có thể sinh trưởng "
        "sau khi áp lực bệnh được giảm.",
        "Duy trì tưới nước và dinh dưỡng hợp lý.",
        "Không kích cây ra đọt quá mạnh khi điều kiện "
        "vẫn thuận lợi cho nấm.",
        "Theo dõi đợt lộc mới và bảo vệ chùm hoa khỏe."
    ],
    ["Mango___Bacterial_canker", "Mango___Die_back"],
    [
        "Thán thư chủ yếu gây vết hoại tử trên lá, "
        "hoa và quả.",
        "Loét vi khuẩn có thể làm tổn thương sần hoặc nứt."
    ]
)


_add_plant_disease(
    "Mango___Bacterial_canker",
    "Xoài (Mango)",
    "Bệnh loét vi khuẩn trên xoài",
    "Xanthomonas citri pv. mangiferaeindicae",
    "Vi khuẩn",
    "Cao - Có thể ảnh hưởng lá, cành và quả.",
    [
        "Lá có đốm sẫm màu, đôi khi kèm quầng vàng.",
        "Vết bệnh có thể trở nên sần hoặc nứt.",
        "Triệu chứng cũng có thể xuất hiện trên quả.",
        "Khi bệnh nặng, vết tổn thương có thể lan rộng."
    ],
    [
        "Sử dụng cây giống khỏe, tránh nguồn bệnh từ vườn ươm.",
        "Vệ sinh dụng cụ chăm sóc.",
        "Tránh gây vết thương không cần thiết trên cây.",
        "Giữ tán thông thoáng.",
        "Thu gom bộ phận nhiễm bệnh nặng theo hướng dẫn "
        "quản lý vườn."
    ],
    [
        "Ưu tiên vệ sinh, quản lý nguồn bệnh "
        "và kiểm soát lây lan.",
        "Có thể tham khảo các biện pháp sinh học "
        "được đánh giá phù hợp cho bệnh vi khuẩn."
    ],
    [
        "Nếu bệnh mới khu trú, các phần cành khỏe "
        "vẫn có thể tiếp tục phát triển.",
        "Duy trì dinh dưỡng cân đối, tránh làm cây "
        "suy yếu do thiếu nước hoặc úng.",
        "Theo dõi các vết bệnh mới trên lá và quả.",
        "Không kỳ vọng mô đã bị loét nặng sẽ "
        "phục hồi hoàn toàn."
    ],
    ["Mango___Anthracnose"],
    [
        "Vết loét vi khuẩn có thể sần và nứt.",
        "Thán thư thường gây vùng hoại tử nâu đen "
        "do nấm, nhưng cần giám định khi khó phân biệt."
    ]
)


_add_plant_disease(
    "Mango___Cutting_weevil",
    "Xoài (Mango)",
    "Bọ vòi voi cắt lá xoài",
    "Deporaus spp. và loài liên quan",
    "Côn trùng gây hại",
    "Trung bình - Có thể ảnh hưởng đợt lộc non.",
    [
        "Lá non bị cắt hoặc gãy rụng bất thường.",
        "Một số lá có dấu vết tổn thương ở cuống.",
        "Có thể tìm thấy lá mới bị cắt dưới gốc.",
        "Đợt lộc non bị hại nặng làm cây chậm phát triển."
    ],
    [
        "Theo dõi các đợt ra lộc non.",
        "Thu gom lá bị cắt rụng khi phù hợp.",
        "Kiểm tra sự hiện diện của côn trùng trên cây.",
        "Giữ vệ sinh vườn và hạn chế nguồn côn trùng "
        "tồn lưu từ các đợt lộc trước."
    ],
    [
        "Bảo vệ thiên địch ăn mồi và ký sinh.",
        "Áp dụng theo dõi quần thể và quản lý dịch hại "
        "tổng hợp, tránh xử lý khi chưa có nhu cầu."
    ],
    [
        "Nếu chồi sinh trưởng còn khỏe, xoài có "
        "khả năng phát triển đợt lá mới.",
        "Duy trì nước tưới và dinh dưỡng phù hợp.",
        "Bảo vệ đợt lộc tiếp theo trước tác hại côn trùng.",
        "Không cắt bỏ các lá còn xanh không cần thiết."
    ],
    ["Mango___Gall_midge"],
    [
        "Bọ vòi voi gây hiện tượng lá bị cắt hoặc rụng.",
        "Muỗi gây u sần thường tạo nốt hoặc u "
        "trên phiến lá."
    ],
    [
        "Chỉ cân nhắc biện pháp kiểm soát côn trùng "
        "được phép sử dụng nếu mật độ gây hại đáng kể.",
        "Ưu tiên quản lý tổng hợp và hướng dẫn chuyên môn."
    ]
)


_add_plant_disease(
    "Mango___Die_back",
    "Xoài (Mango)",
    "Bệnh khô cành, chết ngược trên xoài",
    "Lasiodiplodia spp. và các tác nhân liên quan",
    "Nấm hoặc tác nhân khác",
    "Cao - Có thể làm chết nhiều đầu cành.",
    [
        "Phần đầu cành có biểu hiện héo và khô dần.",
        "Lá trên cành nhiễm bệnh chuyển nâu hoặc khô.",
        "Vùng khô có thể tiến từ ngọn về phía thân.",
        "Chỉ ảnh lá đơn lẻ thường không đủ để "
        "khẳng định bệnh chết ngược."
    ],
    [
        "Quan sát cành và thân để xác định phạm vi bệnh.",
        "Loại bỏ phần cành chết hoặc hư hại nặng "
        "bằng quy trình tỉa cành an toàn.",
        "Vệ sinh dụng cụ giữa các cây.",
        "Hạn chế làm tổn thương cây không cần thiết.",
        "Đảm bảo đất thoát nước tốt và kiểm tra rễ."
    ],
    [
        "Có thể tham khảo chế phẩm vi sinh hỗ trợ "
        "quản lý nấm tại vườn ươm khi phù hợp.",
        "Ưu tiên giảm nguồn bệnh và chăm sóc sức khỏe cây."
    ],
    [
        "Nếu thân chính và các cành còn lại khỏe, "
        "cây có thể phát triển chồi mới.",
        "Duy trì nước và dinh dưỡng cân đối.",
        "Theo dõi điểm tiếp giáp giữa cành khỏe "
        "và phần đã bị khô.",
        "Không cố giữ cành đã chết hoàn toàn.",
        "Với cây mất nhiều cành, cần đánh giá sức cây "
        "trước khi kỳ vọng ra hoa, mang quả trở lại."
    ],
    ["Mango___Anthracnose"],
    [
        "Chết ngược biểu hiện rõ trên đầu cành và "
        "tiến dần về thân.",
        "Thán thư có thể gây đốm lá và tổn thương hoa, quả."
    ]
)


_add_plant_disease(
    "Mango___Gall_midge",
    "Xoài (Mango)",
    "Muỗi gây u sần lá xoài",
    "Procontarinia spp. và loài liên quan",
    "Côn trùng gây hại",
    "Trung bình đến cao khi tấn công mạnh lá non.",
    [
        "Trên lá xuất hiện các nốt hoặc u sần nhỏ.",
        "Tổn thương có thể chuyển màu nâu hoặc sẫm.",
        "Lá non nhiễm nặng có thể biến dạng.",
        "Mức độ tổn thương khác nhau theo loài muỗi gây hại."
    ],
    [
        "Theo dõi các đợt phát triển lộc non.",
        "Thu gom lá nhiễm nặng khi phù hợp.",
        "Giữ vườn thông thoáng.",
        "Theo dõi sự xuất hiện của côn trùng trưởng thành "
        "và tỷ lệ lá mới bị hại."
    ],
    [
        "Bảo vệ thiên địch ký sinh và bắt mồi.",
        "Duy trì quản lý dịch hại tổng hợp, "
        "tránh can thiệp không cần thiết."
    ],
    [
        "Lá đã hình thành u sần thường không "
        "trở lại hình dạng ban đầu.",
        "Nếu chồi còn khỏe, cây có thể ra đợt lá mới.",
        "Duy trì nước và dinh dưỡng cân đối.",
        "Theo dõi đợt lộc tiếp theo để đánh giá "
        "hiệu quả kiểm soát côn trùng."
    ],
    ["Mango___Cutting_weevil"],
    [
        "Muỗi gây u sần tạo các nốt trên lá.",
        "Bọ vòi voi cắt lá gây đứt hoặc rụng lá non."
    ],
    [
        "Chỉ xem xét kiểm soát côn trùng bằng sản phẩm "
        "được phép khi cần thiết và có hướng dẫn chuyên môn.",
        "Không dùng thuốc trừ nấm để xử lý u sần do côn trùng."
    ]
)


_add_plant_disease(
    "Mango___Powdery_mildew",
    "Xoài (Mango)",
    "Bệnh phấn trắng xoài",
    "Nhóm nấm gây phấn trắng trên xoài",
    "Nấm",
    "Cao khi gây hại hoa và ảnh hưởng đậu quả.",
    [
        "Trên lá non hoặc hoa có lớp bột trắng.",
        "Mô bị bệnh có thể chuyển vàng hoặc nâu.",
        "Chùm hoa nhiễm nặng có thể khô.",
        "Bệnh có thể ảnh hưởng khả năng đậu quả."
    ],
    [
        "Theo dõi thời kỳ ra hoa và lộc non.",
        "Tạo tán thông thoáng, tránh mật độ cành quá dày.",
        "Duy trì vệ sinh vườn.",
        "Kiểm tra điều kiện thời tiết thuận lợi cho bệnh."
    ],
    [
        "Có thể cân nhắc biện pháp sinh học có "
        "đánh giá phù hợp cho nhóm nấm phấn trắng.",
        "Kết hợp theo dõi bệnh sớm và quản lý tán cây."
    ],
    [
        "Lá mới có thể phát triển sau khi bệnh "
        "được kiểm soát.",
        "Duy trì dinh dưỡng cân đối, không kích đọt "
        "quá mạnh khi cây đang bị bệnh.",
        "Theo dõi hoa khỏe và khả năng đậu quả.",
        "Không kỳ vọng chùm hoa đã khô nặng sẽ hồi phục."
    ],
    ["Mango___Sooty_mould"],
    [
        "Phấn trắng có lớp bột màu trắng.",
        "Muội đen tạo lớp phủ đen như bồ hóng."
    ]
)


_add_plant_disease(
    "Mango___Sooty_mould",
    "Xoài (Mango)",
    "Bệnh muội đen trên xoài",
    "Nhóm nấm muội đen phát triển trên dịch mật",
    "Nấm bề mặt, thường liên quan côn trùng",
    "Trung bình - Có thể làm giảm quang hợp.",
    [
        "Lá xuất hiện lớp phủ màu đen như bồ hóng.",
        "Lớp đen nằm trên bề mặt lá, có thể lan rộng.",
        "Bệnh thường đi kèm côn trùng tiết dịch mật "
        "như rệp.",
        "Lớp phủ dày có thể làm giảm ánh sáng đến lá."
    ],
    [
        "Kiểm tra mặt dưới lá và cành để tìm rệp "
        "hoặc côn trùng tiết dịch mật.",
        "Quản lý nguồn dịch mật là biện pháp quan trọng.",
        "Tỉa cành để tán thông thoáng.",
        "Giữ vệ sinh vườn và theo dõi sự xuất hiện "
        "của lớp muội trên lá mới."
    ],
    [
        "Bảo vệ thiên địch của rệp và côn trùng "
        "tiết dịch mật.",
        "Ưu tiên các biện pháp quản lý tổng hợp "
        "đối với nguồn phát sinh lớp muội."
    ],
    [
        "Khi nguồn dịch mật giảm, lớp muội mới "
        "sẽ ít xuất hiện hơn.",
        "Các lá còn xanh có thể tiếp tục quang hợp "
        "sau khi bề mặt lá được cải thiện.",
        "Duy trì dinh dưỡng và nước tưới phù hợp.",
        "Theo dõi rệp và các đợt lá mới để ngăn tái phát."
    ],
    ["Mango___Powdery_mildew"],
    [
        "Muội đen tạo lớp bám màu đen.",
        "Phấn trắng tạo lớp bột màu trắng.",
        "Cần tìm nguồn côn trùng tiết dịch mật "
        "thay vì chỉ xử lý lớp muội."
    ],
    [
        "Không nên chỉ tập trung vào thuốc trừ nấm "
        "mà bỏ qua nguyên nhân tiết dịch mật.",
        "Quản lý côn trùng liên quan theo hướng dẫn IPM "
        "và quy định hiện hành."
    ]
)


_add_plant_disease(
    "Mango___healthy",
    "Xoài (Mango)",
    "Lá xoài khỏe mạnh",
    "Mangifera indica",
    "Không phát hiện bệnh rõ ràng",
    "An toàn",
    [
        "Lá phát triển bình thường theo tuổi.",
        "Lá non có thể màu đồng hoặc đỏ nhạt tùy giống.",
        "Không có đốm hoại tử hay lớp mốc nổi bật."
    ],
    [
        "Giữ tán cây thông thoáng.",
        "Tưới nước phù hợp giai đoạn sinh trưởng.",
        "Theo dõi lộc non, lá, chùm hoa và quả."
    ],
    [
        "Bảo vệ thiên địch trong vườn.",
        "Duy trì quản lý dịch hại tổng hợp."
    ],
    [
        "Bón phân theo tuổi cây và giai đoạn ra hoa.",
        "Theo dõi sức phát triển của đợt lộc mới.",
        "Không cần dùng thuốc khi không có dịch hại."
    ],
    ["Mango___Anthracnose", "Mango___Powdery_mildew"],
    [
        "Lá khỏe không có vùng hoại tử hoặc "
        "lớp phấn trắng bất thường."
    ]
)


# ============================================================
# F. LUA NUOC - RICE - 3 LOP
# ============================================================

_add_plant_disease(
    "Rice___Bacterial_leaf_blight",
    "Lúa nước (Rice)",
    "Bệnh bạc lá vi khuẩn trên lúa",
    "Xanthomonas oryzae pv. oryzae",
    "Vi khuẩn",
    "Cao - Có thể làm giảm diện tích lá quang hợp và năng suất.",
    [
        "Vết bệnh thường bắt đầu từ chóp hoặc mép lá.",
        "Vết bệnh kéo dài dọc phiến lá, chuyển từ "
        "vàng nhạt sang trắng bạc hoặc nâu khô.",
        "Ranh giới vùng bệnh có thể không đều.",
        "Bệnh nặng làm khô nhiều diện tích lá."
    ],
    [
        "Ưu tiên giống chống chịu phù hợp địa phương.",
        "Quản lý phân đạm cân đối, không bón thừa.",
        "Điều tiết nước hợp lý theo giai đoạn cây.",
        "Hạn chế gây tổn thương lá khi chăm sóc.",
        "Không dùng nguồn giống từ ruộng có nguy cơ "
        "nhiễm bệnh cao nếu chưa kiểm tra."
    ],
    [
        "Ưu tiên quản lý giống, nước và dinh dưỡng.",
        "Chỉ xem xét chế phẩm sinh học có hiệu quả "
        "được xác nhận phù hợp với bệnh vi khuẩn."
    ],
    [
        "Mô lá bị bạc và chết sẽ không xanh lại.",
        "Nếu lúa còn lá non và điểm sinh trưởng khỏe, "
        "cây có thể tiếp tục sinh trưởng.",
        "Duy trì nước và dinh dưỡng cân đối.",
        "Theo dõi lá mới và lá đòng vì chúng có "
        "vai trò quan trọng trong nuôi hạt."
    ],
    ["Rice___Brown_spot", "Rice___Leaf_smut"],
    [
        "Bạc lá thường tạo vệt dài từ chóp hoặc mép lá.",
        "Đốm nâu có nhiều vết bệnh rời rạc hơn."
    ]
)


_add_plant_disease(
    "Rice___Brown_spot",
    "Lúa nước (Rice)",
    "Bệnh đốm nâu trên lúa",
    "Bipolaris oryzae",
    "Nấm",
    "Trung bình đến cao - Có thể nặng khi cây suy dinh dưỡng.",
    [
        "Lá có các đốm tròn hoặc bầu dục màu nâu.",
        "Một số đốm có tâm xám hoặc nâu sáng.",
        "Viền vết bệnh thường sẫm hơn phần trung tâm.",
        "Lá bị nhiều đốm có thể vàng và khô sớm."
    ],
    [
        "Sử dụng hạt giống khỏe và ruộng giống tốt.",
        "Bón phân cân đối, đặc biệt khi đất thiếu dinh dưỡng.",
        "Quản lý nước theo giai đoạn sinh trưởng.",
        "Quản lý tàn dư sau vụ thu hoạch.",
        "Theo dõi sự phát triển bệnh trên các lá mới."
    ],
    [
        "Có thể cân nhắc biện pháp sinh học phù hợp "
        "với nấm gây đốm nâu nếu đã được đánh giá.",
        "Ưu tiên quản lý sức khỏe cây và điều kiện đất."
    ],
    [
        "Nếu bệnh chưa làm mất nhiều lá, cây có thể "
        "tiếp tục phát triển bình thường khi điều kiện tốt lên.",
        "Điều chỉnh dinh dưỡng theo thực trạng đất và cây.",
        "Duy trì nước hợp lý, tránh stress kéo dài.",
        "Theo dõi lá non và khả năng phát triển của bông."
    ],
    ["Rice___Bacterial_leaf_blight", "Rice___Leaf_smut"],
    [
        "Đốm nâu thường xuất hiện thành những đốm "
        "tròn hoặc bầu dục riêng lẻ.",
        "Bạc lá vi khuẩn thường kéo dài từ mép hoặc chóp lá."
    ]
)


_add_plant_disease(
    "Rice___Leaf_smut",
    "Lúa nước (Rice)",
    "Bệnh đốm than lá lúa",
    "Entyloma oryzae",
    "Nấm",
    "Thường nhẹ đến trung bình, tùy điều kiện và giống.",
    [
        "Lá xuất hiện các chấm hoặc đốm nhỏ màu nâu đen.",
        "Một số vết có thể hơi nổi trên bề mặt.",
        "Tổn thương có thể có quầng vàng nhạt.",
        "Dấu hiệu dễ nhầm với các bệnh đốm lá khác."
    ],
    [
        "Sử dụng hạt giống khỏe.",
        "Vệ sinh ruộng và quản lý tàn dư bệnh.",
        "Duy trì dinh dưỡng, nước tưới cân đối.",
        "Theo dõi mức độ bệnh trước khi quyết định "
        "có cần biện pháp can thiệp bổ sung hay không."
    ],
    [
        "Ưu tiên quản lý tổng hợp và duy trì cây khỏe.",
        "Chỉ áp dụng chế phẩm đối kháng nếu có "
        "bằng chứng hiệu quả đối với tác nhân này."
    ],
    [
        "Nếu mức độ bệnh nhẹ, lúa thường vẫn có "
        "khả năng tiếp tục sinh trưởng.",
        "Duy trì nước và dinh dưỡng theo giai đoạn.",
        "Không làm tổn thương lá còn xanh không cần thiết.",
        "Theo dõi lá mới và mức độ lan rộng của vết bệnh."
    ],
    ["Rice___Brown_spot", "Rice___Bacterial_leaf_blight"],
    [
        "Đốm than thường tạo các chấm nhỏ sẫm màu.",
        "Cần quan sát cấu trúc vết bệnh và có thể "
        "kiểm tra mẫu để phân biệt với đốm nâu."
    ]
)


# ============================================================
# G. BO SUNG CAC NHOM BENH DE NHAM LAN
# ============================================================

SIMILARITY_GROUPS.extend([
    {
        "group_name": "Nhóm 5: Bệnh đốm và xoăn lá trên ớt cay",
        "classes": [
            "Chili___Bacterial_spot",
            "Chili___Cercospora_leaf_spot",
            "Chili___Curl_virus"
        ],
        "confusion_reason": (
            "Tổn thương lá ớt có thể biểu hiện đốm nâu, "
            "vàng lá hoặc biến dạng; một ảnh đơn lẻ "
            "không luôn xác định được tác nhân."
        ),
        "differentiation_guide": (
            "- Đốm vi khuẩn: đốm nhỏ sẫm màu, đôi khi có quầng vàng.\n"
            "- Cercospora: đốm tròn, tâm xám sáng, viền nâu.\n"
            "- Xoăn lá: biến dạng phiến lá; cần phân biệt virus với côn trùng."
        )
    },
    {
        "group_name": "Nhóm 6: Bệnh gỉ sắt và Phoma trên cà phê",
        "classes": [
            "Coffee___Rust",
            "Coffee___Phoma"
        ],
        "confusion_reason": (
            "Cả hai đều có thể làm lá vàng, hoại tử "
            "và rụng, gây giảm quang hợp."
        ),
        "differentiation_guide": (
            "- Rust: ổ bào tử dạng bột vàng cam ở mặt dưới lá.\n"
            "- Phoma: đốm hoặc vùng hoại tử nâu, không có ổ bột gỉ sắt."
        )
    },
    {
        "group_name": "Nhóm 7: Bệnh cháy và đốm lá ngô",
        "classes": [
            "Corn___Blight",
            "Corn___Gray_leaf_spot",
            "Corn___Common_rust"
        ],
        "confusion_reason": (
            "Các bệnh đều có thể làm lá ngô đổi màu "
            "và giảm diện tích quang hợp."
        ),
        "differentiation_guide": (
            "- Blight: vết cháy lá lớn hoặc kéo dài.\n"
            "- Gray leaf spot: đốm hình chữ nhật giới hạn bởi gân.\n"
            "- Common rust: ổ bào tử nổi màu nâu đỏ."
        )
    },
    {
        "group_name": "Nhóm 8: Thán thư, đốm lá và khô lá cao su",
        "classes": [
            "Rubber___Anthracnose",
            "Rubber___Leaf_spot",
            "Rubber___Dry_leaf"
        ],
        "confusion_reason": (
            "Đều có thể biểu hiện mô lá nâu khô, "
            "nhưng Dry_leaf chưa phải tên tác nhân cụ thể."
        ),
        "differentiation_guide": (
            "- Anthracnose: có thể gây đốm và biến dạng lá non.\n"
            "- Leaf_spot: nhóm triệu chứng đốm lá do nhiều tác nhân.\n"
            "- Dry_leaf: cần kiểm tra cả nguyên nhân sinh lý và bệnh."
        )
    },
    {
        "group_name": "Nhóm 9: Bệnh và côn trùng gây hại lá xoài",
        "classes": [
            "Mango___Anthracnose",
            "Mango___Bacterial_canker",
            "Mango___Powdery_mildew",
            "Mango___Sooty_mould",
            "Mango___Gall_midge",
            "Mango___Cutting_weevil"
        ],
        "confusion_reason": (
            "Nấm, vi khuẩn và côn trùng đều có thể "
            "gây đổi màu hoặc biến dạng lá xoài."
        ),
        "differentiation_guide": (
            "- Anthracnose: vết hoại tử nâu đen.\n"
            "- Bacterial canker: vết bệnh có thể sần hoặc nứt.\n"
            "- Powdery mildew: lớp bột trắng.\n"
            "- Sooty mould: lớp phủ đen như bồ hóng.\n"
            "- Gall midge: các u sần trên lá.\n"
            "- Cutting weevil: lá non bị cắt hoặc rụng."
        )
    },
    {
        "group_name": "Nhóm 10: Bạc lá, đốm nâu và đốm than lá lúa",
        "classes": [
            "Rice___Bacterial_leaf_blight",
            "Rice___Brown_spot",
            "Rice___Leaf_smut"
        ],
        "confusion_reason": (
            "Đều tạo vùng đổi màu hoặc hoại tử "
            "trên phiến lá lúa."
        ),
        "differentiation_guide": (
            "- Bacterial leaf blight: vùng bạc lá kéo dài từ mép hoặc chóp.\n"
            "- Brown spot: các đốm nâu bầu dục riêng lẻ.\n"
            "- Leaf smut: chấm hoặc đốm nhỏ màu nâu đen."
        )
    }
])


# ============================================================
# H. KIEM TRA 26 LOP MOI
# ============================================================

NEW_CLASSES = [
    "Chili___Bacterial_spot",
    "Chili___Cercospora_leaf_spot",
    "Chili___Curl_virus",
    "Chili___healthy",

    "Coffee___healthy",
    "Coffee___Rust",
    "Coffee___Phoma",

    "Corn___Blight",
    "Corn___Common_rust",
    "Corn___Gray_leaf_spot",
    "Corn___healthy",

    "Rubber___Anthracnose",
    "Rubber___Dry_leaf",
    "Rubber___healthy",
    "Rubber___Leaf_spot",

    "Mango___Anthracnose",
    "Mango___Bacterial_canker",
    "Mango___Cutting_weevil",
    "Mango___Die_back",
    "Mango___Gall_midge",
    "Mango___healthy",
    "Mango___Powdery_mildew",
    "Mango___Sooty_mould",

    "Rice___Bacterial_leaf_blight",
    "Rice___Brown_spot",
    "Rice___Leaf_smut",
]

assert len(NEW_CLASSES) == 26

for _class_name in NEW_CLASSES:
    assert _class_name in DISEASE_DATABASE, _class_name
    assert DISEASE_DATABASE[_class_name]["in_dataset"]

    _entry = DISEASE_DATABASE[_class_name]

    for _field in [
        "crop",
        "vietnamese_name",
        "scientific_name",
        "pathogen_type",
        "danger_level",
        "symptoms",
        "treatment_protocol",
        "post_disease_recovery",
        "confusion_analysis",
    ]:
        assert _field in _entry, (_class_name, _field)

    for _field in [
        "farming_practices",
        "biological_control",
        "chemical_control",
    ]:
        assert _field in _entry["treatment_protocol"]

    assert _entry["post_disease_recovery"]

print(
    "[DISEASE DATABASE] Da bo sung du "
    f"{len(NEW_CLASSES)} lop cua 6 loai cay."
)
