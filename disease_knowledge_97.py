
# -*- coding: utf-8 -*-
"""
CAM NANG BENH LA CAY - 97 LOP

- Ke thua co so tri thuc 41 lop cu
- Bo sung 56 lop cua 9 loai cay moi
- Khong thay doi model hoac du lieu Train/Test
- Khong tu suy doan loai tac nhan khi nhan chua ro
"""

import json
from pathlib import Path

from disease_knowledge import (
    DISEASE_DATABASE as OLD_DATABASE,
    SIMILARITY_GROUPS as OLD_GROUPS,
)

ROOT = Path(__file__).resolve().parent

DISEASE_DATABASE = dict(OLD_DATABASE)
SIMILARITY_GROUPS = list(OLD_GROUPS)


# Moi dong:
# class | ten Viet | nhom tac nhan | tac nhan tham khao
# | trieu chung 1 | dau hieu phan biet

NEW_DATA = """
Banana___Cordana|Chuối - Bệnh đốm lá Cordana|fungus|Neocordana/Cordana spp.|Lá có đốm nâu sẫm, có thể xuất hiện quầng vàng quanh vết bệnh.|Cần phân biệt với những vệt đốm kéo dài của Sigatoka.
Banana___Sigatoka|Chuối - Bệnh Sigatoka|fungus|Pseudocercospora spp.|Lá có các vệt hoặc đốm nâu kéo dài, nhiều vết có thể liên kết thành vùng cháy lá.|Phân biệt với Cordana bằng hình dạng và sự phân bố vết bệnh.
Banana___healthy|Chuối - Lá khỏe mạnh|healthy|Không xác định tác nhân gây bệnh.|Lá không có tổn thương bệnh lý rõ ràng trong phạm vi quan sát.|Lá khỏe vẫn có thể có rách cơ học hoặc thay đổi màu theo tuổi.
Cashew___Fungal_infection|Điều - Nghi nhiễm nấm lá|fungus|Chưa xác định loài nấm từ nhãn dữ liệu.|Lá có thể xuất hiện đốm nâu, vàng hoặc vùng hoại tử.|Cần kiểm tra vết bệnh và tác nhân trước khi kết luận nhiễm nấm.
Cashew___healthy|Điều - Lá khỏe mạnh|healthy|Không xác định tác nhân gây bệnh.|Lá phát triển tương đối đều, không có vùng tổn thương nổi bật.|Đối chiếu với lá cùng tuổi trên cây và điều kiện sinh trưởng.
Durian___Algal|Sầu riêng - Đốm tảo|alga|Cephaleuros spp. (tham khảo).|Lá có các đốm tròn màu vàng cam đến nâu đỏ, đôi khi bề mặt hơi thô.|Đốm tảo có thể khác vết hoại tử nâu lan rộng do nấm.
Durian___Blight|Sầu riêng - Cháy lá|symptom|Chưa xác định tác nhân cụ thể từ nhãn Blight.|Một phần lá chuyển nâu, khô hoặc cháy lan thành mảng.|Cháy lá là biểu hiện rộng, cần phân biệt với Rhizoctonia và Colletotrichum.
Durian___Colletotrichum|Sầu riêng - Bệnh do Colletotrichum|fungus|Colletotrichum spp.|Lá có đốm hoại tử màu nâu, vết bệnh có thể lan rộng và kết hợp.|Cần đối chiếu hình dạng đốm và kiểm tra mẫu để phân biệt với các nấm khác.
Durian___Phomopsis|Sầu riêng - Bệnh do Phomopsis|fungus|Phomopsis/Diaporthe spp. (tham khảo).|Lá có đốm hoặc vùng hoại tử, đôi khi xuất hiện nhiều vết trên cùng lá.|Không thể xác nhận Phomopsis chỉ dựa trên ảnh lá.
Durian___Rhizoctonia|Sầu riêng - Bệnh do Rhizoctonia|fungus|Rhizoctonia spp.|Lá có những vùng cháy nâu không đều, đôi khi lan thành mảng lớn.|Cần phân biệt với lá già và cháy lá do nguyên nhân khác.
Durian___healthy|Sầu riêng - Lá khỏe mạnh|healthy|Không xác định tác nhân gây bệnh.|Lá có màu sắc phù hợp tuổi lá, không có vùng hoại tử bệnh lý rõ.|Lá khỏe có thể khác nhau về màu, độ bóng và hình dạng tùy tuổi.
Guava___Anthracnose|Ổi - Bệnh thán thư|fungus|Colletotrichum spp.|Lá có đốm nâu sẫm hoặc các vùng hoại tử không đều.|Vết thán thư có thể giống nhiều bệnh đốm lá, cần kiểm tra thêm.
Guava___Canker|Ổi - Bệnh loét|symptom|Chưa xác định tác nhân từ nhãn Canker.|Mô lá hoặc bộ phận cây có thể xuất hiện vết tổn thương sẫm màu.|Cần phân biệt tổn thương do vi sinh vật với tổn thương cơ học.
Guava___Dot|Ổi - Dạng đốm lá Dot|symptom|Nhãn Dot chưa đủ xác định tác nhân.|Lá có các đốm nhỏ, màu sắc và mật độ có thể khác nhau.|Không nên coi Dot là tên bệnh đã xác định về mặt khoa học.
Guava___Rust|Ổi - Bệnh gỉ sắt|fungus|Nhóm nấm gỉ sắt; cần xác minh loài.|Lá có thể xuất hiện các đốm vàng cam hoặc nâu và vùng hoại tử.|Kiểm tra mặt dưới lá để phân biệt ổ bào tử với đốm lá khác.
Guava___healthy|Ổi - Lá khỏe mạnh|healthy|Không xác định tác nhân gây bệnh.|Lá không có triệu chứng bệnh nổi bật.|Cần so sánh với lá cùng tuổi và điều kiện ánh sáng.
Jackfruit___Algal|Mít - Đốm tảo|alga|Cephaleuros spp. (tham khảo).|Lá có đốm vàng cam hoặc nâu, một số vết có bề mặt khác mô lá bình thường.|Phân biệt với Anthracnose bằng hình thái và cấu trúc bề mặt vết bệnh.
Jackfruit___Anthracnose|Mít - Bệnh thán thư|fungus|Colletotrichum spp.|Lá xuất hiện đốm nâu, vết bệnh có thể mở rộng thành mảng hoại tử.|Dễ nhầm với đốm tảo và một số bệnh đốm lá khác.
Jackfruit___Leaf_spot|Mít - Bệnh đốm lá|symptom|Nhiều tác nhân có thể gây đốm lá.|Lá có các vết đốm riêng rẽ, màu nâu hoặc xám.|Tên Leaf spot mô tả triệu chứng chứ chưa xác định loài gây bệnh.
Jackfruit___Phyllosticta|Mít - Bệnh đốm lá Phyllosticta|fungus|Phyllosticta spp. (tham khảo).|Lá có đốm hoại tử màu nâu, đôi khi tâm vết nhạt hơn viền.|Dễ nhầm với thán thư; cần kiểm tra cấu trúc sinh sản của nấm.
Jackfruit___Sooty_mould|Mít - Nấm bồ hóng|fungus|Nhóm nấm bồ hóng phát triển trên bề mặt.|Bề mặt lá có lớp phủ màu đen như muội than.|Phân biệt lớp nấm phủ bề mặt với mô lá bị hoại tử.
Jackfruit___healthy|Mít - Lá khỏe mạnh|healthy|Không xác định tác nhân gây bệnh.|Lá không có vết bệnh rõ rệt, màu sắc phù hợp tuổi lá.|Cần phân biệt biến đổi tự nhiên với vùng hoại tử bất thường.
Lemon___Anthracnose|Chanh - Bệnh thán thư|fungus|Colletotrichum spp.|Lá có đốm nâu hoặc vùng mô khô hoại tử.|Dễ nhầm với khô lá do stress hoặc bệnh khác.
Lemon___Bacterial_blight|Chanh - Nghi cháy lá vi khuẩn|bacterium|Chưa xác định loài vi khuẩn từ nhãn.|Lá có các vùng tổn thương lan rộng, có thể chuyển vàng hoặc nâu.|Cần giám định trước khi kết luận vi khuẩn là nguyên nhân.
Lemon___Citrus_canker|Chanh - Bệnh loét cây có múi|bacterium|Xanthomonas citri (tham khảo).|Lá có những vết loét nổi hoặc sần, đôi khi kèm quầng vàng.|Phân biệt vết loét sần với các đốm phẳng do tác nhân khác.
Lemon___Curl_virus|Chanh - Xoăn lá nghi do virus|virus|Chưa xác định virus từ nhãn dữ liệu.|Lá cong, xoăn hoặc biến dạng, đôi khi có biến đổi màu.|Xoăn lá còn có thể do côn trùng hoặc rối loạn sinh lý.
Lemon___Deficiency_leaf|Chanh - Dấu hiệu thiếu dinh dưỡng|nutrient|Rối loạn dinh dưỡng; chưa xác định nguyên tố.|Lá vàng không đều hoặc xuất hiện các vùng mất màu.|Cần kết hợp tuổi lá, đất và dinh dưỡng để phân biệt với bệnh.
Lemon___Dry_leaf|Chanh - Hiện tượng khô lá|symptom|Chưa xác định; có thể do stress hoặc bệnh.|Mép hoặc phiến lá chuyển nâu khô, đôi khi lá cong và rụng.|Không thể khẳng định tác nhân chỉ từ triệu chứng khô lá.
Lemon___Sooty_mould|Chanh - Nấm bồ hóng|fungus|Nhóm nấm bồ hóng trên bề mặt.|Lá có lớp phủ đen có thể che khuất bề mặt xanh.|Kiểm tra côn trùng tiết mật và phân biệt với đốm hoại tử.
Lemon___Spider_mites|Chanh - Nhện hại lá|pest|Nhóm nhện hại; chưa xác định loài.|Lá xuất hiện các chấm vàng hoặc bạc li ti, có thể khô khi bị hại nặng.|Quan sát mặt dưới lá để tìm nhện và mạng tơ.
Lemon___healthy|Chanh - Lá khỏe mạnh|healthy|Không xác định tác nhân gây bệnh.|Lá phát triển bình thường và không có vùng tổn thương rõ.|Lá non và lá già có thể khác nhau về màu sắc.
Papaya___Anthracnose|Đu đủ - Bệnh thán thư|fungus|Colletotrichum spp.|Lá có đốm hoại tử nâu, có thể lan rộng hoặc liên kết.|Nhãn dữ liệu Đu đủ cần xác minh trước khi dùng để kết luận.
Papaya___Bacterial_spot|Đu đủ - Nghi đốm vi khuẩn|bacterium|Chưa xác định loài vi khuẩn từ nhãn.|Lá có đốm nhỏ sẫm màu, có thể kèm vàng mô xung quanh.|Triệu chứng có thể giống đốm lá do các tác nhân khác.
Papaya___Curl|Đu đủ - Xoăn lá|symptom|Chưa xác định tác nhân gây xoăn lá.|Phiến lá cong, xoắn hoặc biến dạng so với lá bình thường.|Có thể liên quan virus, nhện hại hoặc yếu tố môi trường.
Papaya___Mealybug|Đu đủ - Rệp sáp hại lá|pest|Nhóm rệp sáp Pseudococcidae.|Lá hoặc cuống có thể xuất hiện cụm rệp trắng dạng bông.|Cần quan sát trực tiếp côn trùng, không chỉ dựa vào lá vàng.
Papaya___Mite_damage|Đu đủ - Nhện hại|pest|Nhóm nhện hại; chưa xác định loài.|Lá có chấm mất màu, biến dạng hoặc khô từng vùng.|Kiểm tra mặt dưới lá và phân biệt với thiếu dinh dưỡng.
Papaya___Mosaic|Đu đủ - Triệu chứng khảm lá|virus|Chưa xác định virus từ nhãn Mosaic.|Lá có các mảng xanh đậm và xanh nhạt không đều.|Khảm lá không đủ để xác định chính xác loại virus.
Papaya___Ringspot|Đu đủ - Triệu chứng đốm vòng|virus|Papaya ringspot virus là tác nhân tham khảo cần xác minh.|Lá có thể biến dạng, khảm hoặc biểu hiện đốm vòng trên cây.|Cần xác minh bệnh bằng triệu chứng trên nhiều bộ phận và xét nghiệm phù hợp.
Papaya___healthy|Đu đủ - Lá khỏe mạnh|healthy|Không xác định tác nhân gây bệnh.|Lá không có biểu hiện bất thường rõ trong ảnh.|Lá bình thường vẫn có biến thiên do tuổi và môi trường.
Sugarcane___Banded_chlorosis|Mía - Vàng lá thành dải|symptom|Chưa xác định nguyên nhân cụ thể.|Lá xuất hiện những dải màu nhạt hoặc vàng xen vùng xanh.|Cần phân biệt với thiếu dinh dưỡng và một số bệnh gây sọc lá.
Sugarcane___Brown_rust|Mía - Bệnh gỉ sắt nâu|fungus|Puccinia melanocephala (tham khảo).|Trên lá có các ổ bệnh màu nâu gỉ sắt, có thể nổi trên bề mặt.|Cần phân biệt ổ bào tử gỉ sắt với đốm nâu phẳng.
Sugarcane___Brown_spot|Mía - Bệnh đốm nâu|fungus|Tác nhân nấm cần được xác minh.|Lá có nhiều đốm màu nâu, có thể mở rộng khi bệnh tiến triển.|Phân biệt với gỉ sắt bằng việc kiểm tra ổ bào tử nổi.
Sugarcane___Dried_leaves|Mía - Hiện tượng khô lá|symptom|Có thể liên quan stress hoặc bệnh.|Lá chuyển vàng rồi khô nâu ở mép, chóp hoặc toàn phiến.|Lá khô không tự chứng minh cây nhiễm nấm.
Sugarcane___Grassy_shoot|Mía - Bệnh chồi cỏ|other|Nhóm phytoplasma là tác nhân tham khảo.|Cây có thể phát sinh nhiều chồi nhỏ, lá hẹp và phát triển bất thường.|Nên quan sát toàn cây vì ảnh lá đơn lẻ thường không đủ.
Sugarcane___Pokkah_boeng|Mía - Bệnh Pokkah Boeng|fungus|Fusarium spp. (tham khảo).|Lá non có thể xoắn, nhăn hoặc xuất hiện các vùng vàng và hoại tử.|Cần quan sát lá non và đỉnh sinh trưởng để phân biệt.
Sugarcane___Sett_rot|Mía - Thối hom giống|symptom|Nhiều tác nhân có thể gây thối hom.|Hom giống có thể bị biến màu, thối hoặc mọc mầm yếu.|Ảnh lá không đủ xác nhận bệnh thối hom; cần kiểm tra hom và gốc.
Sugarcane___Smut|Mía - Bệnh than|fungus|Sporisorium scitamineum (tham khảo).|Cây bệnh có thể phát triển chồi bất thường và cấu trúc dạng roi đen.|Cần quan sát chồi ngọn, không thể xác định từ mọi ảnh lá.
Sugarcane___Viral_disease|Mía - Triệu chứng nghi do virus|virus|Chưa xác định virus từ nhãn.|Lá có thể có sọc, khảm hoặc vàng bất thường.|Cần phân biệt với thiếu dinh dưỡng và các nguyên nhân khác.
Sugarcane___Yellow_leaf|Mía - Vàng lá|symptom|Có thể liên quan virus, dinh dưỡng hoặc stress.|Lá vàng dọc gân hoặc vàng từng vùng, tùy nguyên nhân.|Không nên mặc định mọi trường hợp vàng lá đều do virus.
Sugarcane___healthy|Mía - Lá khỏe mạnh|healthy|Không xác định tác nhân gây bệnh.|Lá có sắc xanh phù hợp, không có vùng bệnh lý nổi bật.|Cần đánh giá đồng thời lá non và lá già.
Tea___Blister_blight|Chè - Bệnh phồng lá|fungus|Exobasidium vexans.|Lá non có đốm nhạt và vùng phồng hoặc biến dạng.|Phân biệt với đốm nâu và các tổn thương do côn trùng.
Tea___Brown_blight|Chè - Bệnh cháy lá nâu|fungus|Nhóm nấm gây cháy lá nâu; cần xác minh loài.|Lá có những vùng nâu khô, đôi khi viền sẫm và tâm nhạt.|Dễ nhầm với cháy lá do stress hoặc các nấm khác.
Tea___Red_rust|Chè - Gỉ đỏ do tảo|alga|Cephaleuros spp.|Lá có những đốm vàng cam hoặc đỏ gỉ, bề mặt có thể hơi thô.|Gỉ đỏ do tảo khác với bệnh gỉ sắt do nấm.
Tea___Red_spider_mite|Chè - Nhện đỏ|pest|Nhóm nhện đỏ hại chè.|Lá có chấm mất màu, bạc hoặc vàng; hại nặng có thể khô lá.|Kiểm tra mặt dưới lá để tìm nhện hại.
Tea___Tea_mosquito_bug|Chè - Bọ xít muỗi|pest|Helopeltis spp. (tham khảo).|Chồi và lá non có các chấm tổn thương nhỏ do chích hút.|Quan sát đọt non và côn trùng để phân biệt với đốm nấm.
Tea___healthy|Chè - Lá khỏe mạnh|healthy|Không xác định tác nhân gây bệnh.|Lá không có đốm bệnh hoặc dấu hiệu hại rõ.|Lá chè non và trưởng thành có khác biệt tự nhiên về màu sắc.
"""


# ======================================================
# HUONG DAN CHUNG THEO NHOM TAC NHAN
# ======================================================

GUIDES = {
    "fungus": {
        "condition": [
            "Theo dõi ẩm độ, thời gian lá bị ướt và "
            "mật độ tán cây.",
            "Không phải mọi đốm nâu đều do nấm."
        ],
        "farming": [
            "Giữ tán cây thông thoáng và chăm sóc cân đối.",
            "Theo dõi tốc độ lan rộng của vết bệnh.",
            "Quản lý lá và tàn dư bệnh theo hướng dẫn "
            "canh tác phù hợp."
        ],
        "biological": [
            "Chỉ cân nhắc biện pháp sinh học đã được "
            "đánh giá phù hợp với tác nhân."
        ]
    },
    "bacterium": {
        "condition": [
            "Theo dõi thời tiết ẩm và tổn thương trên lá.",
            "Cần xác minh tác nhân bằng phương pháp phù hợp."
        ],
        "farming": [
            "Hạn chế làm tổn thương lá khi chăm sóc.",
            "Tránh làm lây lan vật liệu cây bị bệnh.",
            "Duy trì thông thoáng và vệ sinh dụng cụ."
        ],
        "biological": [
            "Chỉ sử dụng biện pháp đối kháng có bằng chứng "
            "phù hợp với bệnh đang gặp."
        ]
    },
    "virus": {
        "condition": [
            "Theo dõi sự xuất hiện của côn trùng truyền bệnh.",
            "Triệu chứng khảm hoặc xoăn không đủ để xác định virus."
        ],
        "farming": [
            "Theo dõi cây có triệu chứng và cây xung quanh.",
            "Quản lý côn trùng truyền bệnh theo hướng tổng hợp.",
            "Sử dụng vật liệu giống có nguồn gốc đáng tin cậy."
        ],
        "biological": [
            "Ưu tiên quản lý dịch hại tổng hợp và "
            "bảo vệ các loài thiên địch."
        ]
    },
    "pest": {
        "condition": [
            "Quan sát mặt dưới lá, chồi non và quần thể côn trùng.",
            "Mức độ hại thay đổi theo điều kiện thời tiết."
        ],
        "farming": [
            "Kiểm tra trực tiếp sự hiện diện của sinh vật hại.",
            "Theo dõi tỷ lệ lá bị hại trước khi can thiệp.",
            "Giữ vệ sinh vườn và quản lý cây trồng hợp lý."
        ],
        "biological": [
            "Ưu tiên bảo vệ thiên địch và các biện pháp "
            "quản lý dịch hại tổng hợp."
        ]
    },
    "alga": {
        "condition": [
            "Theo dõi điều kiện ẩm và tán lá rậm.",
            "Cần kiểm tra hình thái đốm để phân biệt với nấm."
        ],
        "farming": [
            "Giữ thông thoáng và quản lý ẩm độ thích hợp.",
            "Quan sát các đốm mới xuất hiện trên lá.",
            "Duy trì chăm sóc cây khỏe."
        ],
        "biological": [
            "Chưa đưa ra biện pháp sinh học đặc hiệu "
            "khi chưa xác minh tác nhân."
        ]
    },
    "nutrient": {
        "condition": [
            "Triệu chứng phụ thuộc dinh dưỡng, pH đất, "
            "nước tưới và hoạt động của bộ rễ."
        ],
        "farming": [
            "Kiểm tra tình trạng đất và dinh dưỡng.",
            "Đối chiếu biểu hiện trên lá già và lá non.",
            "Không tự bổ sung quá mức một nguyên tố "
            "khi chưa biết nguyên nhân."
        ],
        "biological": [
            "Không mặc định cần sử dụng chế phẩm "
            "diệt nấm hay vi khuẩn."
        ]
    },
    "symptom": {
        "condition": [
            "Đây có thể là biểu hiện của nhiều nguyên nhân.",
            "Cần quan sát toàn cây và điều kiện sinh trưởng."
        ],
        "farming": [
            "Theo dõi triệu chứng trên lá và bộ phận khác.",
            "Kiểm tra nước, đất, dinh dưỡng và sinh vật hại.",
            "Chỉ lựa chọn cách xử lý đặc hiệu sau "
            "khi xác định nguyên nhân."
        ],
        "biological": [
            "Không sử dụng biện pháp sinh học đặc hiệu "
            "chỉ dựa vào tên triệu chứng."
        ]
    },
    "other": {
        "condition": [
            "Cần kiểm tra cây và bộ phận có triệu chứng.",
            "Tên bệnh trong dữ liệu chưa thay thế giám định."
        ],
        "farming": [
            "Sử dụng giống khỏe và nguồn giống tin cậy.",
            "Quan sát sự phát triển bất thường của cây.",
            "Theo dõi khả năng lan rộng trong ruộng."
        ],
        "biological": [
            "Chỉ dùng biện pháp đặc hiệu khi "
            "đã xác định nhóm tác nhân."
        ]
    },
    "healthy": {
        "condition": [
            "Màu sắc và hình dạng lá thay đổi theo tuổi, "
            "giống và điều kiện môi trường."
        ],
        "farming": [
            "Duy trì nước và dinh dưỡng phù hợp.",
            "Theo dõi lá mới và các dấu hiệu bất thường.",
            "Không cần xử lý bệnh khi chưa có "
            "bằng chứng cây bị bệnh."
        ],
        "biological": [
            "Duy trì cân bằng sinh thái và bảo vệ thiên địch."
        ]
    }
}


CROPS = {
    "Banana": "Chuối",
    "Cashew": "Điều",
    "Durian": "Sầu riêng",
    "Guava": "Ổi",
    "Jackfruit": "Mít",
    "Lemon": "Chanh",
    "Papaya": "Đu đủ",
    "Sugarcane": "Mía",
    "Tea": "Chè"
}


SIMILAR_PAIRS = {
    "Durian___Rhizoctonia": [
        "Durian___Blight",
        "Durian___healthy",
        "Durian___Colletotrichum"
    ],
    "Durian___healthy": [
        "Durian___Rhizoctonia",
        "Durian___Blight"
    ],
    "Jackfruit___Algal": [
        "Jackfruit___Anthracnose",
        "Jackfruit___Leaf_spot"
    ],
    "Jackfruit___Anthracnose": [
        "Jackfruit___Algal",
        "Jackfruit___Phyllosticta"
    ],
    "Guava___Dot": [
        "Guava___Rust",
        "Guava___Anthracnose"
    ],
    "Papaya___Curl": [
        "Papaya___Mosaic",
        "Papaya___Mite_damage"
    ],
    "Tea___Red_rust": [
        "Tea___Brown_blight",
        "Tea___Blister_blight"
    ],
    "Sugarcane___Brown_rust": [
        "Sugarcane___Brown_spot"
    ],
    "Sugarcane___Yellow_leaf": [
        "Sugarcane___Dried_leaves",
        "Sugarcane___Viral_disease"
    ]
}


def make_entry(key, title, kind, cause, symptom, distinction):
    crop_code = key.split("___", 1)[0]
    crop_name = CROPS[crop_code]
    guide = GUIDES[kind]

    healthy = (kind == "healthy")

    if healthy:
        recovery = [
            "Tiếp tục chăm sóc theo nhu cầu của cây.",
            "Theo dõi sự xuất hiện triệu chứng mới."
        ]
        danger = "Chưa thấy dấu hiệu bệnh rõ trong nhãn."
    else:
        recovery = [
            "Mô lá đã chết thường không xanh trở lại.",
            "Duy trì nước và dinh dưỡng phù hợp.",
            "Theo dõi các lá và chồi mới.",
            "Nếu triệu chứng lan rộng, cần kiểm tra "
            "thêm trước khi chọn biện pháp xử lý."
        ]
        danger = (
            "Chưa có đánh giá định lượng mức độ "
            "nguy hại từ một ảnh."
        )

    return {
        "crop": crop_name,
        "in_dataset": True,
        "vietnamese_name": title,
        "scientific_name": cause,
        "pathogen_type": cause,
        "danger_level": danger,
        "symptoms": [
            symptom,
            distinction,
            "Một ảnh đơn lẻ không đủ để "
            "xác nhận chính xác tác nhân."
        ],
        "favorable_conditions": guide["condition"],
        "treatment_protocol": {
            "farming_practices": guide["farming"],
            "biological_control": guide["biological"],
            "chemical_control": [
                "Không đề xuất thuốc bảo vệ thực vật "
                "khi chưa xác minh bệnh và điều kiện "
                "sử dụng phù hợp."
            ]
        },
        "post_disease_recovery": recovery,
        "confusion_analysis": {
            "similar_diseases": SIMILAR_PAIRS.get(key, []),
            "common_features": (
                "Có thể có triệu chứng tương tự "
                "với các bệnh hoặc stress khác."
            ),
            "key_distinctions": [distinction]
        },
        "knowledge_status": (
            "Thông tin tham khảo theo tên nhãn. "
            "Chưa giám định mẫu thực tế."
        )
    }


# ======================================================
# TAO 56 HO SO MOI
# ======================================================

NEW_KEYS = []

for raw_line in NEW_DATA.strip().splitlines():
    parts = [part.strip() for part in raw_line.split("|")]

    if len(parts) != 6:
        raise ValueError(
            "Dong du lieu sai dinh dang: " + raw_line
        )

    key, title, kind, cause, symptom, distinction = parts

    if kind not in GUIDES:
        raise ValueError(
            "Nhom tac nhan khong hop le: " + kind
        )

    if key in DISEASE_DATABASE:
        raise ValueError(
            "Nhãn mới trùng với cơ sở tri thức cũ: " + key
        )

    DISEASE_DATABASE[key] = make_entry(
        key, title, kind, cause, symptom, distinction
    )

    NEW_KEYS.append(key)


if len(NEW_KEYS) != 56:
    raise ValueError(
        f"Can bo sung 56 lop, hien co {len(NEW_KEYS)}"
    )

if len(set(NEW_KEYS)) != 56:
    raise ValueError("Co nhan moi bi trung.")


# ======================================================
# NHOM BENH DE NHAM TU KET QUA TEST THUC TE
# ======================================================

SIMILARITY_GROUPS.extend([
    {
        "group_name": (
            "Sầu riêng: Lá khỏe mạnh và Rhizoctonia"
        ),
        "classes": [
            "Durian___healthy",
            "Durian___Rhizoctonia"
        ],
        "confusion_reason": (
            "Trong tap Test noi bo, 23 anh "
            "Durian healthy bi du doan thanh Rhizoctonia."
        ),
        "differentiation_guide": (
            "Quan sat nhieu la tren cung cay va "
            "qua trinh tien trien vet benh. "
            "Khong ket luan nhiem Rhizoctonia "
            "chi tu mot anh."
        )
    },
    {
        "group_name": (
            "Mít: Đốm tảo và thán thư"
        ),
        "classes": [
            "Jackfruit___Algal",
            "Jackfruit___Anthracnose"
        ],
        "confusion_reason": (
            "Trong tap Test noi bo, 16 anh "
            "Jackfruit Algal bi du doan "
            "thanh Jackfruit Anthracnose."
        ),
        "differentiation_guide": (
            "Doi chieu mau sac, be mat va "
            "hinh dang vet dom. "
            "Kiem tra them khi hai benh "
            "co trieu chung tuong tu."
        )
    }
])


# ======================================================
# KIEM TRA KHOP VOI 97 NHAN CUA MODEL
# ======================================================

CLASS_FILE = (
    ROOT / "training_runs"
    / "plant97_full_20261118_064241"
    / "class_names.json"
)

if CLASS_FILE.is_file():
    class_names = json.loads(
        CLASS_FILE.read_text(encoding="utf-8")
    )

    missing = [
        name for name in class_names
        if name not in DISEASE_DATABASE
    ]

    if missing:
        raise ValueError(
            "Con thieu ho so benh cho nhan: "
            + ", ".join(missing)
        )

    print(
        "[DISEASE KNOWLEDGE 97] "
        f"Da bao phu {len(class_names)}/"
        f"{len(class_names)} nhan."
    )

print(
    "[DISEASE KNOWLEDGE 97] "
    f"Da bo sung {len(NEW_KEYS)} ho so moi."
)
