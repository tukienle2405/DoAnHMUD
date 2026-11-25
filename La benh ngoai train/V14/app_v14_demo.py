# -*- coding: utf-8 -*-
"""Independent Streamlit demo. Does not modify old app.py or any checkpoint."""
from __future__ import annotations
import hashlib
import json
import csv
from pathlib import PurePosixPath
from datetime import datetime
from pathlib import Path
import streamlit as st
import torch
from core_v14 import (
    CROP_MODEL, DISEASE_MODEL, CROP_VI, get_models, prepare_image_bytes,
    rank_crops, rank_diseases, crop_label,
)

st.set_page_config(page_title='V14 – Xác nhận cây trước khi xem bệnh',page_icon='🌿',layout='wide')
st.title('🌿 V14 · Nhận diện lá cây có bước xác nhận loại cây')
st.caption('Bản demo độc lập – dùng V3 (18 cây) + V2 (97 lớp); không sửa app.py gốc.')
st.info('Các dự đoán chỉ mang tính tham khảo; không đủ để khẳng định nguyên nhân bệnh hay quyết định xử lý cây. Điểm softmax không phải xác suất đúng đã được hiệu chuẩn.')

@st.cache_resource(show_spinner='Đang tải mô hình V3 và V2…')
def cached_models():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crop_model, names18, disease_model, names97=get_models(device)
    return device,crop_model,names18,disease_model,names97

try:
    device, crop_model, names18, disease_model, names97 = cached_models()
except Exception as e:
    st.error('Không tải được checkpoint. Chạy `py -3 -u kiem_tra_v14.py --mode check` trong PowerShell để xem lỗi.')
    st.exception(e)
    st.stop()

with st.sidebar:
    st.header('Thông tin')
    st.write(f'Thiết bị tính toán: **{device}**')
    st.write(f'Mô hình cây: **{len(names18)} cây**')
    st.write(f'Mô hình bệnh: **{len(names97)} lớp**')
    st.caption('Không huấn luyện, không ghi đè checkpoint, không sửa ảnh gốc.')

source_csv = Path(__file__).resolve().parent.parent / 'V12' / 'KET_QUA_QUET_ANH_V12' / 'DANH_SACH_UNG_VIEN_V12.csv'
source_options = (['Ảnh đã quét từ V12B', 'Tải ảnh mới'] if source_csv.is_file() else ['Tải ảnh mới'])
source_mode = st.radio('Nguồn ảnh', source_options, horizontal=True)

if source_mode == 'Ảnh đã quét từ V12B':
    with source_csv.open(encoding='utf-8-sig',newline='') as f:
        candidates = list(csv.DictReader(f))
    if not candidates:
        st.error('Danh sách V12B đang trống.')
        st.stop()
    idx = st.selectbox('Chọn một trong các ảnh V12B (không cần chép ảnh)',list(range(len(candidates))),
        format_func=lambda i:f"{i+1}. {candidates[i].get('crop','?')} / {candidates[i].get('folder_label_hint','?')} / {Path(candidates[i].get('copied_relative_path','')).name}")
    candidate=candidates[idx]
    relative=candidate.get('copied_relative_path','').replace('\\','/')
    parts=PurePosixPath(relative)
    base=source_csv.parent.resolve()
    image_path=(base/Path(*parts.parts)).resolve()
    if (not relative or parts.is_absolute() or '..' in parts.parts or ':' in relative
            or not image_path.is_relative_to(base) or not image_path.is_file()):
        st.error('Đường dẫn ảnh V12B không hợp lệ. Hãy chọn ảnh khác hoặc tải ảnh mới.')
        st.stop()
    content=image_path.read_bytes()
    image_name=image_path.name
    st.caption('Tên thư mục V12B là gợi ý chưa được xác minh; hãy kiểm tra bằng mắt trước khi xác nhận cây.')
else:
    uploaded=st.file_uploader('Chọn ảnh lá cây (JPG, PNG hoặc WEBP)',type=['jpg','jpeg','png','webp'])
    if uploaded is None:
        st.write('Tải lên một ảnh để bắt đầu. Sau khi xem AI gợi ý, bạn có thể xác nhận một trong 18 cây hoặc chọn “Không thuộc danh sách”.')
        st.stop()
    content=uploaded.getvalue()
    image_name=uploaded.name
if not content or len(content)>18*1024*1024:
    st.error('Ảnh trống hoặc lớn hơn 18 MB. Hãy chọn ảnh khác.')
    st.stop()
image_hash=hashlib.sha256(content).hexdigest()
if st.session_state.get('image_hash_v14')!=image_hash:
    st.session_state['image_hash_v14']=image_hash
    st.session_state['confirmed_v14']=False
    st.session_state['chosen_v14']='-- Chưa xác nhận --'

try:
    pil_image,x=prepare_image_bytes(content)
    with torch.inference_mode():
        out_crop=crop_model(x.to(device))[0].cpu()
        out_disease=disease_model(x.to(device))[0].cpu()
    ranking_crop=rank_crops(out_crop,names18,k=3)
except Exception as e:
    st.error('Không thể đọc ảnh hoặc suy luận trên ảnh này.')
    st.exception(e)
    st.stop()

left,right=st.columns([1,1.35],gap='large')
with left:
    st.image(pil_image,caption='Ảnh vừa tải',use_container_width=True)
    st.subheader('AI gợi ý loại cây')
    for rank,row in enumerate(ranking_crop,1):
        st.write(f"{rank}. **{crop_label(row['crop'])}** · điểm softmax {row['score']:.4f}")
    if ranking_crop[0]['score']<0.65:
        st.warning('Điểm dự đoán loại cây thấp: nên xác nhận cây trước khi sử dụng phần dự đoán bệnh.')

with right:
    st.subheader('Xác nhận loại cây')
    options=['-- Chưa xác nhận --'] + sorted(names18,key=lambda x:CROP_VI.get(x,x)) + ['-- Không thuộc 18 cây --']
    def reset_confirmation():
        st.session_state['confirmed_v14']=False
    chosen=st.selectbox('Bạn biết ảnh thuộc cây nào?',options,format_func=lambda v:crop_label(v) if v in names18 else v,key='chosen_v14',on_change=reset_confirmation)
    confirmed=st.checkbox('Tôi đã kiểm tra và xác nhận loại cây mình chọn',key='confirmed_v14')
    automatic_crop=ranking_crop[0]['crop']
    if chosen=='-- Không thuộc 18 cây --' and confirmed:
        st.warning('Ảnh nằm ngoài 18 cây hỗ trợ. Không hiển thị bệnh như một kết luận.')
    elif confirmed and chosen in names18:
        st.success('Đã xác nhận loại cây để giới hạn các bệnh được dự đoán.')
        if chosen!=automatic_crop:
            st.warning('AI đoán '+crop_label(automatic_crop)+' nhưng bạn chọn '+crop_label(chosen)+'. Kết quả dưới đây theo cây do bạn xác nhận.')
    else:
        st.warning('Chưa xác nhận loại cây; các dự đoán bệnh bên dưới chỉ là gợi ý tự động.')

    automatic_diseases=rank_diseases(out_disease,names97,automatic_crop,3)
    if confirmed and chosen=='-- Không thuộc 18 cây --':
        st.info('Đã xác nhận ngoài phạm vi; tạm ẩn dự đoán bệnh tự động để tránh gây hiểu nhầm.')
    else:
        st.subheader('Bệnh theo cây AI tự đoán')
        for item in automatic_diseases:
            st.write(f"- `{item['label']}` · điểm tương đối {item['relative_score']:.4f}")

    selected_diseases=None
    if confirmed and chosen in names18:
        selected_diseases=rank_diseases(out_disease,names97,chosen,3)
        st.subheader('Bệnh theo cây đã xác nhận')
        for item in selected_diseases:
            st.write(f"- `{item['label']}` · điểm tương đối {item['relative_score']:.4f}")
        if chosen==automatic_crop:
            st.caption('Loại cây bạn xác nhận trùng với gợi ý AI, nên danh sách bệnh cũng trùng.')
    st.caption('Các điểm số chỉ so sánh các nhãn trong cùng một cây và chưa được hiệu chuẩn. Ảnh có thể có nhiều vấn đề hoặc nằm ngoài 97 nhãn hỗ trợ.')

st.divider()
if st.button('Lưu báo cáo thử nghiệm V14',type='primary'):
    output_dir=Path(__file__).resolve().parent/'KET_QUA_V14'
    output_dir.mkdir(parents=True,exist_ok=True)
    payload={
        'mode':'V14_DEMO_READ_ONLY','time_local':datetime.now().astimezone().isoformat(),
        'filename':image_name,'sha256_image':image_hash,
        'crop_suggested_v3':ranking_crop,'crop_confirmed_by_user':(chosen if confirmed and chosen in names18 else None),
        'user_selected_out_of_scope':bool(confirmed and chosen=='-- Không thuộc 18 cây --'),
        'disease_v2_on_auto_crop':(None if confirmed and chosen=='-- Không thuộc 18 cây --' else automatic_diseases),'disease_v2_on_confirmed_crop':selected_diseases,
        'labels_verified_by_external_expert':False,'independent_test':False,'accuracy_calculated':False,
        'warning':'Day la demo, nhan benh chua duoc xac minh; score khong phai xac suat chan doan.',
    }
    dest=output_dir/'KET_QUA_DEMO_V14.json'
    dest.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    st.success('Đã lưu báo cáo tại: '+str(dest))
    st.download_button('Tải báo cáo JSON',data=json.dumps(payload,ensure_ascii=False,indent=2),file_name='KET_QUA_DEMO_V14.json',mime='application/json')
