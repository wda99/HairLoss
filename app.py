import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('C:/Users/dnehd/Desktop/my_dense.h5')

st.title("탈모진행 상태측정")
file = st.file_uploader("이미지를 업로드해주세요.", type=['jpg', 'png'])

if file is None:
    st.text("이미지를 먼저 업로드해주세요.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    img_resized = image.resize((224, 224), Image.ANTIALIAS)
    img_resized = img_resized.convert("RGB")
    img_resized = np.asarray(img_resized)
    img_resized = img_resized.astype('float32') / 255.0  # 전처리: 스케일링
    img_resized = np.expand_dims(img_resized, axis=0)  # 전처리: 차원 추가

    pred = model.predict(img_resized)
    # 예측 결과 해석 및 출력
    class_index = np.argmax(pred)
    confidence = pred[0, class_index]
    if class_index == 0:
        class_name = "양호"
        result = f"진단결과 당신의 두피상태는 {class_name}입니다. \n\n지금처럼 꾸준히 관리하세요. \n\n정확도 : {confidence * 100:.2f}%"
    elif class_index == 1:
        class_name = "경증"
        result = f"진단결과 당신의 두피상태는 {class_name}입니다. \n\n두피건강에 신경써주세요. \n\n정확도 : {confidence * 100:.2f}%"
    elif class_index == 2:
        class_name = "중등도"
        result = f"진단결과 당신의 두피상태는 {class_name}입니다. \n\n전문의와 상담을 권고드립니다. \n\n정확도 : {confidence * 100:.2f}%"
    else :
        class_name = '중증'
        result = f"진단결과 당신의 두피상태는 {class_name}입니다. \n\n전문의와 상담을 권고드리며 집중적인 관리가 필요합니다. \n\n정확도 : {confidence * 100:.2f}%"
    st.markdown(st.success(result))