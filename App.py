import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from utils import PrecisionMetric, RecallMetric, IoUMetric, get_masked_image

if "model" not in st.session_state:
    st.session_state.data_list = []
    st.session_state.data_list.append(
        tf.keras.models.load_model(
            "./model/unet_model.keras",
            custom_objects={
                "PrecisionMetric": PrecisionMetric,
                "RecallMetric": RecallMetric,
                "IoUMetric": IoUMetric,
            },
        )
    )
    print("Model loaded")
    st.session_state["model"] = False

st.title("Brain Tumor Segmentation 	:stethoscope:")

model = st.session_state.data_list[0]

uploaded_file = st.file_uploader("Please choose the CT scan images :arrow_up:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (256, 256))
        image = image.reshape((1, 256, 256, 3))
        image = tf.cast(image, tf.float32) / 255.0
        out_image = model.predict(image)
        out_image[out_image >= 0.5] = 255
        out_image[out_image < 0.5] = 0
        image = get_masked_image(image[0].numpy(), out_image[0], (0, 0, 255))
        out_image = out_image / 255
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Image with mask")
            st.image(image)

        with col2:
            st.subheader("Predicted Mask")
            st.image(out_image[0])
    except:
        st.error("Invalid File. Please use suitable file. System supports files upto 256x256 resolution")
