import streamlit as st
import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.title("OCR Application")
st.write("Upload an image to extract text using EasyOCR")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(img)

   
    for detection in result:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        text = detection[1]
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 5)
        img = cv2.putText(img, text, top_left, font, 0.5, (0,0,0), 2, cv2.FONT_HERSHEY_PLAIN)
    
   
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   
    st.image(img_rgb, caption='Processed Image', use_column_width=True)

   
    st.write("Detected text:")
    for detection in result:
        st.write(f"{detection[1]} (Confidence: {detection[2]:.2f})")
     