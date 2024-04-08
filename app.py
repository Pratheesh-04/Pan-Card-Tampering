import streamlit as st
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import numpy as np

st.title('Pan Card Tampering Detection')

# Function to preprocess images
def preprocess_images(original_img, tampered_img):
    original_img = cv2.resize(original_img,(250, 160))
    tampered_img = tampered_img.resize((250,160))
    return original_img, tampered_img

# Function to compute SSIM and detect tampering
def detect_tampering(original, tampered):
    original_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    tampered_cv = cv2.cvtColor(np.array(tampered), cv2.COLOR_RGB2BGR)
    
    original_gray = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered_cv, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
    diff = (diff * 255).astype("uint8")
    
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(tampered_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return original_cv, tampered_cv, diff, thresh, score

# User input for images
# uploaded_file_original = st.file_uploader("Upload Original Image", type=["jpg", "png", "jpeg"])
uploaded_file_tampered = st.file_uploader("Upload Tampered Image", type=["jpg", "png", "jpeg"])

uploaded_file_original = cv2.imread('pan-card.jpg')

if uploaded_file_original is not None and uploaded_file_tampered is not None:
    # original = Image.open(uploaded_file_original)
    original = cv2.imread('pan-card.jpg')
    print(original.shape)
    tampered = Image.open(uploaded_file_tampered)

    original, tampered = preprocess_images(original, tampered)
    
    st.subheader("Original Image")
    st.image(original, caption='Original Image', use_column_width=True)

    st.subheader("Tampered Image")
    st.image(tampered, caption='Tampered Image', use_column_width=True)

    # Compute SSIM and detect tampering
    original_with_contour, tampered_with_contour, diff, thresh, ssim_score = detect_tampering(original, tampered)

    # st.subheader("Original Image with Contour")
    # st.image(original_with_contour, caption='Original Image with Contour', use_column_width=True)

    # st.subheader("Tampered Image with Contour")
    # st.image(tampered_with_contour, caption='Tampered Image with Contour', use_column_width=True)

    # st.subheader("Difference Image")
    # st.image(diff, caption='Difference Image', use_column_width=True)

    # st.subheader("Threshold Image")
    # st.image(thresh, caption='Threshold Image', use_column_width=True)

    # st.subheader("SSIM Score")
    # st.write(f"The Structural Similarity Index (SSIM) Score: {ssim_score}")

    # st.subheader("SSIM Score")
    if ssim_score >= 0.70 and ssim_score<= 1:
        st.write(f"<div style='text-align:center'><span style='color:green; font-size:100px;'>Not Tampered</span></div>", unsafe_allow_html=True)
    else:
        st.write(f"<div style='text-align:center'><span style='color:red; font-size:100px;'>Tampered</span></div>", unsafe_allow_html=True)
