import streamlit as st

# import the necessary packages for image recognition
#from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.applications import InceptionV3
#from tensorflow.keras.applications import Xception  # TensorFlow ONLY
#from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import pandas as pd
import urllib
from predict import make_prediction_

# set page layout
st.set_page_config(
    page_title="Leukemia Cell Classification",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Leukemia Cell Classification")
st.sidebar.subheader("Input")
models_list = ["VGG16"]
network = st.sidebar.selectbox("Select a Model", models_list)


# component to upload images
uploaded_file = st.sidebar.file_uploader(
    "Choose an infected or non-infected blood smear image to classify", type=["jpg", "jpeg", "png"]
)
# component for toggling code
show_code = st.sidebar.checkbox("Show Code")

if uploaded_file:
    bytes_data = uploaded_file.read()

    inputShape = (64, 64)
    preprocess = imagenet_utils.preprocess_input

    image = Image.open(BytesIO(bytes_data))
    image = image.convert("RGB")

    image = image.resize(inputShape)
    image = img_to_array(image)

    processed_image = np.array(image, dtype="float") / 255.0
    pred, label = make_prediction_(processed_image, "C:/Users/akinr/Desktop/GitHub/Leukemia-Diagnosis-ML/Output")
    
    st.image(bytes_data)
    st.subheader(f"Top Predictions from {network}")
    
    df = pd.DataFrame(
            label, columns=["Classification"]
        )
    df["Confidence"] = pred
    st.dataframe(
        df
    )

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = "https://raw.githubusercontent.com/abooBala/Leukemia-Diagnosis-ML/master/" + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


if show_code:
    st.code(get_file_content_as_string("ml_frontend.py"))
