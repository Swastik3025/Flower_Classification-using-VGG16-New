import streamlit as st
import numpy as np
from io import StringIO
import time
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.saving import load_model

st.set_page_config(page_title="Flowers", page_icon=":Flowers:")

labels=['Lilly','Lotus','Orchid','Sunflower','Tulip']

@st.cache_resource
def load_model_from_path(model_path): # loading model
    model = load_model(model_path)
    print("Model loaded...")
    return model

def preprocess_image(img): # input data preprocessing
    image_arr = img_to_array(img)
    image_arr = image_arr.reshape((1, image_arr.shape[0], image_arr.shape[1], image_arr.shape[2]))
    resized_img = preprocess_input(image_arr)
    return resized_img

def predict_tumor(model, image_path, labels): # prediction
    image = load_img(image_path, target_size = (256,256))
    print("Image found...")
    print("Processing Image...")
    processed_image = preprocess_image(image)
    y_pred = tf.math.argmax(model.predict(processed_image), axis=1)
    Output = labels[y_pred[0]]
    return Output

# add sessions for variables, so those can be tracked throughout the session
if("model" not in st.session_state.keys()):
    st.session_state["model"] = load_model_from_path('Flower_class_new.h5')

model = st.session_state["model"]

st.write('# :blue[Flower Class Predictor]')
st.write("##### *Upload image and I will Predict!!!*")
path='https://e0.pxfuel.com/wallpapers/450/958/desktop-wallpaper-garden-flowers-colorful-garden-beautiful-grass-summer-pretty-field-freshness-flowers.jpg'
st.image(path, caption=None)

st.info('Please upload Flower image only!', icon="ℹ️")

img_actually_uploaded = False
uploaded_image_path = st.file_uploader(label="Choose an image", type=['png', 'jpeg', 'jpg'], label_visibility="collapsed")

if uploaded_image_path is not None:
    st.write(":green[Uploaded a file!]")
    img_actually_uploaded = True
    st.image(uploaded_image_path)
    st.write("Here is what I got!")

if img_actually_uploaded:
    if st.button(":red[Classify:]", key="prediction"):
        progress_text = "Prediction in progress, please wait..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text=progress_text)
        pred_tum = predict_tumor(model, uploaded_image_path, labels)

        st.success('Classification Successfull', icon="✅")
        st.write("Image is predicted to be ", pred_tum)
        st.info('Remove the uploaded image to predict again :)', icon="ℹ️")
else:
    st.button(":green[Predict for new]",disabled=True,key="prediction")