import streamlit as st
from PIL import Image # Image processing library
import tensorflow as tf
from tensorflow.keras.models import load_model
# import tensorflow.keras.model import load_model
import numpy as np

# Create the title for the APP
st.title("Cat_Dog Image Classification")
st.write("Upload an image of a cat or a dog, and we'll predict which is it")

# Create a file uploader
uploaded_file = st.file_uploader("Upload an image..", type=["jpg", "jpeg", "png"])

# Check if the image is uploaded
if uploaded_file is not None:
    # display the image
    image = Image.open(uploaded_file)
    # st.image(image, caption="Uploaded Image", use_column_width= "auto")
    st.image(image, caption="Uploaded Image")
    st.write("")

    # Preprocess the image
    img = np.array(image) # convert the image to numpy array from Pillow(PIL image)
    # resize the image
    img = tf.image.resize(img, (64, 64))
    # normalize the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    #st.write(f"{img, shape}")

    # Load the trained model
    # model = load_model("C:/Users/Olamide Joseph/image_st_dep/vgg_model.keras")
    model = load_model("vgg_model.keras")
    

    # Make predictions
    prediction = model.predict(img)
    label = "Cat" if prediction[0][0] > 0.5 else "Dog"

    # Display the prediction
    st.write(f" ## Predicted Image is: {label}")
