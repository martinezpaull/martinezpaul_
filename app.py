import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

def main():
    st.title("Streamlit App")
   
    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model('final_model.h5')
        return model
    
    def import_and_predict(image_data, model):
        size=(28,28)
        image = ImageOps.fit(image_data,size, Image.LANCZOS)
        image = np.asarray(image)
        image = image / 255.0
        img_reshape = np.reshape(image, (1, 28, 28, 1))
        prediction = model.predict(img_reshape)
        return prediction

    model = load_model()
    class_names =  ["T-shirt/top", "Trouser/pants", "Pullover shirt", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    st.write("# Fashion-Mnist Classifier")
   
    file = st.file_uploader("Choose photo from computer", type=["jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        string = "Glucose Level: " + class_name
        st.success(string)
 
if __name__ == "__main__":
    main()
