import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "C:\\Users\\varda\\OneDrive\\Desktop\\Case Study\\Efficient_net_improved.h5"

class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)

    def call(self, inputs, training=None):
        if training is None:
            training = True
        if not training:
            return inputs
        return super(FixedDropout, self).call(inputs, training=training)

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'FixedDropout': FixedDropout})
    return model

@st.cache(allow_output_mutation=True)
def get_model():
    return load_model(MODEL_PATH)

def preprocess_image(image):
    # Preprocess the image as required for your model
    image = image.resize((96, 96))
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

def predict_class(image, model):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

def main():
    st.title('Road Surface Type Classifier')
    file = st.file_uploader("Upload an image of road surface", type=["jpg", "jpeg", "png"])

    if file is None:
        st.text('Please upload an image.')
    else:
        try:
            slot = st.empty()
            slot.text('Running inference....')
            
            test_image = Image.open(file)
            st.image(test_image, caption="Input Image", use_column_width=True)
            
            model = get_model()
            class_names = ['wet_asphalt_severe', 'wet_concrete_severe', 'wet_gravel']  # Define your classes
            pred = predict_class(test_image, model)
            result_idx = np.argmax(pred)
            result = class_names[result_idx]
            st.success(f"The road surface is predicted as {result}.")
            
            slot.text('Done')
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
