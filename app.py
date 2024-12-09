import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# Load the model
@st.cache_resource  # Caches the loaded model to avoid reloading on every app interaction
def load_model_from_file(model_path):
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Preprocess the input image
def preprocess_image(image):
    # Convert to RGB if needed
    if image.shape[-1] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[-1] == 1:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize to match model's input size
    image = cv2.resize(image, (150, 150))

    # Normalize pixel values
    image = image / 255.0

    # Expand dimensions to match batch size
    image = np.expand_dims(image, axis=0)

    return image

# Predict using the model
def predict_image(model, image):
    prediction = model.predict(image)
    return prediction[0][0]

# Streamlit App
def main():
    st.title("Lung Cancer Detection System")
    st.write("Upload a chest X-ray image to predict lung cancer.")

    # Load the model
    model_path = r"model/lung_cancer_model.h5"  # Adjust path as needed
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure the correct path.")
        return
    model = load_model_from_file(model_path)

    # Upload an image
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the uploaded file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = predict_image(model, processed_image)

        # Display the prediction
        if prediction > 0.5:
            st.warning(f"Prediction: Cancer Detected (Confidence: {prediction:.2f})")
        else:
            st.success(f"Prediction: Normal (Confidence: {1 - prediction:.2f})")

if __name__ == "__main__":
    main()


# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense, Dropout
# from tensorflow.keras import regularizers
# import numpy as np
# import cv2
# from PIL import Image


# class LungCancerDetectionSystem:
#     def __init__(self):
#         # Define the model architecture
#         self.model = Sequential()
#         self.model.add(Flatten(input_shape=(150, 150, 3)))  # Input shape (150x150x3)
#         self.model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#         self.model.add(Dropout(0.1))
#         self.model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#         self.model.add(Dropout(0.1))
#         self.model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#         self.model.add(Dropout(0.1))
#         self.model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#         self.model.add(Dropout(0.1))
#         self.model.add(Dense(1, activation='sigmoid'))

#         # Compile the model
#         self.model.compile(
#             optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )

#     def load_weights(self, weights_path):
#         # Load pre-trained weights
#         self.model.load_weights(weights_path)

#     def preprocess_image(self, image):
#         # Convert image to RGB if needed and resize to 150x150
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
#         image_array = np.array(image)
#         image_resized = cv2.resize(image_array, (150, 150))
#         return np.expand_dims(image_resized / 255.0, axis=0)  # Normalize and add batch dimension

#     def predict(self, image):
#         # Predict on the preprocessed image
#         return self.model.predict(image)[0][0]  # Return the probability


# # Streamlit Application
# def main():
#     st.title("Lung Cancer Detection System")
#     st.write("Upload a chest X-ray image to predict whether it indicates lung cancer.")

#     # Initialize the detection system
#     detector = LungCancerDetectionSystem()
#     detector.load_weights(r'model/lung_cancer_model.h5')  # Update with your model weights file path

#     # Image Upload
#     uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         # Display uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Preprocess and Predict
#         if st.button("Predict"):
#             try:
#                 preprocessed_image = detector.preprocess_image(image)
#                 prediction = detector.predict(preprocessed_image)
#                 if prediction > 0.5:
#                     st.success(f"The model predicts **Lung Cancer** with a confidence of {prediction:.2f}.")
#                 else:
#                     st.success(f"The model predicts **Normal** with a confidence of {1 - prediction:.2f}.")
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")


# if __name__ == "__main__":
#     main()
