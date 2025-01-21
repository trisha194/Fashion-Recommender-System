import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load feature embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])



# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #0f0f0f;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        .container {
            max-width: 800px;
            margin: 30px auto;
            padding: 30px;
            background-color: #1e1e1e;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        }
        h1 {
            color: #ffa500;
            text-align: center;
        }
        .file-upload {
            margin-top: 20px;
            text-align: center;
        }
        button {
            background-color: #ffa500;
            color: #000;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #ffb733;
        }
        .result {
            margin-top: 20px;
            text-align: center;
        }
        .recommended-images {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .recommended-images img {
            width: 120px;
            border-radius: 10px;
            margin: 10px;
            border: 2px solid #ffa500;
        }
        .recommended-images img:hover {
            transform: scale(1.1);
            transition: transform 0.3s;
        }
    </style>
""", unsafe_allow_html=True)

st.title('✨Fashion Recommender System✨')

# Ensure 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')


def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file⚠️: {e}")
        return None


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# File upload
uploaded_file = st.file_uploader("📂Upload an image (JPG/PNG)")
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_container_width=True)

        # Feature extraction
        features = feature_extraction(file_path, model)

        # Recommendations
        indices = recommend(features, feature_list)

        # Show recommended images
        st.text("Recommended Items")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[0][i]], width=100)
    else:
        st.error("File upload failed. Please try again.")



