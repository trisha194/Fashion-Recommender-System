import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm

# Load the ResNet50 model without the top classification layers
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add a global max pooling layer
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    # Use tf.keras.utils.load_img and tf.keras.utils.img_to_array
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Get all image file paths
filenames = [os.path.join('images', file) for file in os.listdir('images')]

# Extract features for all images with a progress bar
feature_list = []
for file in tqdm(filenames, desc="Extracting features", unit="image"):
    feature_list.append(extract_features(file, model))

# Save the features and filenames using pickle
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
