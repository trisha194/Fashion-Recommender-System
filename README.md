# ✨ Fashion Recommender System ✨

A **Fashion Recommender System** that suggests similar fashion items based on an uploaded image. The system uses **ResNet50** for feature extraction and **K-Nearest Neighbors (KNN)** for finding similar items. Built with **Streamlit** for an interactive web interface.

### Live Demo

Check out the live version of my portfolio: [Trisha's Portfolio]( http://192.168.0.103:8501)


---

## 🚀 Features
- Upload an image of a fashion item (JPG/PNG).
- Extract features using **ResNet50** (pre-trained on ImageNet).
- Find and display 5 similar items from the dataset using KNN.
- Easy-to-use web interface built with **Streamlit**.

---

## 📂 Project Structure
```plaintext
Fashion-Recommender-System/
│
├── app.py                # Main Streamlit application
├── main.py               # Script for feature extraction and embeddings generation
├── embeddings.pkl        # Pre-computed feature embeddings
├── filenames.pkl         # Corresponding image filenames
├── images/               # Folder containing fashion images
├── uploads/              # Folder to store uploaded images
└── README.md             # Project documentation
