import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
import random
from PIL import Image
import warnings
import zipfile
import io

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(page_title="Leaf Disease Classifier", layout="wide")

# Project details
def show_project_details():
    st.title("GROUNDNUT LEAF DISEASE PREDICTION")
    st.subheader("MIT WPU")
    # st.markdown("### Groundnut leaf disease prediction")
    # st.markdown("### ACADEMIC SESSION \"JAN-APRIL 2025\"")
    st.markdown("#### CARRIED OUT BY- Tanmay Gulhane 1262250855")

# Load model and dataset
@st.cache_resource
def initialize_resources():
    # Load the trained model
    model = load_model("ensemble_model.h5", compile=False)
    
    # Extract dataset if needed
    dataset_zip = "predict.zip"
    dataset_dir = "dataset"
    
    if not os.path.exists(dataset_dir):
        with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
    
    # Collect image file paths
    all_images = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                all_images.append(os.path.join(root, file))
    
    print(f"Total images found: {len(all_images)}")
    return model, all_images

# Define class labels
class_labels = [
    'early_leaf_spot_1',
    'early_rust_1',
    'healthy_leaf_1',
    'late_leaf_spot_1',
    'nutrition_deficiency_1',
    'rust_1'
]

# Function to predict an image
def predict_image(model, image):
    """
    Predict disease from image
    
    Parameters:
    - model: Loaded TensorFlow model
    - image: Image as numpy array or path to image
    
    Returns:
    - predicted class name and confidence
    """
    # Handle different input types
    if isinstance(image, str):  # If path is provided
        img = cv2.imread(image)
        if img is None:
            st.error(f"Error: Unable to read image from path: {image}")
            return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:  # If numpy array is provided
        img = image
    
    # Preprocess
    img = cv2.resize(img, (256, 256))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    predictions = model.predict(img)
    probabilities = tf.nn.softmax(predictions).numpy()
    predicted_class_idx = np.argmax(probabilities)
    confidence = probabilities[0][predicted_class_idx]
    
    return class_labels[predicted_class_idx], confidence

def main():
    show_project_details()
    
    # Initialize model and dataset
    model, all_images = initialize_resources()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Random Image Prediction", "Upload Your Image"])
    
    with tab1:
        st.header("Random Image Prediction")
        if st.button('Select a Random image and predict'):
            if all_images:
                random_image_path = random.choice(all_images)
                
                # Get prediction
                predicted_class, confidence = predict_image(model, random_image_path)
                
                # Display results
                if predicted_class and confidence:
                    st.image(Image.open(random_image_path), caption="Selected Image", use_column_width=True)
                    
                    # Show prediction with color based on confidence
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Prediction:** {predicted_class}")
                    with col2:
                        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                        st.markdown(f"**Confidence:** <span style='color:{confidence_color}'>{confidence:.2f}</span>", unsafe_allow_html=True)
                    
                    # Display additional information about the disease
                    st.subheader("Disease Information")
                    if predicted_class == "healthy_leaf_1":
                        st.success("This leaf appears to be healthy!")
                    else:
                        st.warning(f"This leaf shows signs of {predicted_class.replace('_', ' ')}.")
                else:
                    st.error("Failed to generate prediction for the selected image.")
            else:
                st.error("No images available in the dataset.")
    
    with tab2:
        st.header("Upload Your Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read and display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert image for prediction
            img_array = np.array(image)
            
            # Make prediction when user clicks
            if st.button("Predict Uploaded Image"):
                # Get prediction
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence = predict_image(model, img_array)
                
                # Display results
                if predicted_class and confidence:
                    # Show prediction with color based on confidence
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Prediction:** {predicted_class}")
                    with col2:
                        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                        st.markdown(f"**Confidence:** <span style='color:{confidence_color}'>{confidence:.2f}</span>", unsafe_allow_html=True)
                    
                    # Display additional information about the disease
                    st.subheader("Disease Information")
                    if predicted_class == "healthy_leaf_1":
                        st.success("This leaf appears to be healthy!")
                    else:
                        st.warning(f"This leaf shows signs of {predicted_class.replace('_', ' ')}.")
                        
                        # Add treatment suggestions based on predicted disease
                        st.subheader("Treatment Suggestions")
                        if "leaf_spot" in predicted_class:
                            st.info("- Apply copper-based fungicides\n- Remove and destroy infected leaves\n- Ensure proper spacing between plants for air circulation")
                        elif "rust" in predicted_class:
                            st.info("- Apply sulfur or copper-based fungicides\n- Remove infected plant debris\n- Avoid overhead watering")
                        elif "nutrition_deficiency" in predicted_class:
                            st.info("- Apply balanced fertilizer\n- Check soil pH and adjust if necessary\n- Consider foliar feeding with micronutrients")
                else:
                    st.error("Failed to generate prediction for the uploaded image.")

if __name__ == "__main__":
    main()
#import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
# import os
# import random
# from PIL import Image
# import warnings
# import zipfile

# # Suppress warnings
# warnings.filterwarnings("ignore")

# dataset_zip = "predict.zip"
# dataset_dir = "dataset"  # Folder to extract images

# # Extract ZIP if not already extracted
# if not os.path.exists(dataset_dir):
#     with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
#         zip_ref.extractall(dataset_dir)

# # Collect image file paths from subfolders
# all_images = []
# for root, _, files in os.walk(dataset_dir):
#     for file in files:
#         if file.lower().endswith((".png", ".jpg", ".jpeg")):  # Add other formats if needed
#             all_images.append(os.path.join(root, file))

# # Check if images were found
# if not all_images:
#     raise FileNotFoundError("No images found in the extracted dataset directory!")

# print(f"Total images found: {len(all_images)}")

# # Project details
# def show_project_details():
#     st.title("DEPT OF INFORMATION TECHNOLOGY")
#     st.subheader("NATIONAL INSTITUTE OF TECHNOLOGY")
#     st.markdown("### COURSE PROJECT TITLE")
#     st.markdown("### ACADEMIC SESSION \"JAN-APRIL 2025\"")
#     st.markdown("#### CARRIED OUT BY- Chaitanya Gulhane 221AI015 AND Gagan Deepankar 221AI019")

# # Load the trained model
# model = load_model("ensemble_model.h5", compile=False)

# # Define class labels
# class_labels = [
#     'early_leaf_spot_1',
#     'early_rust_1',
#     'healthy_leaf_1',
#     'late_leaf_spot_1',
#     'nutrition_deficiency_1',
#     'rust_1'
# ]

# # Function to predict an image
# def predict_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         st.error("Error: Unable to read image.")
#         return None, None
    
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (256, 256))
#     image = img_to_array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
    
#     predictions = model.predict(image)
#     probabilities = tf.nn.softmax(predictions).numpy()
#     predicted_class = np.argmax(probabilities)
#     confidence = probabilities[0][predicted_class]
    
#     return class_labels[predicted_class], confidence

# # Display project details
# show_project_details()

# # Function to select random image and make prediction
# def run():
#     if all_images:
#         random_image_path = random.choice(all_images)
#         print(f"Selected Image: {random_image_path}")
        
#         # Get prediction for the selected image
#         predicted_class, confidence = predict_image(random_image_path)
        
#         # Check if prediction was successful
#         if predicted_class is not None and confidence is not None:
#             # Display image and prediction results
#             st.image(Image.open(random_image_path), caption="Selected Image", use_column_width=True)
#             st.write(f"**Prediction:** {predicted_class}  \n**Confidence:** {confidence:.2f}")
#         else:
#             st.error("Failed to generate prediction for the selected image.")
#     else:
#         st.error("No images available for prediction.")

# # Create button to trigger prediction
# st.button('Select a Random image and predict', on_click=run)
