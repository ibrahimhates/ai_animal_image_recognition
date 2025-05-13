import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import json

from model.model import ImageClassifier
from model.utils import load_class_mapping, visualize_predictions
from preprocessing.preprocessing import ImagePreprocessor

# Set page configuration
st.set_page_config(
    page_title="Hayvan Sınıflandırıcı",
    page_icon="🐾",
    layout="wide"
)

# Define constants
MODEL_PATH = "model/model_weights.pth"
CLASS_MAPPING_PATH = "model/class_mapping.json"
METRICS_PATH = "model/metrics.json"
IMAGE_SIZE = (224, 224)

# İtalyanca'dan Türkçe'ye hayvan isimlerinin çevirisi
ANIMAL_TRANSLATIONS = {
    "cane": "Köpek",
    "cavallo": "At",
    "elefante": "Fil",
    "farfalla": "Kelebek",
    "gallina": "Tavuk",
    "gatto": "Kedi",
    "mucca": "İnek",
    "pecora": "Koyun",
    "ragno": "Örümcek",
    "scoiattolo": "Sincap"
}

# Translate animal name from Italian to Turkish
def translate_animal_name(italian_name):
    return ANIMAL_TRANSLATIONS.get(italian_name, italian_name)

@st.cache_resource
def load_model():
    """
    Load the trained model and class mapping.
    
    Returns:
        tuple: (model, class_mapping, preprocessor)
    """
    try:
        # Load class mapping
        class_mapping = load_class_mapping(CLASS_MAPPING_PATH)
        num_classes = len(class_mapping)
        
        # Initialize the model with the actual number of classes
        model = ImageClassifier(num_classes=num_classes)
        model.load_state_dict(MODEL_PATH)
        
        # Initialize the preprocessor
        preprocessor = ImagePreprocessor(image_size=IMAGE_SIZE)
        
        return model, class_mapping, preprocessor
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None, None, None

def load_metrics():
    """
    Load model metrics if available
    """
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                return json.load(f)
        return None
    except Exception:
        return None

def main():
    # Display header
    st.title("🐾 Hayvan Sınıflandırıcı")
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-left: 5px solid #6c757d; color: #495057;'>
    Bu uygulama, yüklediğiniz hayvan görüntülerini yapay zeka kullanarak sınıflandırır.
    Sınıflandırılacak bir hayvan görüntüsü yükleyin ve sonuçları görün!
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize model variables
    model, class_mapping, preprocessor = None, None, None
    
    # Check if model files exist
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_MAPPING_PATH):
        # Load the model
        model, class_mapping, preprocessor = load_model()
        
        # Check if model is loaded successfully
        if model is None:
            st.warning("⚠️ Model yüklenemedi. Lütfen modelin eğitilmiş olduğundan emin olun.")
            return
            
        # Load and display metrics
        metrics = load_metrics()
        if metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric("Doğruluk (Accuracy)", f"{metrics['accuracy']:.2%}")
            col2.metric("Kesinlik (Precision)", f"{metrics['precision']:.2%}")
            col3.metric("Duyarlılık (Recall)", f"{metrics['recall']:.2%}")
    else:
        st.warning("""
        ⚠️ Eğitilmiş model bulunamadı. Lütfen önce modeli eğitin:
        ```
        python train_model.py --num_classes 10
        ```
        """)
        return
    
    # Display number of animal classes
    st.markdown(f"""
    <div style='margin-top: 15px; margin-bottom: 25px;'>
        <h3 style='color: #28a745;'>Toplam {len(class_mapping)} farklı hayvan türü tanıyabiliyorum</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for the UI
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # File uploader widget
        uploaded_file = st.file_uploader("Sınıflandırmak için bir hayvan görüntüsü yükleyin:", 
                                       type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen Hayvan Görüntüsü", use_container_width=True)
            
            # Preprocess button
            if st.button("Tahmin Et"):
                # Process the uploaded image
                with st.spinner("Hayvan türü tanımlanıyor..."):
                    # Preprocess the image for the model
                    processed_image = preprocessor.preprocess_image(image)
                    
                    # Make prediction
                    pred_idx, probabilities = model.predict(processed_image)
                    
                    # Prediction index check
                    if pred_idx in class_mapping:
                        italian_name = class_mapping[pred_idx]
                        pred_class = translate_animal_name(italian_name)
                    else:
                        pred_class = f"Bilinmeyen ({pred_idx})"
                    
                    # Display results in the second column
                    with col2:
                        st.markdown("""
                        <h2 style='color: #0275d8; margin-bottom: 20px;'>Sınıflandırma Sonuçları</h2>
                        """, unsafe_allow_html=True)
                        
                        # Display the prediction with larger font and emoji
                        confidence = probabilities[pred_idx] * 100
                        
                        # Select color based on confidence
                        if confidence > 85:
                            confidence_color = "#28a745"  # green for high confidence
                        elif confidence > 60:
                            confidence_color = "#ffc107"  # yellow for medium confidence
                        else:
                            confidence_color = "#dc3545"  # red for low confidence
                            
                        st.markdown(f"""
                        <div style='padding: 20px; background-color: #f0f7fb; border-radius: 10px; border: 2px solid #0275d8;'>
                            <h1 style='text-align: center; color: #0275d8;'>🐾 {pred_class} 🐾</h1>
                            <h3 style='text-align: center; color: {confidence_color};'>Güven Skoru: {confidence:.2f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Only show the top predicted animal types
                        top_k = 5  # Show only top 5 predictions
                        st.markdown("""
                        <div style='margin-top: 30px;'>
                            <h3 style='color: #17a2b8;'>🔍 En Olası Hayvan Türleri:</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get top k predictions
                        top_probs, top_classes = torch.tensor(probabilities).topk(min(top_k, len(probabilities)))
                        
                        # Create a table for the top predictions
                        data = []
                        for i, (prob, idx) in enumerate(zip(top_probs, top_classes)):
                            idx_item = idx.item()
                            if idx_item in class_mapping:
                                italian_name = class_mapping.get(idx_item)
                                class_name = translate_animal_name(italian_name)
                            else:
                                class_name = f"Bilinmeyen ({idx_item})"
                            probability = prob.item() * 100
                            data.append([f"{i+1}. {class_name}", f"{probability:.2f}%"])
                        
                        # Display the prediction table
                        st.table(data)

if __name__ == "__main__":
    main() 