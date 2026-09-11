import streamlit as st
import requests
import json
from PIL import Image
import base64
import time

# Configure the Streamlit page to look like a professional application
st.set_page_config(
    page_title="AI DermatoDiagnose Pro",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful, premium result card
st.markdown("""
<style>
    /* Premium Header */
    .premium-header {
        background: rgba(43, 88, 118, 0.1);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .premium-header h1 {
        margin: 0;
        padding: 0;
    }

    /* Result card styling */
    .result-card {
        background: rgba(16, 185, 129, 0.1);
        border-left: 5px solid #10b981;
        padding: 2rem;
        margin-top: 1rem;
        text-align: center;
    }
    
    .disease-name {
        font-size: 2rem;
        font-weight: bold;
        color: #10b981;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Disease Information Database (Mockup for Professionalism)
# -------------------------------------------------------------
DISEASE_INFO = {
    "Acne and Rosacea": "Common inflammatory skin conditions causing redness, pimples, and broken blood vessels.",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "Pre-cancerous or cancerous growths usually caused by long-term sun exposure. Requires medical attention.",
    "Atopic Dermatitis": "A chronic condition making skin red and itchy. Also known as eczema.",
    "Bullous Disease": "A group of rare autoimmune conditions causing fluid-filled blisters on the skin.",
    "Cellulitis Impetigo and other Bacterial Infections": "Common and potentially serious bacterial infections of the deep layers of the skin.",
    "Eczema": "A condition that makes skin red, itchy, and inflamed.",
    "Exanthems and Drug Eruptions": "A widespread rash usually occurring in children or as a reaction to medication.",
    "Hair Loss Photos Alopecia and other Hair Diseases": "Conditions resulting in abnormal hair loss from the scalp or body.",
    "Herpes HPV and other STDs": "Viral infections that can cause sores or warts on the skin.",
    "Light Diseases and Disorders of Pigmentation": "Conditions affecting skin color, such as vitiligo or melasma.",
    "Lupus and other Connective Tissue diseases": "Autoimmune conditions that can cause distinctive rashes, particularly on the face.",
    "Melanoma Skin Cancer Nevi and Moles": "Different types of skin spots, ranging from benign moles to aggressive skin cancer (melanoma).",
    "Nail Fungus and other Nail Disease": "Fungal infections or structural problems affecting fingernails and toenails.",
    "Poison Ivy Photos and other Contact Dermatitis": "Allergic skin reactions caused by direct contact with irritating substances.",
    "Psoriasis pictures Lichen Planus and related diseases": "Autoimmune diseases causing thick, scaly patches of skin.",
    "Scabies Lyme Disease and other Infestations and Bites": "Skin reactions caused by mites, ticks, or other parasites.",
    "Seborrheic Keratoses and other Benign Tumors": "Non-cancerous skin growths that typically appear in older adults.",
    "Systemic Disease": "Skin manifestations of internal diseases affecting the whole body.",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Common fungal infections that can affect any part of the body.",
    "Urticaria Hives": "An outbreak of swollen, pale red bumps or plaques on the skin that appear suddenly.",
    "Vascular Tumors": "Abnormal growths of blood vessels in or under the skin."
}

# -------------------------------------------------------------
# Main Application UI
# -------------------------------------------------------------
st.markdown("""
<div class="premium-header">
    <h1>🧬 AI DermatoDiagnose Pro</h1>
    <p style="color: #94a3b8; font-size: 1.1rem;">Enterprise-Grade Dermatological Classification System</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### 📸 Image Analysis")
    st.markdown("Upload a high-resolution dermoscopic or clinical image for AI processing.")
    
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Patient Image", use_container_width=True)
        
        # Analyze button
        analyze_btn = st.button("🔍 Run AI Diagnostics", type="primary")

with col2:
    st.markdown("### 🔬 Diagnostic Results")
    
    if uploaded_file is None:
        st.info("Waiting for image upload... Please provide a clinical image on the left panel to begin diagnostic analysis.")
    elif 'analyze_btn' in locals() and analyze_btn:
        with st.spinner("Initializing PyTorch Tensor Cores (RTX 3050)..."):
            
            # Send file to local FastAPI backend
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # API Call to fastAPI backend
                response = requests.post("http://localhost:8000/predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("status") == "success":
                        prediction = data["predicted_class"]
                        confidence = data["confidence"]
                        
                        # Generate the result card
                        st.markdown(f"""
                        <div class="result-card">
                            <span style="text-transform: uppercase; letter-spacing: 2px; font-size: 0.8rem; color: #10b981;">Primary Finding</span>
                            <div class="disease-name">{prediction}</div>
                            <div style="font-size: 1.2rem; margin-top: 10px;">
                                AI Confidence Score: <b>{confidence * 100:.2f}%</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show extra clinical information
                        info = DISEASE_INFO.get(prediction, "No specific clinical description available in the database for this classification.")
                        st.success(f"**Clinical Note:** {info}")
                        
                        st.divider()
                        
                        # Show top 5 probabilities
                        st.markdown("#### Differential Diagnosis (Top 5)")
                        all_probs = data["all_probabilities"]
                        sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)[:5]
                        
                        for disease, prob in sorted_probs:
                            pct = prob * 100
                            st.progress(prob, text=f"{disease} ({pct:.1f}%)")
                            
                    else:
                        st.error("API Error: The model could not process this image.")
                else:
                    st.error(f"Backend Offline. Ensure your FastAPI server is running on port 8000. Error Code: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection failed! Make sure your backend API is running (`python api.py`).")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# Sidebar info
with st.sidebar:
    st.markdown("## System Architecture")
    st.markdown("""
    This project incorporates state-of-the-art Deep Learning technologies:
    
    - **Backend Framework:** PyTorch & FastAPI
    - **Architecture:** EfficientNet-B3 (Transfer Learning)
    - **Frontend:** Streamlit Modern UI
    """)
    st.divider()
    st.success("System Status: Online & Connected")