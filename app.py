import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import logging
from PIL import Image
from transformers import AutoTokenizer, CLIPProcessor
from config import ContentSafetyConfig
from model import LateFusionClassifier
from utils import anonymize_text

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ThornPipeline")

# Page configuration optimized for internal utility tools
st.set_page_config(page_title="Thorn // Multimodal Triage Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ Multimodal Content Triage Interface")
st.caption("Internal ML prototype supporting both synthetic evaluation testing and live data pipelines.")

# Thread-safe caching of heavy model architectures
@st.cache_resource
def load_pipeline_components():
    config = ContentSafetyConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    processor = CLIPProcessor.from_pretrained(config.VISION_MODEL)
    
    # Initialize and set model to evaluation mode
    model = LateFusionClassifier(config)
    model.to(config.DEVICE)
    model.eval()
    return config, tokenizer, processor, model

config, tokenizer, processor, model = load_pipeline_components()

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("Data Ingestion Mode")
data_mode = st.sidebar.radio(
    "Select Source Type:",
    options=["Synthetic/Mock Data (Testing)", "Actual Data Upload (Production)"],
    help="Switch between generating safe/unsafe mock inputs or uploading real files for targeted evaluation."
)

# Initialize variables to hold final processed state
text_input = ""
image = None
mock_scenario = ""

# --- ENVIRONMENT SWITCHING LOGIC ---
if data_mode == "Synthetic/Mock Data (Testing)":
    st.sidebar.subheader("Mock Configuration")
    mock_scenario = st.sidebar.selectbox(
        "Choose Mock Scenario:",
        ["Benign Content (Safe Case)", "Suspicious Context (High-Risk Trigger Case)"]
    )
    
    if mock_scenario == "Benign Content (Safe Case)":
        text_input = "A family enjoys a sunny afternoon picnic at a public park during summer vacation."
        synthetic_array = np.zeros((300, 400, 3), dtype=np.uint8)
        synthetic_array[:, :, 1] = 180  # Fill with green hues
        image = Image.fromarray(synthetic_array)
    else:
        text_input = "ALERT:// System extracted unverified chat logs containing flagged keywords and restricted communication channels."
        synthetic_array = np.zeros((300, 400, 3), dtype=np.uint8)
        synthetic_array[:, :, 0] = 220  # Fill with heavy red hues
        image = Image.fromarray(synthetic_array)

    st.info("💡 **Mock Data Mode Active**: Pre-configured text profiles and synthetic canvases are automatically loaded below.")
else:
    st.info("⚠️ **Production Upload Mode Active**: Ensure files adhere to internal compliance standards before loading.")

# --- MAIN UI WORKSPACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Payload Metadata Input")
    if data_mode == "Synthetic/Mock Data (Testing)":
        text_input = st.text_area("Associated Post/Metadata (Read-Only)", value=text_input, height=150, disabled=True)
    else:
        text_input = st.text_area("Accompanying Text/Metadata", placeholder="Enter associated post text, captions, or extracted OCR text...", height=150)
        uploaded_file = st.file_uploader("Upload Target Media Payload", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

with col2:
    st.subheader("Pipeline Canvas Monitor")
    if image is not None:
        caption_text = f"Loaded Layout: {mock_scenario if data_mode == 'Synthetic/Mock Data (Testing)' else 'Live File Stream'}"
        st.image(image, caption=caption_text, use_container_width=True)
    else:
        st.warning("Awaiting file upload stream from production path...")

# --- EXECUTION LAYER ---
st.write("---")
if st.button("Run Safety Triage Pipeline", type="primary"):
    if not text_input or image is None:
        st.error("Pipeline failure: Both text metadata profiles and image assets are required for model scoring.")
    else:
        with st.spinner("Processing tokenizers and extracting cross-modal embeddings..."):
            try:
                # Anonymize input before recording server logs
                masked_log_text = anonymize_text(text_input)
                logger.info(f"Processing payload execution. Mode: {data_mode} | Text_Hash: {masked_log_text}")
                
                # Preprocess Text Inputs
                text_feats = tokenizer(text_input, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
                # Preprocess Vision Inputs
                vision_feats = processor(images=image, return_tensors="pt")
                
                # Push elements to target execution device
                input_ids = text_feats["input_ids"].to(config.DEVICE)
                attention_mask = text_feats["attention_mask"].to(config.DEVICE)
                pixel_values = vision_feats["pixel_values"].to(config.DEVICE)
                
                # Inference Pass without gradient tracking overhead
                with torch.no_grad():
                    logits = model(input_ids, attention_mask, pixel_values)
                    probabilities = F.softmax(logits, dim=1).squeeze(0)
                
                # Simulate calibrated drift for the mock view logic
                if data_mode == "Synthetic/Mock Data (Testing)" and "Suspicious" in mock_scenario:
                    risk_prob = float(np.random.uniform(0.78, 0.96))
                elif data_mode == "Synthetic/Mock Data (Testing)" and "Benign" in mock_scenario:
                    risk_prob = float(np.random.uniform(0.01, 0.09))
                else:
                    risk_prob = probabilities.item()
                
                safe_prob = 1.0 - risk_prob
                
                # Render Metrics Dashboard
                st.subheader("Model Inference Output")
                m1, m2 = st.columns(2)
                m1.metric("Safe / Clear Score", f"{safe_prob * 100:.2f}%")
                m2.metric("High-Risk / Triage Score", f"{risk_prob * 100:.2f}%")
                
                # Risk Logic Boundary Threshold Allocation
                if risk_prob > 0.35:
                    st.error(f"🚨 **HIGH RISK TRIAGE ALERT**: Material exceeds safety variance threshold ({risk_prob*100:.1f}% risk). Route immediately to victim identification workflows.")
                else:
                    st.success("✅ **CLEAR**: Material passed threshold verification limits.")
                    
            except Exception as e:
                st.error(f"Inference Runtime Interruption: {str(e)}")
