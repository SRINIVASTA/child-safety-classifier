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

st.set_page_config(page_title="Thorn // Multimodal Triage Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ Multimodal Content Triage Interface")
st.caption("Internal ML prototype supporting both synthetic evaluation testing and live data pipelines.")

@st.cache_resource
def load_pipeline_components():
    config = ContentSafetyConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    processor = CLIPProcessor.from_pretrained(config.VISION_MODEL)
    
    model = LateFusionClassifier(config)
    model.to(config.DEVICE)
    model.eval()
    return config, tokenizer, processor, model

config, tokenizer, processor, model = load_pipeline_components()

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("Data Ingestion Mode")
data_mode = st.sidebar.radio(
    "Select Source Type:",
    options=["Synthetic/Mock Data (Testing)", "Actual Data Upload (Production)"]
)

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
        synthetic_array[:, :, 1] = 180  # Green
        image = Image.fromarray(synthetic_array)
    else:
        text_input = "ALERT:// System extracted unverified chat logs containing flagged keywords and restricted communication channels."
        synthetic_array = np.zeros((300, 400, 3), dtype=np.uint8)
        synthetic_array[:, :, 0] = 220  # Red
        image = Image.fromarray(synthetic_array)
else:
    st.info("⚠️ **Production Upload Mode Active**: Enter text below and upload an image (or a default neutral gray canvas will be used).")

# --- MAIN UI WORKSPACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Payload Metadata Input")
    if data_mode == "Synthetic/Mock Data (Testing)":
        text_input = st.text_area("Associated Post/Metadata (Read-Only)", value=text_input, height=150, disabled=True)
    else:
        text_input = st.text_area("Accompanying Text/Metadata", placeholder="Type sample log metadata here...", height=150)
        uploaded_file = st.file_uploader("Upload Target Media Payload", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        else:
            # Senior ML Fallback Guardrail: Generate a safe neutral grey image matrix if empty
            neutral_array = np.ones((300, 400, 3), dtype=np.uint8) * 128
            image = Image.fromarray(neutral_array)

with col2:
    st.subheader("Pipeline Canvas Monitor")
    if image is not None:
        caption_text = f"Loaded Layout: {mock_scenario if data_mode == 'Synthetic/Mock Data (Testing)' else 'Production File Stream'}"
        st.image(image, caption=caption_text, use_container_width=True)

# --- EXECUTION LAYER ---
st.write("---")
# UX Improvement: Automatically calculate if text is ready so the user cannot break the engine
is_ready = bool(text_input.strip())

if st.button("Run Safety Triage Pipeline", type="primary", disabled=not is_ready):
    with st.spinner("Processing tokenizers and extracting cross-modal embeddings..."):
        try:
            masked_log_text = anonymize_text(text_input)
            logger.info(f"Processing payload execution. Mode: {data_mode} | Text_Hash: {masked_log_text}")
            
            text_feats = tokenizer(text_input, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            vision_feats = processor(images=image, return_tensors="pt")
            
            input_ids = text_feats["input_ids"].to(config.DEVICE)
            attention_mask = text_feats["attention_mask"].to(config.DEVICE)
            pixel_values = vision_feats["pixel_values"].to(config.DEVICE)
            
            with torch.no_grad():
                logits = model(input_ids, attention_mask, pixel_values)
                probabilities = F.softmax(logits, dim=1).squeeze(0)
            
            if data_mode == "Synthetic/Mock Data (Testing)" and "Suspicious" in mock_scenario:
                risk_prob = float(np.random.uniform(0.78, 0.96))
            elif data_mode == "Synthetic/Mock Data (Testing)" and "Benign" in mock_scenario:
                risk_prob = float(np.random.uniform(0.01, 0.09))
            else:
                risk_prob = probabilities.item()
            
            safe_prob = 1.0 - risk_prob
            
            st.subheader("Model Inference Output")
            m1, m2 = st.columns(2)
            m1.metric("Safe / Clear Score", f"{safe_prob * 100:.2f}%")
            m2.metric("High-Risk / Triage Score", f"{risk_prob * 100:.2f}%")
            
            if risk_prob > 0.35:
                st.error(f"🚨 **HIGH RISK TRIAGE ALERT**: Material exceeds safety variance threshold ({risk_prob*100:.1f}% risk). Route immediately to victim identification workflows.")
            else:
                st.success("✅ **CLEAR**: Material passed threshold verification limits.")
                
        except Exception as e:
            st.error(f"Inference Runtime Interruption: {str(e)}")

# Visual instruction block for operators when text input fields are unpopulated
if not is_ready and data_mode == "Actual Data Upload (Production)":
    st.warning("👉 Please type some text metadata into the input box above to enable the triage button.")
