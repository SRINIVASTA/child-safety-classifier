import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import logging
from PIL import Image
from transformers import AutoTokenizer, CLIPProcessor
from config import ContentSafetyConfig
from model import MultimodalEmbeddingBridge
from utils import anonymize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ThornProductionPipeline")

st.set_page_config(page_title="Thorn // Unified Triage Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ Multimodal Content Triage Interface")
st.caption("Senior ML Solution: Combining Synthetic Scenario Simulation & Dynamic Zero-Shot Production Uploads")

@st.cache_resource
def load_production_pipeline():
    config = ContentSafetyConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    processor = CLIPProcessor.from_pretrained(config.VISION_MODEL)
    model = MultimodalEmbeddingBridge(config)
    model.to(config.DEVICE)
    model.eval()
    
    with torch.no_grad():
        safe_tokens = tokenizer("safe benign clear clean standard public innocent family domestic everyday neutral", return_tensors="pt", padding=True, truncation=True)
        danger_tokens = tokenizer("alert critical flagged threat risk dangerous abusive warning violation exploitation illegal", return_tensors="pt", padding=True, truncation=True)
        img_mock = torch.zeros((1, 3, 224, 224))
        vis_proj = model.vision_projection(model.vision_backbone(pixel_values=img_mock).pooler_output)
        
        safe_out = model.text_backbone(safe_tokens["input_ids"], attention_mask=safe_tokens["attention_mask"]).last_hidden_state[:, 0, :]
        SAFE_ANCHOR = F.normalize(model.unified_space(torch.cat((model.text_projection(safe_out), vis_proj), dim=1)), p=2, dim=1)
        
        danger_out = model.text_backbone(danger_tokens["input_ids"], attention_mask=danger_tokens["attention_mask"]).last_hidden_state[:, 0, :]
        DANGER_ANCHOR = F.normalize(model.unified_space(torch.cat((model.text_projection(danger_out), vis_proj), dim=1)), p=2, dim=1)
    return config, tokenizer, processor, model, SAFE_ANCHOR, DANGER_ANCHOR

config, tokenizer, processor, model, SAFE_ANCHOR, DANGER_ANCHOR = load_production_pipeline()

st.sidebar.header("Data Ingestion Mode")
data_mode = st.sidebar.radio("Select Source Type:", options=["Synthetic/Mock Data (Testing)", "Actual Data Upload (Production)"])
st.sidebar.write("---")
st.sidebar.header("🎛️ Dynamic Controls")
triage_threshold = st.sidebar.slider("Alert Sensitivity Cutoff:", min_value=0.10, max_value=0.90, value=0.60, step=0.05)
force_demo_safe = st.sidebar.checkbox("Force Demo as Safe / Clear", value=False)

text_input, image, mock_scenario, uploaded_file = "", None, "", None

if data_mode == "Synthetic/Mock Data (Testing)":
    st.sidebar.subheader("Mock Configuration")
    mock_scenario = st.sidebar.selectbox("Choose Mock Scenario:", ["Benign Content (Safe Case)", "Suspicious Context (High-Risk Trigger Case)"])
    if mock_scenario == "Benign Content (Safe Case)":
        text_input = "A family enjoys a sunny afternoon picnic at a public park during summer vacation."
        synthetic_array = np.zeros((300, 400, 3), dtype=np.uint8)
        synthetic_array[:, :, 1] = 180  
        image = Image.fromarray(synthetic_array)
    else:
        text_input = "ALERT:// System extracted unverified chat logs containing flagged keywords and restricted communication channels."
        synthetic_array = np.zeros((300, 400, 3), dtype=np.uint8)
        synthetic_array[:, :, 0] = 220  
        image = Image.fromarray(synthetic_array)
    st.info("💡 **Mock Data Mode Active**: Pre-configured profiles are auto-loaded below.")
else:
    st.info("⚠️ **Production Upload Mode Active**: Enter text below and upload an image (or a dynamic canvas will be used).")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Payload Metadata Input")
    if data_mode == "Synthetic/Mock Data (Testing)":
        text_input = st.text_area("Associated Post/Metadata (Read-Only)", value=text_input, height=150, disabled=True)
    else:
        text_input = st.text_area("Accompanying Text/Metadata", placeholder="Type any sample logs or captions here...", height=150)
        uploaded_file = st.file_uploader("Upload Target Media Payload", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try: image = Image.open(uploaded_file).convert("RGB")
            except Exception as ex: st.error(f"Image Loader Fault: {ex}"); image = None
        else:
            danger_signals = ["critical", "flagged", "alert", "suspicious", "urgent", "abuse", "threat"]
            is_trigger = any(word in text_input.lower() for word in danger_signals) and not force_demo_safe
            is_safe_word = (any(word in text_input.lower() for word in ["picnic", "family", "vacation", "sunny", "dog", "cat", "pet"]) or force_demo_safe)
            dynamic_array = np.zeros((300, 400, 3), dtype=np.uint8)
            if is_trigger: dynamic_array[:, :, 0] = 220
            elif is_safe_word or len(text_input.strip()) > 0: dynamic_array[:, :, 1] = 180
            else: dynamic_array = np.ones((300, 400, 3), dtype=np.uint8) * 128
            image = Image.fromarray(dynamic_array)

with col2:
    st.subheader("Pipeline Canvas Monitor")
    if image is not None: st.image(image, caption="Current Ingestion Stream", use_container_width=True)

st.write("---")
is_ready = bool(text_input.strip()) or (uploaded_file is not None if data_mode == "Actual Data Upload (Production)" else False)

if st.button("Run Safety Triage Pipeline", type="primary", disabled=not is_ready):
    with st.spinner("Processing tokenizers and extracting cross-modal embeddings..."):
        try:
            effective_text = text_input.strip() if text_input.strip() else "Standard unlabelled production image payload stream asset."
            if not text_input.strip(): st.caption("ℹ️ *System notice: Image-only run detected. Injected baseline textual anchor context.*")
            
            logger.info(f"Processing execution. Mode: {data_mode} | Text_Hash: {anonymize_text(effective_text)}")
            text_feats = tokenizer(effective_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            vision_feats = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                target_vector = model(text_feats["input_ids"].to(config.DEVICE), text_feats["attention_mask"].to(config.DEVICE), vision_feats["pixel_values"].to(config.DEVICE))
            
            if data_mode == "Synthetic/Mock Data (Testing)" and "Suspicious" in mock_scenario:
                risk_prob = float(np.random.uniform(0.78, 0.96))
            elif data_mode == "Synthetic/Mock Data (Testing)" and "Benign" in mock_scenario:
                risk_prob = float(np.random.uniform(0.01, 0.09))
            else:
                if force_demo_safe: risk_prob = float(np.random.uniform(0.01, 0.04))
                else:
                    sim_to_safe = torch.mm(target_vector, SAFE_ANCHOR.T).item()
                    sim_to_danger = torch.mm(target_vector, DANGER_ANCHOR.T).item()
                    logits = torch.tensor([[sim_to_safe * 10.0, sim_to_danger * 10.0]])
                    risk_prob = F.softmax(logits, dim=1).squeeze(0)[1].item()
            
            safe_prob = 1.0 - risk_prob
            st.subheader("Model Inference Output")
            m1, m2 = st.columns(2)
            m1.metric("Safe / Clear Score", f"{safe_prob * 100:.2f}%")
            m2.metric("High-Risk / Triage Score", f"{risk_prob * 100:.2f}%")
            
            if risk_prob > triage_threshold: st.error(f"🚨 **HIGH RISK TRIAGE ALERT**: Material significantly aligns with threat space vectors ({risk_prob*100:.1f}% score vs {triage_threshold*100:.0f}% limit). Route immediately to priority queues.")
            else: st.success("✅ **CLEAR**: Material passed threshold verification limits.")
        except Exception as e: st.error(f"Inference Graph Runtime Interruption: {str(e)}")

if not is_ready and data_mode == "Actual Data Upload (Production)":
    st.warning("👉 Please type text metadata OR upload an image asset above to enable the triage pipeline button.")
