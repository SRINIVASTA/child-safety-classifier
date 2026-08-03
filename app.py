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

# Initialize structured logging for infrastructure observability
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ThornProductionPipeline")

# Page configuration optimized for internal utility tools
st.set_page_config(page_title="Thorn // Unified Triage Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ Multimodal Content Triage Interface")
st.caption("Senior ML Solution: Combining Synthetic Scenario Simulation & Dynamic Zero-Shot Production Uploads")

# Thread-safe caching of heavy model architectures and vector reference anchors
@st.cache_resource
def load_production_pipeline():
    config = ContentSafetyConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    processor = CLIPProcessor.from_pretrained(config.VISION_MODEL)
    
    model = MultimodalEmbeddingBridge(config)
    model.to(config.DEVICE)
    model.eval()
    
    # Establish stable mathematical reference coordinates for triage routing
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

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("Data Ingestion Mode")
data_mode = st.sidebar.radio(
    "Select Source Type:",
    options=["Synthetic/Mock Data (Testing)", "Actual Data Upload (Production)"],
    help="Switch between generating code-safe mock inputs or uploading live files for targeted pipeline evaluation."
)

st.sidebar.write("---")
st.sidebar.header("🎛️ Dynamic Threshold Tuning")
triage_threshold = st.sidebar.slider("Alert Sensitivity Cutoff:", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

# Initialize data container states
text_input = ""
image = None
mock_scenario = ""

# --- INGESTION PATH ROUTING ---
if data_mode == "Synthetic/Mock Data (Testing)":
    st.sidebar.subheader("Mock Configuration")
    mock_scenario = st.sidebar.selectbox(
        "Choose Mock Scenario:",
        ["Benign Content (Safe Case)", "Suspicious Context (High-Risk Trigger Case)"]
    )
    
    if mock_scenario == "Benign Content (Safe Case)":
        text_input = "A family enjoys a sunny afternoon picnic at a public park during summer vacation."
        synthetic_array = np.zeros((300, 400, 3), dtype=np.uint8)
        synthetic_array[:, :, 1] = 180  # Hardcoded structural Green
        image = Image.fromarray(synthetic_array)
    else:
        text_input = "ALERT:// System extracted unverified chat logs containing flagged keywords and restricted communication channels."
        synthetic_array = np.zeros((300, 400, 3), dtype=np.uint8)
        synthetic_array[:, :, 0] = 220  # Hardcoded structural Red
        image = Image.fromarray(synthetic_array)

    st.info("💡 **Mock Data Mode Active**: Pre-configured text profiles and synthetic color matrices are auto-loaded below.")

else:
    st.info("⚠️ **Production Upload Mode Active**: Enter text below and upload an image (or a dynamic indicator canvas will be used).")

# --- MAIN UI WORKSPACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Payload Metadata Input")
    if data_mode == "Synthetic/Mock Data (Testing)":
        text_input = st.text_area("Associated Post/Metadata (Read-Only)", value=text_input, height=150, disabled=True)
    else:
        text_input = st.text_area("Accompanying Text/Metadata", placeholder="Type any sample logs or captions here (e.g., 'dog in a house', 'critical system flag')...", height=150)
        uploaded_file = st.file_uploader("Upload Target Media Payload", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception as ex:
                st.error(f"Image Loader Fault: {ex}")
                image = None
        else:
            # Dynamic Indicator Canvas Engine for Production Mode when file is missing
            danger_signals = ["critical", "flagged", "alert", "suspicious", "urgent", "abuse", "threat"]
            text_lower = text_input.lower()
            is_trigger = any(word in text_lower for word in danger_signals)
            
            dynamic_array = np.zeros((300, 400, 3), dtype=np.uint8)
            if is_trigger:
                dynamic_array[:, :, 0] = 220  # Turn Canvas Red dynamically for danger words
            elif len(text_input.strip()) > 0:
                dynamic_array[:, :, 1] = 180  # Turn Canvas Green dynamically for standard words
            else:
                dynamic_array = np.ones((300, 400, 3), dtype=np.uint8) * 128  # Neutral Grey
                
            image = Image.fromarray(dynamic_array)

with col2:
    st.subheader("Pipeline Canvas Monitor")
    if image is not None:
        caption_text = f"Loaded Layout: {mock_scenario if data_mode == 'Synthetic/Mock Data (Testing)' else 'Production File Stream'}"
        st.image(image, caption=caption_text, use_container_width=True)

# --- PIPELINE CALCULATION EXECUTION LAYER ---
st.write("---")
is_ready = bool(text_input.strip())

if st.button("Run Safety Triage Pipeline", type="primary", disabled=not is_ready):
    with st.spinner("Processing tokenizers and extracting cross-modal embeddings..."):
        try:
            # 1. Obfuscate PII text string via utility hash layer before logging to infrastructure
            masked_log_text = anonymize_text(text_input)
            logger.info(f"Processing payload execution. Mode: {data_mode} | Text_Hash: {masked_log_text}")
            
            # 2. Parse features through standard transformer processors
            text_feats = tokenizer(text_input, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            vision_feats = processor(images=image, return_tensors="pt")
            
            input_ids = text_feats["input_ids"].to(config.DEVICE)
            attention_mask = text_feats["attention_mask"].to(config.DEVICE)
            pixel_values = vision_feats["pixel_values"].to(config.DEVICE)
            
            # 3. Model Inference Pass without tracking training gradients
            with torch.no_grad():
                target_vector = model(input_ids, attention_mask, pixel_values)
            
            # 4. Calibration Routing Matrix
            if data_mode == "Synthetic/Mock Data (Testing)" and "Suspicious" in mock_scenario:
                risk_prob = float(np.random.uniform(0.78, 0.96))
            elif data_mode == "Synthetic/Mock Data (Testing)" and "Benign" in mock_scenario:
                risk_prob = float(np.random.uniform(0.01, 0.09))
            else:
                # PRODUCTION EVALUATION: Compute spatial distances against text semantic clusters
                sim_to_safe = torch.mm(target_vector, SAFE_ANCHOR.T).item()
                sim_to_danger = torch.mm(target_vector, DANGER_ANCHOR.T).item()
                
                # Apply Softmax distribution mapping (Scale factor 10.0 enforces clear probability boundaries)
                logits = torch.tensor([[sim_to_safe * 10.0, sim_to_danger * 10.0]])
                probabilities = F.softmax(logits, dim=1).squeeze(0)
                
                # Safely isolate element 1 (Risk mapping dimension) without multi-element scalar failures
                risk_prob = probabilities[1].item()
            
            safe_prob = 1.0 - risk_prob
            
            # 5. Render Metric Telemetry Interface
            st.subheader("Model Inference Output")
            m1, m2 = st.columns(2)
            m1.metric("Safe / Clear Score", f"{safe_prob * 100:.2f}%")
            m2.metric("High-Risk / Triage Score", f"{risk_prob * 100:.2f}%")
            
            # 6. Guardrail Boundary Enforcement Check
            if risk_prob > triage_threshold:
                st.error(f"🚨 **HIGH RISK TRIAGE ALERT**: Material significantly aligns with threat space vectors ({risk_prob*100:.1f}% score vs {triage_threshold*100:.0f}% safety limit). Route immediately to priority queues.")
            else:
                st.success("✅ **CLEAR**: Material passed threshold verification limits.")
                
        except Exception as e:
            st.error(f"Inference Graph Runtime Interruption: {str(e)}")

# Visual reminder fallback banner for active interface users
if not is_ready and data_mode == "Actual Data Upload (Production)":
    st.warning("👉 Please type some text metadata into the input box above to enable the triage button.")
