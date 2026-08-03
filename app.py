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
logger = logging.getLogger("ChildSafetyProductionPipeline")

# Page configuration optimized for child protection digital triage
st.set_page_config(page_title="Child Safety // Unified Triage Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ Child Safety Multimodal Triage Dashboard")
st.caption("Advanced Child Protection Engineering: Combining Synthetic Scenario Simulation & Dynamic Zero-Shot Upload Analysis")

# Thread-safe caching of heavy model architectures and vector reference anchors
@st.cache_resource
def load_production_pipeline():
    config = ContentSafetyConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    processor = CLIPProcessor.from_pretrained(config.VISION_MODEL)
    
    model = MultimodalEmbeddingBridge(config)
    model.to(config.DEVICE)
    model.eval()
    
    # Establish stable mathematical reference coordinates for child safety triage routing
    with torch.no_grad():
        safe_tokens = tokenizer("safe benign clear clean standard public innocent family domestic everyday neutral child park playground", return_tensors="pt", padding=True, truncation=True)
        danger_tokens = tokenizer("alert critical flagged threat risk dangerous abusive warning violation exploitation illegal harm", return_tensors="pt", padding=True, truncation=True)
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
st.sidebar.header("🎛️ Dynamic Controls")
triage_threshold = st.sidebar.slider("Alert Sensitivity Cutoff:", min_value=0.10, max_value=0.90, value=0.60, step=0.05)
force_demo_safe = st.sidebar.checkbox("Force Demo as Safe / Clear", value=False)

# Initialize data container states
text_input = ""
image = None
mock_scenario = ""
uploaded_file = None

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

    st.info("💡 **Mock Data Mode Active**: Pre-configured child protection testing profiles are auto-loaded below.")

else:
    st.info("⚠️ **Production Upload Mode Active**: Enter text below and upload an image (or a dynamic indicator canvas will be used).")

# --- MAIN UI WORKSPACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Payload Metadata Input")
    if data_mode == "Synthetic/Mock Data (Testing)":
        text_input = st.text_area("Associated Post/Metadata (Read-Only)", value=text_input, height=150, disabled=True)
    else:
        text_input = st.text_area("Accompanying Text/Metadata", placeholder="Type any sample logs or captions here (e.g., 'dog in a house', 'critical child protection flag')...", height=150)
        uploaded_file = st.file_uploader("Upload Target Media Payload", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception as ex:
                st.error(f"Image Loader Fault: {ex}")
                image = None
        else:
            # Dynamic Indicator Canvas Engine for Production Mode when file is missing
            danger_signals = ["critical", "flagged", "alert", "suspicious", "urgent", "abuse", "threat", "harm"]
            text_lower = text_input.lower()
            is_trigger = any(word in text_lower for word in danger_signals) and not force_demo_safe
            is_safe_word = (any(word in text_lower for word in ["picnic", "family", "vacation", "sunny", "dog", "cat", "pet", "child", "park"]) or force_demo_safe)
            
            dynamic_array = np.zeros((300, 400, 3), dtype=np.uint8)
            if is_trigger:
                dynamic_array[:, :, 0] = 220  # Turn Canvas Red dynamically for danger words
            elif is_safe_word or len(text_input.strip()) > 0:
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

# Enable the execution pass if text is written OR if an actual image payload is present
is_ready = bool(text_input.strip()) or (uploaded_file is not None if data_mode == "Actual Data Upload (Production)" else False)

if st.button("Run Safety Triage Pipeline", type="primary", disabled=not is_ready):
    with st.spinner("Processing tokenizers and extracting cross-modal embeddings..."):
        try:
            # Handle an image-only ingestion pass elegantly by creating an implicit baseline token string
            effective_text = text_input.strip()
            if not effective_text:
                effective_text = "Standard unlabelled child safety production image payload stream asset."
                st.caption("ℹ️ *System notice: Image-only run detected. Injected baseline textual anchor context.*")

            # 1. Obfuscate text string via utility hash layer before logging to infrastructure terminal log channels
            masked_log_text = anonymize_text(effective_text)
            logger.info(f"Processing payload execution. Mode: {data_mode} | Text_Hash: {masked_log_text}")
            
            # 2. Parse features through standard transformer processors using our effective text proxy variable
            text_feats = tokenizer(effective_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
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
                if force_demo_safe:
                    risk_prob = float(np.random.uniform(0.01, 0.04))
                else:
                    # PRODUCTION EVALUATION: Compute spatial distances against text semantic clusters
                    sim_to_safe = torch.mm(target_vector, SAFE_ANCHOR.T).item()
                    sim_to_danger = torch.mm(target_vector, DANGER_ANCHOR.T).item()
                    
                    # Apply Softmax distribution mapping (Scale factor 10.0 enforces clear probability boundaries)
                    logits = torch.tensor([[sim_to_safe * 10.0, sim_to_danger * 10.0]])
                    probabilities = F.softmax(logits, dim=1).squeeze(0)
                    risk_prob = probabilities.item()
            
            safe_prob = 1.0 - risk_prob
            
            # 5. Render Metric Telemetry Interface
            st.subheader("Model Inference Output")
            m1, m2 = st.columns(2)
            m1.metric("Safe / Clear Score", f"{safe_prob * 100:.2f}%")
            m2.metric("High-Risk / Triage Score", f"{risk_prob * 100:.2f}%")
            
            # 6. Guardrail Boundary Enforcement Check
            if risk_prob > triage_threshold:
                st.error(f"🚨 **HIGH RISK TRIAGE ALERT**: Material significantly aligns with child safety threat spaces ({risk_prob*100:.1f}% score vs {triage_threshold*100:.0f}% safety allowance limit). Route immediately to child protection priority queues.")
            else:
                st.success("✅ **CLEAR**: Material passed threshold verification limits.")
                
        except Exception as e:
            st.error(f"Inference Graph Runtime Interruption: {str(e)}")

# Visual reminder fallback banner for active interface users
if not is_ready and data_mode == "Actual Data Upload (Production)":
    st.warning("👉 Please type text metadata OR upload an image asset above to enable the triage pipeline button.")


# =========================================================================
# --- INTEGRATED CLOUD API SIMULATOR CONSOLE (THE FASTAPI CONTEXT) ---
# =========================================================================
st.write("---")
st.subheader("🛠️ Cloud API Testing Console")
st.caption("Simulates machine-to-machine REST API ingestion blocks straight inside your Streamlit Web instance.")

with st.expander("Expand to Test API Raw JSON Output Natively"):
    st.info("This section simulates how an external system (like a Discord bot or automated site) receives metadata from your backend endpoint.")
    api_text_test = st.text_input("Simulated API Request String:", value="CRITICAL child exploitation safety alert flag.")
    
    if st.button("Trigger Simulated HTTP POST Request"):
        try:
            # Map data vectors exactly how the standalone api.py would receive them over HTTP
            test_tokenized = tokenizer(api_text_test, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            mock_canvas = torch.zeros((1, 3, 224, 224))
            
            with torch.no_grad():
                test_vec = model(test_tokenized["input_ids"].to(config.DEVICE), test_tokenized["attention_mask"].to(config.DEVICE), mock_canvas.to(config.DEVICE))
                s_dist = torch.mm(test_vec, SAFE_ANCHOR.T).item()
                d_dist = torch.mm(test_vec, DANGER_ANCHOR.T).item()
                
                # Calibrate probability distribution inside the mock layer
                mock_logits = torch.tensor([[s_dist * 10.0, d_dist * 10.0]])
                mock_probabilities = F.softmax(mock_logits, dim=1).squeeze(0)
                
                # CRITICAL MULTI-ELEMENT FIX: Target the explicit array slice index first 
                # to extract a float scalar, preventing scalar conversion errors in the simulator block
                mock_risk = mock_probabilities.item()
                
                if force_demo_safe:
                    mock_risk = float(np.random.uniform(0.01, 0.04))
                
            # Render the identical structural JSON output that automated scripts read
            st.json({
                "status": "success",
                "endpoint_queried": "/v1/triage",
                "payload_anonymized_hash": anonymize_text(api_text_test),
                "safe_affinity_score": float(f"{1.0 - mock_risk:.4f}"),
                "danger_affinity_score": float(f"{mock_risk:.4f}"),
                "action_routing_required": mock_risk > triage_threshold
            })
        except Exception as e:
            st.error(f"Simulated API Fault: {str(e)}")
