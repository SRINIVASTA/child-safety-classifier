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

st.set_page_config(page_title="Thorn // Dynamic Metric Triage Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ Multimodal Content Triage Interface")
st.caption("Senior ML Architecture: Zero-Shot Multi-Modal Embedding Indexing (No Keyword Constraints)")

@st.cache_resource
def load_production_pipeline():
    config = ContentSafetyConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    processor = CLIPProcessor.from_pretrained(config.VISION_MODEL)
    
    # Instantiate the new embedding bridge configuration
    model = MultimodalEmbeddingBridge(config)
    model.to(config.DEVICE)
    model.eval()
    
    # --- SENIOR ML ENGINEERING ANCHORS ---
    # We generate mathematically stable text reference anchors for semantic categorization
    with torch.no_grad():
        # Clean/Safe Anchor Setup
        safe_tokens = tokenizer("safe benign clear clean standard public innocent family domestic", return_tensors="pt", padding=True, truncation=True)
        safe_img_mock = torch.zeros((1, 3, 224, 224)) # Neutral anchor canvas
        safe_out = model.text_backbone(safe_tokens["input_ids"], attention_mask=safe_tokens["attention_mask"]).last_hidden_state[:, 0, :]
        safe_proj = model.text_projection(safe_out)
        vis_mock_proj = model.vision_projection(model.vision_backbone(pixel_values=safe_img_mock).pooler_output)
        safe_anchor = model.unified_space(torch.cat((safe_proj, vis_mock_proj), dim=1))
        safe_anchor = F.normalize(safe_anchor, p=2, dim=1)

        # Danger/Threat Anchor Setup
        danger_tokens = tokenizer("alert critical flagged threat risk dangerous abusive restriction warning violation illegal", return_tensors="pt", padding=True, truncation=True)
        danger_out = model.text_backbone(danger_tokens["input_ids"], attention_mask=danger_tokens["attention_mask"]).last_hidden_state[:, 0, :]
        danger_proj = model.text_projection(danger_out)
        danger_anchor = model.unified_space(torch.cat((danger_proj, vis_mock_proj), dim=1))
        danger_anchor = F.normalize(danger_anchor, p=2, dim=1)

    return config, tokenizer, processor, model, safe_anchor, danger_anchor

config, tokenizer, processor, model, SAFE_ANCHOR, DANGER_ANCHOR = load_production_pipeline()

# Sidebar controls for dynamic threshold tuning
st.sidebar.header("🎛️ Operational Parameters")
triage_threshold = st.sidebar.slider("Alert Sensitivity Cutoff:", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

text_input = st.text_area("Accompanying Text/Metadata Ingestion Stream", placeholder="Type any sentence dynamically (e.g., 'dog in a house', 'critical log summary')...", height=150)
uploaded_file = st.file_uploader("Upload Target Media Payload", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
else:
    # Safe grey fallback vector creation
    image = Image.fromarray((np.ones((300, 400, 3), dtype=np.uint8) * 128))

st.image(image, caption="Current Ingestion Canvas", width=300)
st.write("---")

is_ready = bool(text_input.strip())
if st.button("Execute Vector Ingestion & Triage", type="primary", disabled=not is_ready):
    with st.spinner("Extracting hidden states & mapping metric coordinate calculations..."):
        try:
            # Mask PII records instantly
            logger.info(f"Vector calculation initiated for entry: {anonymize_text(text_input)}")
            
            # Formulate cross-modal inputs
            text_feats = tokenizer(text_input, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            vision_feats = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                # Extract normalized structural latent vector for the unclassified input asset
                target_vector = model(
                    text_feats["input_ids"].to(config.DEVICE),
                    text_feats["attention_mask"].to(config.DEVICE),
                    vision_feats["pixel_values"].to(config.DEVICE)
                )
                
                # Compute exact Cosine Similarities against our functional system anchors
                sim_to_safe = torch.mm(target_vector, SAFE_ANCHOR.T).item()
                sim_to_danger = torch.mm(target_vector, DANGER_ANCHOR.T).item()
                
                # Convert raw unbounded similarities to calibrated probabilities via Softmax Scaling
                raw_logits = torch.tensor([[sim_to_safe * 10.0, sim_to_danger * 10.0]]) # Scale factor for distribution separation
                probabilities = F.softmax(raw_logits, dim=1).squeeze(0)
                
                safe_prob = probabilities[0].item()
                risk_prob = probabilities[1].item()
                
            # Render Dynamic Telemetry Metrics
            st.subheader("Dynamic Model Telemetry Output")
            m1, m2 = st.columns(2)
            m1.metric("Calculated Clear Affinity", f"{safe_prob * 100:.2f}%")
            m2.metric("Calculated Threat Affinity", f"{risk_prob * 100:.2f}%")
            
            if risk_prob > triage_threshold:
                st.error(f"🚨 **HIGH RISK TRIAGE ALERT**: Material significantly aligns with threat space vectors ({risk_prob*100:.1f}% score vs {triage_threshold*100:.0f}% safety allowance limit). Route immediately to priority queues.")
            else:
                st.success("✅ **CLEAR**: Material passed threshold verification limits.")
                
        except Exception as e:
            st.error(f"Inference Graph Runtime Error: {str(e)}")
