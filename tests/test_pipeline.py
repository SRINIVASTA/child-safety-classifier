import pytest
import torch
import numpy as np
from PIL import Image
from config import ContentSafetyConfig
from model import LateFusionClassifier
from transformers import AutoTokenizer, CLIPProcessor
from utils import anonymize_text

@pytest.fixture(scope="session")
def pipeline_components():
    """Initializes and caches model artifacts across execution cycles."""
    config = ContentSafetyConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    processor = CLIPProcessor.from_pretrained(config.VISION_MODEL)
    
    model = LateFusionClassifier(config)
    model.to(config.DEVICE)
    model.eval()
    return config, tokenizer, processor, model

def test_anonymization_engine():
    """Validates text obfuscation engine hides raw input details safely."""
    raw_payload = "Target profile under monitoring user@domain.com on IP 192.168.1.1"
    scrambled_output = anonymize_text(raw_payload)
    
    assert "user@domain.com" not in scrambled_output
    assert "192.168.1.1" not in scrambled_output
    assert scrambled_output.startswith("SHA256_")

def test_synthetic_safe_pipeline_execution(pipeline_components):
    """Verifies pipeline processes standard safe mock scenarios completely."""
    config, tokenizer, processor, model = pipeline_components
    
    text_input = "A family enjoys a sunny afternoon picnic at a public park."
    synthetic_array = np.zeros((224, 224, 3), dtype=np.uint8)
    synthetic_array[:, :, 1] = 180  # Green canvas matrix
    image = Image.fromarray(synthetic_array)
    
    text_feats = tokenizer(text_input, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    vision_feats = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(
            text_feats["input_ids"].to(config.DEVICE),
            text_feats["attention_mask"].to(config.DEVICE),
            vision_feats["pixel_values"].to(config.DEVICE)
        )
    
    assert logits.shape == (1, config.NUM_CLASSES)
