import pytest
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, CLIPProcessor
from config import ContentSafetyConfig
from model import LateFusionClassifier

@pytest.fixture(scope="session")
def benchmark_assets():
    """Generates standardized text and image payloads for profiling."""
    config = ContentSafetyConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    processor = CLIPProcessor.from_pretrained(config.VISION_MODEL)
    
    model = LateFusionClassifier(config)
    model.to(config.DEVICE)
    model.eval()
    
    sample_text = "Verification payload stream context parsing index standard log."
    raw_canvas = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    sample_image = Image.fromarray(raw_canvas)
    
    return config, tokenizer, processor, model, sample_text, sample_image

def test_text_tokenization_speed(benchmark, benchmark_assets):
    """Profiles text tokenization layer overhead to detect delays."""
    _, tokenizer, _, _, sample_text, _ = benchmark_assets
    
    def run_tokenization():
        return tokenizer(sample_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    
    benchmark(run_tokenization)

def test_image_preprocessing_speed(benchmark, benchmark_assets):
    """Profiles vision resizing and tensor normalization speeds."""
    _, _, processor, _, _, sample_image = benchmark_assets
    
    def run_image_processing():
        return processor(images=sample_image, return_tensors="pt")
        
    benchmark(run_image_processing)

def test_model_inference_latency(benchmark, benchmark_assets):
    """Profiles forward-pass inference runtime to track execution limits."""
    config, tokenizer, processor, model, sample_text, sample_image = benchmark_assets
    
    text_feats = tokenizer(sample_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    vision_feats = processor(images=sample_image, return_tensors="pt")
    
    input_ids = text_feats["input_ids"].to(config.DEVICE)
    attention_mask = text_feats["attention_mask"].to(config.DEVICE)
    pixel_values = vision_feats["pixel_values"].to(config.DEVICE)
    
    def run_inference():
        with torch.no_grad():
            return model(input_ids, attention_mask, pixel_values)
            
    # Warmup loop to stabilize the runtime engine cache
    for _ in range(5):
        _ = run_inference()
        
    benchmark(run_inference)
