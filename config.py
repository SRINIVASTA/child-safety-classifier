import torch

class ContentSafetyConfig:
    """Centralized architecture and environment configurations with aligned embeddings."""
    TEXT_MODEL: str = "distilbert-base-uncased"
    VISION_MODEL: str = "openai/clip-vit-base-patch32"
    
    # Text backbone embeddings (DistilBERT outputs 768)
    TEXT_EMBED_DIM: int = 768
    
    # CORRECTED: CLIP-ViT-Base-Patch32 natively outputs 768, NOT 512.
    # Aligning this prevents the 1x768 and 512x256 matrix mismatch.
    VISION_EMBED_DIM: int = 768
    
    PROJECTION_DIM: int = 256
    NUM_CLASSES: int = 2  # 0: Safe, 1: High-Risk/Flagged
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
