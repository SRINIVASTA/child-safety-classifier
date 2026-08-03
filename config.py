import torch

class ContentSafetyConfig:
    """Centralized architecture and environment configurations."""
    TEXT_MODEL: str = "distilbert-base-uncased"
    VISION_MODEL: str = "openai/clip-vit-base-patch32"
    TEXT_EMBED_DIM: int = 768
    VISION_EMBED_DIM: int = 512
    PROJECTION_DIM: int = 256
    NUM_CLASSES: int = 2  # 0: Safe, 1: High-Risk/Flagged
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
