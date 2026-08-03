import torch

class ContentSafetyConfig:
    """Centralized architecture and environment configurations."""
    TEXT_MODEL: str = "distilbert-base-uncased"
    VISION_MODEL: str = "openai/clip-vit-base-patch32"
    TEXT_EMBED_DIM: int = 768
    VISION_EMBED_DIM: int = 768  # Aligned to native CLIP base output dimension
    PROJECTION_DIM: int = 256
    NUM_CLASSES: int = 2  
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
